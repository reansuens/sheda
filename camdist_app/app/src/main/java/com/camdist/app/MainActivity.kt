package com.camdist.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.util.Size
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import kotlin.math.min

/**
 * One link to the laptop and one ESP32-CAM reader for the whole process. An activity that is recreated must not start a
 * second reader on the same USB port: two readers fight over the interface and each gets only some of the bytes.
 */
object Shared {
    private var link: Link? = null
    private var esp: EspCam? = null

    @Synchronized fun link(ctx: Context): Link = link ?: Link(ctx.applicationContext).also { link = it }

    @Synchronized fun esp(ctx: Context): EspCam =
        esp ?: EspCam(ctx.applicationContext, link(ctx)).also { esp = it; it.start() }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { MaterialTheme(colorScheme = darkColorScheme()) { App() } }
    }
}

@Composable
fun App() {
    val ctx = LocalContext.current
    fun ok() = ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    var granted by remember { mutableStateOf(ok()) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = ok() }
    LaunchedEffect(Unit) { if (!granted) launcher.launch(Manifest.permission.CAMERA) }
    Surface(color = Color.Black, modifier = Modifier.fillMaxSize()) {
        if (granted) CamScreen() else Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) { Text("Allow the camera") }
        }
    }
}

@Composable
fun CamScreen() {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val link = remember { Shared.link(ctx) }
    val esp = remember { Shared.esp(ctx) }
    val exec = remember { Executors.newSingleThreadExecutor() }
    val previewView = remember {
        PreviewView(ctx).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    var streamingUi by remember { mutableStateOf(true) }
    var editHost by remember { mutableStateOf(false) }

    LaunchedEffect(previewView) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val sel = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(1600, 1200), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val preview = Preview.Builder().setResolutionSelector(sel).build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder().setResolutionSelector(sel)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            var lastRot = -1
            analysis.setAnalyzer(exec) { img ->
                try {
                    // frames upright as the screen shows them: follow the display (it was portrait when this was bound once)
                    val dr = previewView.display?.rotation ?: lastRot
                    if (dr != lastRot) {
                        lastRot = dr
                        analysis.targetRotation = dr
                        android.util.Log.i("camdist", "display rotation $dr, frame rotation ${img.imageInfo.rotationDegrees}")
                    }
                    if (link.streaming && link.host.isNotEmpty()) {
                        var bmp = img.toBitmap()
                        val rot = img.imageInfo.rotationDegrees
                        if (rot != 0) {
                            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(rot.toFloat()) }, true)
                        }
                        val out = ByteArrayOutputStream()
                        bmp.compress(Bitmap.CompressFormat.JPEG, 88, out)
                        link.sendFrame(out.toByteArray(), bmp.width, bmp.height)
                    }
                } finally {
                    img.close()
                }
            }
            provider.unbindAll()
            provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(ctx))
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        // split view when the ESP32-CAM streams: the phone's camera to the left, the ESP32-CAM to the right
        Row(Modifier.fillMaxSize()) {
            Box(Modifier.weight(1f).fillMaxHeight()) {
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
                Overlay(link, 1)
            }
            val bmp = esp.frame
            if (bmp != null) {
                Box(Modifier.weight(1f).fillMaxHeight()) {
                    val img = remember(bmp) { bmp.asImageBitmap() }
                    Image(img, contentDescription = null, contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize())
                    Overlay(link, 2)
                }
            }
        }

        // status
        Column(Modifier.align(Alignment.TopStart).padding(8.dp).background(Color(0x99000000)).padding(6.dp)) {
            Text(
                (if (link.host.isEmpty()) "looking for the laptop ..." else "laptop ${link.host}") +
                    "   ${link.size}   ${"%.1f".format(link.fps)} fps",
                color = Color.White, fontSize = 12.sp,
                modifier = Modifier.clickable { editHost = true },
            )
            if (link.note.isNotEmpty()) Text(link.note, color = Color(0xFFFFB070), fontSize = 12.sp)
            Text(esp.status, color = Color(0xFFB0E0B0), fontSize = 12.sp)
            link.res?.lines?.forEach { Text(it, color = Color(0xFFCCDDEE), fontSize = 11.sp) }
        }

        // buttons: the same as the keys on the laptop
        Column(Modifier.align(Alignment.CenterEnd).fillMaxHeight().verticalScroll(rememberScrollState()).padding(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            val pad = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
            Button(
                onClick = { streamingUi = !streamingUi; link.streaming = streamingUi },
                contentPadding = pad,
                colors = ButtonDefaults.buttonColors(containerColor = if (streamingUi) Color(0xFF0A7D5A) else Color(0xFF444B55)),
            ) { Text(if (streamingUi) "Stream ON" else "Stream OFF") }
            Button(onClick = { esp.connect() }, contentPadding = pad) { Text("ESP connect") }
            Button(onClick = { esp.turn() }, contentPadding = pad) { Text("ESP turn ${esp.rot}") }
            Button(onClick = { esp.toggleMirror() }, contentPadding = pad) { Text(if (esp.mirror) "ESP mirror ON" else "ESP mirror OFF") }
            Button(
                onClick = { esp.toggleLight() },
                contentPadding = pad,
                colors = ButtonDefaults.buttonColors(containerColor = if (esp.light) Color(0xFFB58A00) else Color(0xFF444B55)),
            ) { Text(if (esp.light) "ESP light ON" else "ESP light") }
            for ((label, cmd) in listOf("Log" to "log", "Mesh" to "mesh", "Report" to "report", "CAD" to "cad", "Clear" to "clear", "Reset" to "reset", "Shape" to "shape")) {
                Button(onClick = { link.cmd(cmd) }, contentPadding = pad) { Text(label) }
            }
        }
    }

    if (editHost) {
        var text by remember { mutableStateOf(link.host) }
        AlertDialog(
            onDismissRequest = { editHost = false },
            confirmButton = { TextButton(onClick = { link.chooseHost(text); editHost = false }) { Text("OK") } },
            title = { Text("Laptop address") },
            text = {
                Column {
                    Text("Leave empty to find the laptop automatically.", fontSize = 12.sp)
                    OutlinedTextField(value = text, onValueChange = { text = it }, singleLine = true, label = { Text("192.168.x.x") })
                }
            },
        )
    }
}

/** The outlines the laptop measured over one camera's picture; a tap selects (or, on empty space, clears) an object. */
@Composable
fun Overlay(link: Link, cam: Int) {
    val res = if (cam == 1) link.res else link.res2
    Canvas(
        Modifier.fillMaxSize().pointerInput(cam) {
            detectTapGestures { o ->
                val r = if (cam == 1) link.res else link.res2
                val iw = (r?.w ?: 4).toFloat()
                val ih = (r?.h ?: 3).toFloat()
                val s = min(size.width / iw, size.height / ih)
                val x = (o.x - (size.width - iw * s) / 2) / (iw * s)
                val y = (o.y - (size.height - ih * s) / 2) / (ih * s)
                if (x in 0f..1f && y in 0f..1f) link.tap(x, y, cam)
            }
        }
    ) {
        val r = res ?: return@Canvas
        val s = min(size.width / r.w, size.height / r.h)
        val dx = (size.width - r.w * s) / 2
        val dy = (size.height - r.h * s) / 2
        val anySel = r.objects.any { it.sel }
        val paint = Paint().apply { textSize = 15.sp.toPx(); isAntiAlias = true; isFakeBoldText = true }
        val back = Paint().apply { textSize = paint.textSize; style = Paint.Style.STROKE; strokeWidth = 5f; color = android.graphics.Color.BLACK; isFakeBoldText = true }
        paint.color = android.graphics.Color.WHITE
        for (o in r.objects) {
            val path = Path()
            for (i in 0 until o.hull.size / 2) {
                val px = dx + o.hull[2 * i] * s
                val py = dy + o.hull[2 * i + 1] * s
                if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
            }
            path.close()
            if (o.sel) drawPath(path, Color(0x40FFE000), style = Fill)
            drawPath(path, if (o.sel) Color(0xFFFFE000) else Color(0xAAAABBCC), style = Stroke(width = if (o.sel) 6f else 2.5f))
            for (k in 0 until o.hid.size / 4) {  // the wireframe, when the Mesh button is on: hidden edges dim, the ones facing the camera bright
                drawLine(Color(0x99FF7A59), Offset(dx + o.hid[4 * k] * s, dy + o.hid[4 * k + 1] * s), Offset(dx + o.hid[4 * k + 2] * s, dy + o.hid[4 * k + 3] * s), 1.5f)
            }
            for (k in 0 until o.mesh.size / 4) {
                drawLine(Color(0xFF00F0B0), Offset(dx + o.mesh[4 * k] * s, dy + o.mesh[4 * k + 1] * s), Offset(dx + o.mesh[4 * k + 2] * s, dy + o.mesh[4 * k + 3] * s), 2f)
            }
            if (o.sel || !anySel) {
                var x0 = Float.MAX_VALUE
                var y0 = Float.MAX_VALUE
                for (i in 0 until o.hull.size / 2) {
                    x0 = min(x0, dx + o.hull[2 * i] * s)
                    y0 = min(y0, dy + o.hull[2 * i + 1] * s)
                }
                val ty = (y0 - 8f).coerceAtLeast(paint.textSize)
                drawContext.canvas.nativeCanvas.drawText(o.text, x0, ty, back)
                drawContext.canvas.nativeCanvas.drawText(o.text, x0, ty, paint)
            }
        }
    }
}
