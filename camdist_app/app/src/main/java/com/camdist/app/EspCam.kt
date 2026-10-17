package com.camdist.app

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbManager
import android.hardware.usb.UsbRequest
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import com.hoho.android.usbserial.driver.FtdiSerialDriver
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.TimeoutException
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/**
 * The second camera of the rig: an ESP32-CAM (SerialCam sketch) plugged into the phone's USB-C port. Its JPEG
 * packets (FF AA 55, type, uint32 length, payload; type 1 = a frame) are read over USB serial at 1 Mbaud, shown
 * in the right half of the screen and forwarded to the laptop as the second stream (/frame2). The phone also
 * powers the board.
 */
class EspCam(private val ctx: Context, private val link: Link) {
    private val main = Handler(Looper.getMainLooper())
    private val prefs = ctx.getSharedPreferences("camdist", Context.MODE_PRIVATE)
    var status by mutableStateOf("ESP32-CAM: plug it into the USB-C cable")
    var frame by mutableStateOf<Bitmap?>(null)
    var fps by mutableStateOf(0f)
    var rot by mutableStateOf(prefs.getInt("esp_rot", 0))
    @Volatile private var openedAt = 0L
    @Volatile private var kick = false  // the Connect button: drop the port and start again
    @Volatile private var rx = 0L
    @Volatile private var frames = 0
    @Volatile private var bad = 0
    @Volatile private var skipped = 0
    private val action = "com.camdist.app.USB_PERMISSION"
    private val slot = AtomicReference<Triple<ByteArray, Int, Int>?>(null)
    private val wake = Semaphore(0)

    fun connect() {
        if (System.currentTimeMillis() - openedAt < 25_000 && frame == null) {  // taps while it is still connecting would only restart it
            android.util.Log.i("camdist-esp", "connect tapped while still trying: ignored")
            return
        }
        kick = true
        say("ESP32-CAM: connecting ...")
    }

    var mirror by mutableStateOf(prefs.getBoolean("esp_mirror", true))  // the board's picture is mirrored: a mirror cannot be matched to the phone's room

    fun toggleMirror() {
        mirror = !mirror
        prefs.edit().putBoolean("esp_mirror", mirror).apply()
    }

    var light by mutableStateOf(false)  // the board's white flash LED
    @Volatile private var curPort: UsbSerialPort? = null

    fun toggleLight() {
        light = !light
        sendLight()
    }

    private fun sendLight() {
        val port = curPort ?: return
        val cmd = "led=${if (light) 1 else 0}\nstatus\n".toByteArray()
        thread(isDaemon = true) { try { port.write(cmd, 500) } catch (_: Exception) {} }
    }

    fun turn() {
        rot = (rot + 90) % 360
        prefs.edit().putInt("esp_rot", rot).apply()
    }

    fun start() {
        ContextCompat.registerReceiver(ctx, object : BroadcastReceiver() {
            override fun onReceive(c: Context?, i: Intent?) {}
        }, IntentFilter(action), ContextCompat.RECEIVER_NOT_EXPORTED)
        thread(isDaemon = true) { sender() }
        thread(isDaemon = true) { connectLoop() }
    }

    private fun say(s: String) {
        android.util.Log.i("camdist-esp", s)
        main.post { status = s }
    }

    private fun sender() {
        while (true) {
            wake.acquire()
            val t = slot.getAndSet(null) ?: continue
            link.sendFrame2(t.first, t.second, t.third)
        }
    }

    private fun connectLoop() {
        val mgr = ctx.getSystemService(Context.USB_SERVICE) as UsbManager
        while (true) {
            try {
                val drv = UsbSerialProber.getDefaultProber().findAllDrivers(mgr).firstOrNull()
                if (drv == null) {
                    say("ESP32-CAM: no USB serial device found (${mgr.deviceList.size} USB device(s) seen: " +
                        mgr.deviceList.values.joinToString { "%04x:%04x".format(it.vendorId, it.productId) } + ")")
                } else if (!mgr.hasUsbPermission(drv)) {
                    say("ESP32-CAM: allow USB access in the dialog")
                    val pi = PendingIntent.getBroadcast(ctx, 0, Intent(action).setPackage(ctx.packageName), PendingIntent.FLAG_MUTABLE)
                    mgr.requestPermission(drv.device, pi)
                    pause(3000)
                } else {
                    val conn = mgr.openDevice(drv.device)
                    if (conn == null) {
                        say("ESP32-CAM: cannot open the USB device")
                    } else {
                        val port = drv.ports[0]
                        port.open(conn)
                        openedAt = System.currentTimeMillis()
                        port.setParameters(1_000_000, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
                        releaseLines(port)  // as on the laptop: the board already runs from the phone's power, do not reset it
                        say("ESP32-CAM: waiting for the first frame ...")
                        curPort = port
                        if (light) sendLight()
                        try {
                            readLoop(port, conn)
                        } finally {
                            curPort = null
                            try { port.close() } catch (_: Exception) {}
                            main.post { frame = null; fps = 0f }
                        }
                    }
                }
            } catch (e: Exception) {
                say("ESP32-CAM: ${e.javaClass.simpleName} ${e.message ?: ""}".take(80))
            }
            pause(1000)
        }
    }

    private fun pause(ms: Int) {
        var t = 0
        while (t < ms && !kick) { Thread.sleep(50); t += 50 }
        kick = false
    }

    /** DTR/RTS are wired to the board's boot and reset pins: keep both released, which is how the laptop opens the port. */
    private fun releaseLines(port: UsbSerialPort) {
        try {
            port.dtr = false
            port.rts = false
        } catch (e: Exception) {
            android.util.Log.i("camdist-esp", "control lines not available: ${e.message}")
        }
    }

    /** DTR/RTS are wired to the board's boot and reset pins: boot pin released, then a short reset pulse, so the sketch starts. */
    private fun resetBoard(port: UsbSerialPort) {
        try {
            port.dtr = false
            port.rts = true
            Thread.sleep(120)
            port.rts = false
            android.util.Log.i("camdist-esp", "board reset pulse sent (control lines: ${port.supportedControlLines})")
        } catch (e: Exception) {
            android.util.Log.i("camdist-esp", "control lines not available: ${e.message}")
        }
    }

    private fun UsbManager.hasUsbPermission(d: com.hoho.android.usbserial.driver.UsbSerialDriver) = hasPermission(d.device)

    /**
     * The USB reads, kept queued: the CH340 has no flow control and holds only a few bytes, so a read that is not queued for
     * even a millisecond loses some (the library's own read queues one request at a time and about every frame lost a few
     * hundred bytes). Several requests stay outstanding, as the Linux driver does, and each finished one is queued again.
     */
    private class UsbPipe(private val conn: UsbDeviceConnection, ep: UsbEndpoint, n: Int = 8, size: Int = 4096) {
        private val bufs = HashMap<UsbRequest, ByteBuffer>()

        init {
            repeat(n) {
                val rq = UsbRequest()
                rq.initialize(conn, ep)
                val b = ByteBuffer.allocate(size)
                bufs[rq] = b
                rq.queue(b)
            }
        }

        /** The bytes of the next finished transfer, or null when nothing came within the timeout. */
        fun read(timeoutMs: Long): ByteArray? {
            val rq = try { conn.requestWait(timeoutMs) } catch (_: TimeoutException) { return null } ?: return null
            val b = bufs[rq] ?: return null
            val out = b.array().copyOf(b.position())
            b.clear()
            rq.queue(b)
            return out
        }

        fun close() {
            for (rq in bufs.keys) { try { rq.cancel(); rq.close() } catch (_: Exception) {} }
        }
    }

    /** Only reads, nothing else: decoding happens on another thread. */
    private fun readLoop(port: UsbSerialPort, conn: UsbDeviceConnection) {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO)
        val q = LinkedBlockingQueue<ByteArray>(512)
        val alive = java.util.concurrent.atomic.AtomicBoolean(true)
        val parser = thread(isDaemon = true) { parseLoop(q) { alive.get() } }
        // FTDI adapters put two status bytes in front of every packet, which only the library strips: it reads for them
        val pipe = if (port.driver is FtdiSerialDriver) null else UsbPipe(conn, port.readEndpoint)
        val buf = ByteArray(4096)  // a 16 KB buffer makes the CH340 driver's read return nothing on this phone (4 KB and below give ~100 KB/s)
        var lastData = System.currentTimeMillis()
        var lastSay = 0L
        var lastStat = System.currentTimeMillis()
        var silent = 0
        rx = 0; frames = 0; bad = 0; skipped = 0
        try {
            while (true) {
                if (kick) { kick = false; return }
                val data = if (pipe != null) pipe.read(100) else port.read(buf, 100).let { if (it > 0) buf.copyOf(it) else null }
                if (frame == null && silent == 0 && System.currentTimeMillis() - lastSay > 1000) {
                    lastSay = System.currentTimeMillis()
                    say("ESP32-CAM: ${rx / 1024} KB received, $frames frames, $bad incomplete, $skipped resyncs")
                }
                if (System.currentTimeMillis() - lastStat > 5000) {  // the rates, for `adb logcat -s camdist-esp`
                    lastStat = System.currentTimeMillis()
                    android.util.Log.i("camdist-esp", "rx ${rx / 1024} KB, $frames frames, $bad incomplete, $skipped resyncs, queue ${q.size}")
                }
                if (data != null && data.isNotEmpty()) {
                    rx += data.size
                    lastData = System.currentTimeMillis()
                    if (!q.offer(data)) { q.poll(); q.offer(data) }
                } else if (System.currentTimeMillis() - lastData > 3000) {
                    silent++
                    if (silent <= 2) {  // nothing yet: ask for the settings, then one reset pulse
                        say("ESP32-CAM: no data, ${if (silent == 1) "asking for its settings" else "resetting the board"} ($silent)")
                        if (silent == 2) resetBoard(port)
                        try { port.write("status\n".toByteArray(), 500) } catch (_: Exception) {}
                    } else {
                        say("ESP32-CAM: no data from the board (power? is the SerialCam sketch flashed?)")
                    }
                    lastData = System.currentTimeMillis()
                }
            }
        } finally {
            pipe?.close()
            alive.set(false)
            parser.interrupt()
        }
    }

    private fun parseLoop(q: LinkedBlockingQueue<ByteArray>, alive: () -> Boolean) {
        val store = ByteArray(600_000)
        var n = 0
        var tPrev = 0L
        while (alive()) {
            val chunk = try { q.poll(200, java.util.concurrent.TimeUnit.MILLISECONDS) } catch (_: InterruptedException) { return } ?: continue
            if (n + chunk.size > store.size) n = 0  // far behind or garbage: start over
            System.arraycopy(chunk, 0, store, n, chunk.size)
            n += chunk.size
            while (true) {  // packets: FF AA 55 type len(4, little endian) payload
                var i = 0
                while (i + 2 < n && !(store[i] == 0xFF.toByte() && store[i + 1] == 0xAA.toByte() && store[i + 2] == 0x55.toByte())) i++
                if (i > 0) {
                    if (i + 2 >= n) {  // no header in sight: keep the last two bytes, they may start one
                        System.arraycopy(store, n - 2, store, 0, 2)
                        n = 2
                    } else {
                        System.arraycopy(store, i, store, 0, n - i)
                        n -= i
                    }
                }
                if (n < 8) break
                val type = store[3].toInt()
                val len = (store[4].toInt() and 0xFF) or ((store[5].toInt() and 0xFF) shl 8) or
                    ((store[6].toInt() and 0xFF) shl 16) or ((store[7].toInt() and 0xFF) shl 24)
                if (len < 0 || len > 400_000) { skipped++; System.arraycopy(store, 1, store, 0, n - 1); n -= 1; continue }
                if (n < 8 + len) break
                if (type == 1) {
                    val jpeg = store.copyOfRange(8, 8 + len)
                    val now = System.nanoTime()
                    val inst = if (tPrev > 0) 1e9f / (now - tPrev) else 0f
                    tPrev = now
                    onJpeg(jpeg, inst)
                } else if (type == 2) {
                    android.util.Log.i("camdist-esp", "board says " + String(store, 8, len, Charsets.UTF_8).take(300))
                }
                System.arraycopy(store, 8 + len, store, 0, n - 8 - len)
                n -= 8 + len
            }
        }
    }

    /** A whole frame ends with the JPEG end marker (FF D9), maybe followed by a few padding bytes. */
    private fun complete(j: ByteArray): Boolean {
        for (k in j.size - 2 downTo maxOf(2, j.size - 16)) if (j[k] == 0xFF.toByte() && j[k + 1] == 0xD9.toByte()) return true
        return false
    }

    private fun onJpeg(jpeg: ByteArray, inst: Float) {
        if (jpeg.size < 4 || jpeg[0] != 0xFF.toByte() || jpeg[1] != 0xD8.toByte() || !complete(jpeg)) {  // bytes were lost
            if (bad++ < 6) android.util.Log.i("camdist-esp", "bad frame, ${jpeg.size} bytes, head " + jpeg.take(4).joinToString(" ") { "%02x".format(it) } +
                ", tail " + jpeg.takeLast(14).joinToString(" ") { "%02x".format(it) })
            return
        }
        var bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return  // a frame with lost bytes
        var out = jpeg
        val deg = rot
        if (deg != 0 || mirror) {  // the board may be clamped any way up (and is mirrored): the laptop gets it upright
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height,
                Matrix().apply { if (mirror) postScale(-1f, 1f); postRotate(deg.toFloat()) }, true)
            val o = ByteArrayOutputStream()
            bmp.compress(Bitmap.CompressFormat.JPEG, 90, o)
            out = o.toByteArray()
        }
        if (frames++ == 0) android.util.Log.i("camdist-esp", "first frame ${bmp.width}x${bmp.height}")
        val shown = bmp
        main.post {
            frame = shown
            fps = 0.8f * fps + 0.2f * inst
            status = "ESP32-CAM ${shown.width}x${shown.height}  ${"%.1f".format(fps)} fps"
        }
        if (link.streaming) {
            slot.set(Triple(out, bmp.width, bmp.height))
            wake.release()
        }
    }
}
