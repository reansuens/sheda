package com.camdist.app

import android.content.Context
import android.net.wifi.WifiManager
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.HttpURLConnection
import java.net.InetSocketAddress
import java.net.URL
import kotlin.concurrent.thread

/**
 * One measured object as the laptop drew it: outline (x0, y0, x1, y1, ... in the sent frame's pixels), label, selected.
 * With the mesh switched on, also its fitted wireframe as segments (x0, y0, x1, y1, ...): the edges facing the camera and the hidden ones.
 */
class Obj(val hull: FloatArray, val text: String, val sel: Boolean, val mesh: FloatArray, val hid: FloatArray)

/** The laptop's latest result for the live view. */
class Res(val w: Int, val h: Int, val objects: List<Obj>, val lines: List<String>)

/**
 * The link to camdist on the laptop (plain HTTP on the local network, port 8080): frames go out as JPEG, the
 * newest result comes back with each answer; taps and button presses go out as small posts. The laptop
 * broadcasts "camdist <port>" on UDP 8081 once a second, which is how this finds it.
 */
class Link(ctx: Context) {
    private val main = Handler(Looper.getMainLooper())
    private val prefs = ctx.getSharedPreferences("camdist", Context.MODE_PRIVATE)
    var host by mutableStateOf(prefs.getString("host", "") ?: "")
    var auto by mutableStateOf(true)
    var res by mutableStateOf<Res?>(null)
    var res2 by mutableStateOf<Res?>(null)  // the laptop's result for the second camera (the ESP32-CAM)
    var note by mutableStateOf("looking for the laptop ...")
    var fps by mutableStateOf(0f)
    var size by mutableStateOf("")
    @Volatile var streaming = true
    private var tPrev = 0L

    init {
        try {
            val wm = ctx.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            wm.createMulticastLock("camdist").apply { setReferenceCounted(false); acquire() }
        } catch (_: Exception) {
        }
        thread(isDaemon = true) { listenForLaptop() }
    }

    private fun listenForLaptop() {
        while (true) {
            try {
                DatagramSocket(null).use { s ->
                    s.reuseAddress = true
                    s.bind(InetSocketAddress(8081))
                    val buf = ByteArray(64)
                    while (true) {
                        val p = DatagramPacket(buf, buf.size)
                        s.receive(p)
                        if (auto && String(p.data, 0, p.length).startsWith("camdist")) {
                            val ip = p.address.hostAddress ?: continue
                            if (ip != host) main.post { host = ip; prefs.edit().putString("host", ip).apply() }
                        }
                    }
                }
            } catch (_: Exception) {
                Thread.sleep(1000)
            }
        }
    }

    fun chooseHost(h: String) {
        auto = h.isBlank()
        host = h.trim()
        prefs.edit().putString("host", host).apply()
    }

    private fun open(path: String, timeout: Int): HttpURLConnection {
        val c = URL("http://$host:8080$path").openConnection() as HttpURLConnection
        c.requestMethod = "POST"
        c.doOutput = true
        c.connectTimeout = 1500
        c.readTimeout = timeout
        return c
    }

    /** Called on the analysis thread; returns when the laptop has answered (so frames never pile up). */
    fun sendFrame(jpeg: ByteArray, w: Int, h: Int) {
        if (host.isEmpty()) return
        try {
            val c = open("/frame", 5000)
            c.setFixedLengthStreamingMode(jpeg.size)
            c.outputStream.use { it.write(jpeg) }
            val txt = c.inputStream.bufferedReader().use { it.readText() }
            val r = parse(txt)
            val now = System.nanoTime()
            val inst = if (tPrev > 0) 1e9f / (now - tPrev) else 0f
            tPrev = now
            main.post {
                res = r
                note = ""
                size = "${w}x$h"
                fps = 0.8f * fps + 0.2f * inst
            }
        } catch (e: Exception) {
            main.post { note = "laptop not reachable ($host): is camdist running?" }
            Thread.sleep(500)
        }
    }

    /** The second camera's frame (from the ESP32-CAM on the cable); the answer is that camera's result. */
    fun sendFrame2(jpeg: ByteArray, w: Int, h: Int) {
        if (host.isEmpty()) return
        try {
            val c = open("/frame2", 5000)
            c.setFixedLengthStreamingMode(jpeg.size)
            c.outputStream.use { it.write(jpeg) }
            val r = parse(c.inputStream.bufferedReader().use { it.readText() })
            main.post { res2 = r }
        } catch (e: Exception) {
            Thread.sleep(500)
        }
    }

    private fun parse(txt: String): Res? {
        val r = JSONObject(txt).optJSONObject("res") ?: return null
        val objs = r.getJSONArray("objects")
        val list = ArrayList<Obj>()
        for (i in 0 until objs.length()) {
            val o = objs.getJSONObject(i)
            val hp = o.getJSONArray("hull")
            val pts = FloatArray(hp.length() * 2)
            for (j in 0 until hp.length()) {
                pts[2 * j] = hp.getJSONArray(j).getDouble(0).toFloat()
                pts[2 * j + 1] = hp.getJSONArray(j).getDouble(1).toFloat()
            }
            list.add(Obj(pts, o.getString("text"), o.getBoolean("sel"), segments(o.optJSONArray("mesh")), segments(o.optJSONArray("hid"))))
        }
        val ls = r.getJSONArray("lines")
        return Res(r.getInt("w"), r.getInt("h"), list, List(ls.length()) { ls.getString(it) })
    }

    private fun segments(a: org.json.JSONArray?): FloatArray {
        if (a == null) return FloatArray(0)
        val f = FloatArray(a.length() * 4)
        for (i in 0 until a.length()) {
            val s = a.getJSONArray(i)
            for (k in 0 until 4) f[4 * i + k] = s.getDouble(k).toFloat()
        }
        return f
    }

    private fun post(path: String, body: String) {
        if (host.isEmpty()) return
        thread {
            try {
                val c = open(path, 3000)
                c.outputStream.use { it.write(body.toByteArray()) }
                c.inputStream.close()
            } catch (_: Exception) {
            }
        }
    }

    fun tap(x: Float, y: Float, cam: Int = 1) = post("/tap", """{"x":$x,"y":$y,"cam":$cam}""")
    fun cmd(name: String) = post("/cmd", """{"cmd":"$name"}""")
}
