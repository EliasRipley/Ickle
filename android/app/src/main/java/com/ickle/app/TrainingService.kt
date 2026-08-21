package com.ickle.app

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import androidx.core.app.NotificationCompat
import com.chaquo.python.Python

class TrainingService : Service() {

    private val handler = Handler(Looper.getMainLooper())
    private var running = false
    private var serverUrl = ""
    private var dataDir = ""
    private lateinit var _python: Python
    private var _client: com.chaquo.python.PyObject? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        _python = Python.getInstance()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        serverUrl = intent?.getStringExtra("server_url") ?: "http://127.0.0.1:8788"
        dataDir = intent?.getStringExtra("data_dir") ?: (filesDir.absolutePath + "/ickle_data")

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Ickle Training")
            .setContentText("Contributing compute to the Ickle network")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentIntent(PendingIntent.getActivity(
                this, 0, Intent(this, MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or (if (Build.VERSION.SDK_INT >= 23) PendingIntent.FLAG_IMMUTABLE else 0)
            ))
            .build()

        startForeground(1, notification)

        if (!running) {
            running = true
            startTrainingLoop()
        }

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        running = false
        handler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    private fun startTrainingLoop() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                if (!running) return
                Thread {
                    try {
                        val client = getClient()
                        val result = client.callAttr("do_round")
                        val roundId = result?.get("round_id")?.toInt() ?: 0
                        val metrics = result?.get("metrics")
                        val finalLoss = metrics?.get("final_loss")?.toDouble()
                        val tokenCount = metrics?.get("token_count")?.toInt()
                        recordRoundStats(roundId, finalLoss, tokenCount)
                        updateNotification("Round $roundId complete")
                    } catch (_: Exception) {
                        updateNotification("Training step failed, retrying...")
                    }
                }.start()
                handler.postDelayed(this, 30_000)
            }
        }, 1_000)
    }

    // MainActivity has no direct binding to this foreground service, so the
    // stats view was previously findViewById'd and never touched again --
    // real per-round metrics (do_round() computes final_loss/token_count)
    // never reached the UI. SharedPreferences is the simplest bridge that
    // matches this app's existing polling style rather than adding a
    // broadcast/binder mechanism for a single value pair.
    private fun recordRoundStats(roundId: Int, finalLoss: Double?, tokenCount: Int?) {
        val prefs = getSharedPreferences(STATS_PREFS, MODE_PRIVATE)
        prefs.edit()
            .putInt("last_round_id", roundId)
            .putFloat("last_final_loss", (finalLoss ?: -1.0).toFloat())
            .putInt("last_token_count", tokenCount ?: 0)
            .putLong("last_round_at", System.currentTimeMillis())
            .apply()
    }

    private fun getClient(): com.chaquo.python.PyObject {
        if (_client == null) {
            _client = _python.getModule("ickle_mobile").callAttr(
                "IckleMobileClient", serverUrl, dataDir, "android", "nano"
            )
        }
        return _client!!
    }

    private fun updateNotification(text: String) {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Ickle Training")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentIntent(PendingIntent.getActivity(
                this, 0, Intent(this, MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or (if (Build.VERSION.SDK_INT >= 23) PendingIntent.FLAG_IMMUTABLE else 0)
            ))
            .build()
        startForeground(1, notification)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "Ickle Training", NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    companion object {
        private const val CHANNEL_ID = "ickle_training"
        const val STATS_PREFS = "ickle_training_stats"
    }
}
