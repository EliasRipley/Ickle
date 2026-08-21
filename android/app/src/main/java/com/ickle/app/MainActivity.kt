package com.ickle.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {

    private lateinit var serverUrlInput: EditText
    private lateinit var registerButton: Button
    private lateinit var startStopButton: Button
    private lateinit var statusText: TextView
    private lateinit var statsText: TextView

    private var trainingActive = false
    private val statsHandler = Handler(Looper.getMainLooper())
    private val statsRefresher = object : Runnable {
        override fun run() {
            refreshStatsDisplay()
            statsHandler.postDelayed(this, 15_000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        serverUrlInput = findViewById(R.id.server_url)
        registerButton = findViewById(R.id.register_button)
        startStopButton = findViewById(R.id.start_stop_button)
        statusText = findViewById(R.id.status_text)
        statsText = findViewById(R.id.stats_text)

        registerButton.setOnClickListener { register() }
        startStopButton.setOnClickListener { toggleTraining() }

        requestNotificationPermission()
    }

    override fun onResume() {
        super.onResume()
        statsHandler.post(statsRefresher)
    }

    override fun onPause() {
        statsHandler.removeCallbacks(statsRefresher)
        super.onPause()
    }

    private fun refreshStatsDisplay() {
        val prefs = getSharedPreferences(TrainingService.STATS_PREFS, MODE_PRIVATE)
        val roundId = prefs.getInt("last_round_id", -1)
        if (roundId < 0) {
            statsText.text = "No training rounds completed yet."
            return
        }
        val loss = prefs.getFloat("last_final_loss", -1f)
        val tokenCount = prefs.getInt("last_token_count", 0)
        val lossText = if (loss >= 0f) String.format("%.4f", loss) else "n/a"
        statsText.text = "Round $roundId -- loss $lossText over $tokenCount tokens"
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    1001
                )
            }
        }
    }

    private fun register() {
        val serverUrl = serverUrlInput.text.toString().trim()
        if (serverUrl.isEmpty()) {
            statusText.text = "Enter server URL"
            return
        }

        Thread {
            try {
                val py = Python.getInstance()
                val client = py.getModule("ickle_mobile").callAttr(
                    "IckleMobileClient", serverUrl,
                    filesDir.absolutePath + "/ickle_data", "android", "nano"
                )
                client.callAttr("register")
                runOnUiThread { statusText.text = "Registered" }
            } catch (e: Exception) {
                runOnUiThread { statusText.text = "Error: ${e.message}" }
            }
        }.start()
    }

    private fun toggleTraining() {
        if (trainingActive) {
            stopTraining()
        } else {
            startTraining()
        }
    }

    private fun startTraining() {
        trainingActive = true
        startStopButton.text = "Stop Training"
        statusText.text = "Training..."

        val intent = Intent(this, TrainingService::class.java)
        intent.putExtra("server_url", serverUrlInput.text.toString().trim())
        intent.putExtra("data_dir", filesDir.absolutePath + "/ickle_data")
        ContextCompat.startForegroundService(this, intent)
    }

    private fun stopTraining() {
        trainingActive = false
        startStopButton.text = "Start Training"
        statusText.text = "Idle"
        stopService(Intent(this, TrainingService::class.java))
    }
}
