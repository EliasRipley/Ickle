import java.io.FileInputStream
import java.util.Properties

plugins {
    id("com.android.application")
    id("com.chaquo.python")
}

// Machine-specific Python interpreter for Chaquopy's build-time pip install
// step. Was previously hardcoded to one contributor's local path, which broke
// the build for anyone else and leaked their username into tracked source.
// Override via android/local.properties (gitignored, the standard place for
// machine-specific Android config -- add a line like
// `chaquopy.pythonExecutable=C:/Path/To/python.exe`) or the
// ICKLE_ANDROID_PYTHON env var. Falls back to bare `python` on PATH, which is
// what Chaquopy would need anyway if neither override is set.
val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localProperties.load(FileInputStream(localPropertiesFile))
}
val chaquopyPythonExecutable: String =
    (localProperties.getProperty("chaquopy.pythonExecutable")
        ?: System.getenv("ICKLE_ANDROID_PYTHON")
        ?: "python")

android {
    namespace = "com.ickle.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.ickle.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }

        python {
            buildPython(chaquopyPythonExecutable)
            pip {
                install("numpy")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
}
