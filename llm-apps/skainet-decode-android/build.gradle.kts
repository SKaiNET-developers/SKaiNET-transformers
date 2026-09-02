import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// The Android activity leg of skainet-decode (SKaiNET#1244): load a GGUF, decode, and report
// the same GenerationMetrics as the JVM CLI — on the platform the 2 GB memory arc actually
// targets. The rows the JVM leg cannot show (page-fault rate for mapped weights, RSS on a
// constrained device) are the whole point; the decode flow itself lives in
// :llm-apps:skainet-decode-core so the two legs cannot diverge.
plugins {
    // AGP 9 ships built-in Kotlin support — applying kotlin("android") is an error since 9.0
    // (see :llm-runtime:iree-android, the precedent module). First com.android.application in
    // the repo; not published, so no maven-publish and invisible to bom-coverage by design.
    alias(libs.plugins.androidApplication)
}

android {
    namespace = "sk.ainet.apps.decode"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "sk.ainet.apps.decode"
        minSdk = libs.versions.android.minSdk.get().toInt()
        targetSdk = libs.versions.android.compileSdk.get().toInt()
        versionCode = 1
        versionName = "0.1"

        ndk {
            // Real numbers come from arm64 hardware; x86_64 keeps the emulator lane runnable.
            // Matches the engine's skainet-backend-jni-cpu AAR ABI set.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    packaging {
        resources {
            // The 0.52.0 self-healing kernel dispatch is ServiceLoader-driven
            // (SKaiNET#1240): losing META-INF/services entries silently drops every
            // matmul to the slow decoding reference kernel. Merge, never exclude.
            merges += "META-INF/services/**"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
            optIn.add("sk.ainet.lang.memory.ExperimentalMemoryApi")
        }
    }
}

dependencies {
    implementation(platform(project(":llm-bom")))
    implementation(project(":llm-apps:skainet-decode-core"))
    // AndroidGguf (device-fit pre-flight, mapped loader) and MappedRandomAccessSource.
    implementation(libs.skainet.io.gguf)
    implementation(libs.skainet.io.core)
    // AndroidTraceSink (Perfetto spans), MemoryProbe, GenerationMetrics.
    implementation(libs.skainet.lang.core)
    implementation(libs.kotlinx.coroutines)
    // The NEON kernel AAR; registers itself via ServiceLoader on ART — no bootstrap code.
    // skainet-backend-native-cpu (FFM) must NOT appear here: FFM does not exist on ART.
    runtimeOnly(libs.skainet.backend.jniCpu)
}
