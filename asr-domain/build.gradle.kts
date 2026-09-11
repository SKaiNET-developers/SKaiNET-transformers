import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Generic ASR task types (Transcription, DecodingOptions, FeatureFrames) -- moved up from the
// asr-whisper-iree-cartridge downstream repo (cartridges/dtag/asr-whisper-iree-cartridge's
// asr-domain module), where they lived only because that's where the original whisper-cli
// extraction happened to put them, not because they're Whisper-specific. Both the Whisper and
// Moonshine cartridge families depend on these; keeping them in one cartridge family's repo made
// the other family structurally dependent on it for generic plumbing. Deliberately
// dependency-free (no SKaiNET engine deps) so any downstream consumer can take it without pulling
// in a specific backend.
kotlin {
    android {
        namespace = "sk.ainet.asr.domain"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    linuxX64()
    linuxArm64()
    macosArm64()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}
