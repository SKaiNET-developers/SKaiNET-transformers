import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Framework NN primitives (attention, KV-cache family, embedding, norms, RoPE, FFNs) extracted from
// llm-core so they build on the FULL target matrix — including androidNative (the 32-bit box + phones).
// Depends ONLY on skainet-lang-core (which has androidNative); no io/compile/backend deps. llm-core
// re-exports this module, so existing consumers are unaffected; ARM-native consumers depend on it directly.
kotlin {
    android {
        namespace = "sk.ainet.lang.nn"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions { jvmTarget.set(JvmTarget.JVM_11) }
    }

    jvm()
    androidNativeArm32()
    androidNativeArm64()
    iosArm64()
    iosSimulatorArm64()
    linuxX64()
    linuxArm64()
    macosArm64()
    js { browser() }
    @OptIn(ExperimentalWasmDsl::class) wasmJs { browser() }
    @OptIn(ExperimentalWasmDsl::class) wasmWasi { nodejs() }

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            api(libs.skainet.lang.core)   // public API is lang-core-typed (Tensor/Module/ExecutionContext)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}
