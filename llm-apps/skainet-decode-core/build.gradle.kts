import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
}

kotlin {
    compilerOptions {
        optIn.add("sk.ainet.lang.memory.ExperimentalMemoryApi")
    }

    android {
        namespace = "sk.ainet.apps.decode.core"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    jvm()

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            // api, not implementation: DecodeSession.run's onModelInfo exposes llm-core's
            // GGUFModelInfo in a public signature (same pattern as kllama, see #226).
            api(project(":llm-core"))
            implementation(project(":llm-inference:llama"))
            implementation(project(":llm-inference:qwen"))
            implementation(project(":llm-inference:bitnet"))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.backend.api)
            implementation(libs.skainet.backend.cpu)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
        }
    }
}
