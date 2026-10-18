// Kotlin/Native BitNet b1.58 CLI (linuxArm64 / linuxX64) — Ventuno Q bring-up (P1-T2).
// The native-cpu C kernels (NEON ternary LUT GEMV + fused lm_head stage 1) are linked in
// through skainet-backend-native-cpu's embedded static archive; no JVM, no FFM.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

kotlin {
    linuxX64 { binaries { executable { entryPoint = "sk.ainet.apps.kbitnet.nativecli.main"; baseName = "kbitnet" } } }
    linuxArm64 { binaries { executable { entryPoint = "sk.ainet.apps.kbitnet.nativecli.main"; baseName = "kbitnet" } } }

    sourceSets {
        nativeMain {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(project(":llm-inference:bitnet"))
                implementation(project(":llm-core"))
                implementation(libs.skainet.lang.core)
                implementation(libs.skainet.backend.api)
                implementation(libs.skainet.backend.cpu)
                implementation(libs.skainet.backend.nativeCpu)
                implementation(libs.skainet.io.core)
                implementation(libs.skainet.io.gguf)
                implementation(libs.kotlinx.coroutines)
            }
        }
    }
}
