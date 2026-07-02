plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Moonshine-tiny ASR (encoder-decoder) authored in the SKaiNET NN DSL, bf16-native
// so the DSL->StableHLO export keeps bf16 weights at the matmul (required by the
// Torq NPU: fp32 weights crash the torq compiler's getWeightMemoryFormat — see the
// demo docs/torq-npu-weight-crash.md). Targets mirror the board/host path only
// (jvm for authoring/export tests, linux{X64,Arm64} for the device); the wide
// gemma target set is unnecessary here.
kotlin {
    jvm()
    linuxX64()
    linuxArm64()

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.safetensors)
            implementation(libs.skainet.compile.core)
            implementation(project(":llm-core")) // brings transformer-core transitively
            implementation(libs.kotlinx.io.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.skainet.backend.cpu)
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.skainet.backend.cpu)
                // DSL -> ComputeGraph -> StableHLO export (host tooling), for the
                // MLIR-dump test that proves the encoder traces to bf16 StableHLO.
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.compile.hlo)
                implementation(libs.skainet.compile.opt)
            }
        }
    }
}
