plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Moonshine-tiny ASR (encoder-decoder) authored in the SKaiNET NN DSL, bf16-native
// so the DSL->StableHLO export keeps bf16 weights at the matmul (required by the
// Torq NPU: fp32 weights crash the torq compiler's getWeightMemoryFormat — see the
// demo docs/torq-npu-weight-crash.md).
//
// commonMain depends ONLY on lang-core + transformer-core (both androidNative-capable), so the
// model runs on the edge NPU / phone target set as well as the host. The encoder uses
// transformer-core's eager `TransformerBlock` (not llm-core's compile-capable
// `HybridTransformerBlock`), which is why llm-core / io / compile drop out of commonMain — they
// were only ever used by the jvmTest MLIR-dump / GGUFTokenizer path. (GH #239.)
kotlin {
    jvm()
    androidNativeArm32()
    androidNativeArm64()
    iosArm64()
    linuxX64()
    linuxArm64()

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.skainet.lang.core)
            implementation(project(":transformer-core"))
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.skainet.backend.cpu)
                implementation(project(":llm-core"))     // GGUFTokenizer for the E2E decode test
                // DSL -> ComputeGraph -> StableHLO export (host tooling), for the
                // MLIR-dump tests that prove the encoder/decoder trace to bf16 StableHLO.
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.compile.hlo)
                implementation(libs.skainet.compile.opt)
            }
        }
    }
}
