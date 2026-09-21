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
        val jvmMain by getting {
            dependencies {
                // Host tooling: Hugging Face checkpoint -> StableHLO export (MoonshineV2ExportCli).
                // JVM only, so commonMain keeps its lang-core + transformer-core footprint.
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.compile.hlo)
                implementation(libs.skainet.compile.opt)
                implementation(libs.skainet.io.core)
                implementation(libs.skainet.io.safetensors)
                implementation(libs.kotlinx.serialization.json)
            }
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.skainet.backend.cpu)
                implementation(project(":llm-core"))     // GGUFTokenizer for the E2E decode test
                implementation(libs.skainet.io.safetensors)   // synthetic checkpoint for MoonshineV2ExportTest
                implementation(libs.kotlinx.io.core)
                // DSL -> ComputeGraph -> StableHLO export (host tooling), for the
                // MLIR-dump tests that prove the encoder/decoder trace to bf16 StableHLO.
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.compile.hlo)
                implementation(libs.skainet.compile.opt)
            }
        }
    }
}

tasks.withType<JavaExec>().configureEach {
    minHeapSize = "2g"
    maxHeapSize = "12g"
}

tasks.register<JavaExec>("exportMoonshineV2") {
    group = "bridge"
    description = "Export Moonshine v2 streaming -> StableHLO MLIR (5 graphs) + dec_embed.bin + vocab.bin + manifest.json from a Hugging Face snapshot."
    val jvmMainComp = kotlin.jvm().compilations.getByName("main")
    dependsOn(jvmMainComp.compileTaskProvider)
    classpath = jvmMainComp.output.allOutputs + jvmMainComp.runtimeDependencyFiles
    mainClass.set("sk.ainet.models.moonshine.MoonshineV2ExportCliKt")
    listOf(
        "MOONSHINE_MODEL_DIR", "MOONSHINE_OUT_DIR", "MOONSHINE_GRAPH", "MOONSHINE_FE_SAMPLES",
        "MOONSHINE_ENC_FRAMES", "MOONSHINE_MAX_MEM", "MOONSHINE_STATIC_PAST",
    ).forEach { k -> System.getenv(k)?.let { environment(k, it) } }
}
