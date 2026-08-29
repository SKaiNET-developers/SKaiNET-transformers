plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
}

// FunctionGemma-270M as a standalone function-calling module (the whisper/moonshine
// "one self-contained module" shape): export spec + export harness + contract
// manifest for the DSL -> DAG -> StableHLO -> IREE pipeline.
//
// This module DEPENDS on :llm-inference:gemma for the architecture (GemmaModel,
// GemmaNetworkLoader, Gemma4WeightLoader) — it does NOT fork the model. What it
// owns is the function-calling PRODUCT: the three-graph export (`gemma` redecode,
// `gemma_prefill`, `gemma_with_past` true-dynamic KV), the per-graph safetensors
// (per-trace external-parameter numbering — PR #291), and the `manifest.json`
// contract (function names, arg/output orders, dims, tool map) that the board
// runtime (:llm-runtime:gemma-iree GemmaKvDecoder) consumes mechanically.
//
//   commonMain — FunctionGemmaSpec + FunctionGemmaContract (manifest emission,
//                pure Kotlin; testable without a checkpoint)
//   jvmMain    — FunctionGemmaExportHarness (trace + StableHLO emit + archives)
//                and the env-driven CLI entry (exportFunctionGemma task below)
//
// Targets mirror moonshine's (jvm for authoring/export, linux{X64,Arm64} for
// host tooling).
kotlin {
    compilerOptions {
        optIn.add("sk.ainet.lang.memory.ExperimentalMemoryApi")
    }

    jvm()
    linuxX64()
    linuxArm64()

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
        val jvmMain by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                // Architecture: GemmaModel / GemmaNetworkLoader / Gemma4WeightLoader.
                implementation(project(":llm-inference:gemma"))
                // MultiHeadAttention (KV-cache strip before tracing — see FunctionGemmaExportHarness.export).
                implementation(project(":transformer-core"))
                implementation(libs.skainet.lang.core)
                implementation(libs.skainet.compile.core)
                // DSL -> ComputeGraph -> StableHLO export (JVM-only publications).
                implementation(libs.skainet.compile.hlo)
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.backend.cpu)
                implementation(libs.skainet.io.core)
                implementation(libs.skainet.io.gguf)
                implementation(libs.kotlinx.coroutines)
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.skainet.backend.cpu)
                // GGUFTokenizer for the vmfb greedy-parity test prompt/decode.
                implementation(project(":llm-core"))
                implementation(libs.kotlinx.io.core)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    // HEAP — same 12g rationale as :llm-runtime:kgemma (the config follows the
    // real-checkpoint tests): the FunctionGemma-270M export tests dequantize
    // Q5_K -> FP32 and trace the whole model; peak *live* heap is ~4.3 GiB
    // (loader weights + TraceToGraphBuilder copy + BufferHandle.Owned external
    // params near-simultaneously). 4g OOMs, 8g thrashes, 12g is ~36% occupancy.
    // Costs CI nothing: -Xmx is a virtual reservation and the tests abort in
    // microseconds on the absent checkpoint.
    val maxDirect = (findProperty("functiongemmaTestMaxDirect") as? String) ?: "12g"
    jvmArgs(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector",
        "-XX:MaxDirectMemorySize=$maxDirect",
    )
    maxHeapSize = (findProperty("functiongemmaTestMaxHeap") as? String) ?: "12g"
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=36g")
    minHeapSize = "4g"
    maxHeapSize = "24g"
}

// FunctionGemma compiled-export entry point (the spec-driven successor of
// :llm-runtime:kgemma:exportFunctionGemma — same env contract, so
// compile-gemma.sh can switch by task path alone):
//   GEMMA_GGUF=…Q5_K_M.gguf GEMMA_OUT_DIR=build/mlir GEMMA_GRAPH=all \
//     ./gradlew :llm-inference:functiongemma:exportFunctionGemma
tasks.register<JavaExec>("exportFunctionGemma") {
    group = "bridge"
    description = "Export FunctionGemma -> StableHLO MLIR + per-graph safetensors + manifest.json from the GGUF."
    val jvmMainComp = kotlin.jvm().compilations.getByName("main")
    dependsOn(jvmMainComp.compileTaskProvider)
    classpath = jvmMainComp.output.allOutputs + jvmMainComp.runtimeDependencyFiles
    mainClass.set("sk.ainet.models.functiongemma.FunctionGemmaExportCliKt")
    // GEMMA_GRAPH selects the graph(s): redecode (default) | prefill | with_past | all.
    // GEMMA_QUANT=int8 quantizes the 2D matmul weights (Phase 5).
    // GEMMA_SENTINEL_PAST=1 rolls the with_past graph back to the legacy sentinel-prime trace (#248).
    listOf("GEMMA_GGUF", "GEMMA_OUT_DIR", "GEN_SEQ", "PARTIAL_ROTARY", "GEMMA_DTYPE", "GEMMA_GRAPH", "GEMMA_QUANT", "GEMMA_SENTINEL_PAST").forEach { k ->
        System.getenv(k)?.let { environment(k, it) }
    }
}
