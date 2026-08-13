plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// SmolLM2 compiled-export product module (the whisper/moonshine/functiongemma
// "one self-contained module" shape): the redecode (fixed-seq, in-graph argMax)
// StableHLO export for the DSL -> StableHLO -> IREE compiled leg
// (transformers#305, the compiled counterpart to #272's cross-target
// eager-decode reproducer).
//
// This module DEPENDS on :llm-inference:llama for the architecture
// (LlamaNetworkLoader, DecoderGgufWeightLoader) — it does NOT fork the model.
// What it owns is the export PRODUCT: the `smollm2` redecode graph (one fixed
// `[1,seq]` prefill pass ending in an in-graph argMax, driven in a loop —
// GemmaDecoder's re-decode pattern, not the two-graph KV-cache decode; that's
// follow-up scope once this simpler path is proven on-device) and its
// safetensors archive.
//
//   jvmMain — SmolLm2ExportHarness (trace + StableHLO emit + safetensors) and
//             the env-driven CLI entry (exportSmolLm2 task below).
//
// JVM-only for now: the export harness is host tooling, not on-device code,
// and there's no commonMain/native source to justify publishing empty
// linux{X64,Arm64} klibs (unlike functiongemma, which shares real commonMain
// code across targets). Add native targets if/when this module gains actual
// shared or native-specific source.
kotlin {
    jvm()

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
                // Architecture: LlamaNetworkLoader / DecoderGgufWeightLoader.
                implementation(project(":llm-inference:llama"))
                // MultiHeadAttention (KV-cache strip before tracing — see SmolLm2ExportHarness.export).
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
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    // Same rationale as :llm-inference:llama / :llm-inference:functiongemma —
    // SmolLM2-135M is much smaller than FunctionGemma-270M, so 6g is generous
    // (vs. their 12g), not copied blindly.
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=6g")
    maxHeapSize = (findProperty("smollm2TestMaxHeap") as? String) ?: "6g"
}

// SmolLM2 compiled-export entry point:
//   SMOLLM2_GGUF=…SmolLM2-135M-Instruct-Q8_0.gguf SMOLLM2_OUT_DIR=build/mlir \
//     ./gradlew :llm-inference:smollm2:exportSmolLm2
tasks.register<JavaExec>("exportSmolLm2") {
    group = "bridge"
    description = "Export SmolLM2 -> StableHLO MLIR (redecode, argMax tail) + safetensors from the GGUF."
    val jvmMainComp = kotlin.jvm().compilations.getByName("main")
    dependsOn(jvmMainComp.compileTaskProvider)
    classpath = jvmMainComp.output.allOutputs + jvmMainComp.runtimeDependencyFiles
    mainClass.set("sk.ainet.models.smollm2.SmolLm2ExportCliKt")
    listOf("SMOLLM2_GGUF", "SMOLLM2_OUT_DIR", "SMOLLM2_SEQ", "SMOLLM2_DTYPE").forEach { k ->
        System.getenv(k)?.let { environment(k, it) }
    }
}
