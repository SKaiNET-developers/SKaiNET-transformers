plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Whisper-tiny ASR (encoder-decoder) authored in the SKaiNET NN DSL, targeting the
// IREE/Vulkan Android runtime (skainet-whisper-android). Fully static export shapes:
// a short-window encoder (audioCtx configurable, e.g. 200 = 4 s) and a prefill/step
// decoder pair with the KV cache as explicit graph I/O and host-computed f32 masks
// (no i1/select — the Vulkan/SPIR-V rule). German (or any language) is configuration
// via WhisperExportSpec, not code. Targets mirror moonshine's (jvm for authoring and
// export tests, linux{X64,Arm64} for host tooling).
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
        val jvmMain by getting {
            dependencies {
                // Host tooling: Hugging Face checkpoint -> StableHLO export (WhisperExportCli).
                // JVM only, so commonMain keeps its footprint.
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.compile.hlo)
                implementation(libs.skainet.compile.opt)
                implementation(libs.skainet.backend.cpu)   // the masks are baked as real constants
                implementation(libs.kotlinx.serialization.json)
                // NOTE: sk.ainet.core:skainet-io-iree-params (IrpaWriter) was evaluated for the
                // merged prefill/step weights archive and REJECTED by the IREE 3.x runtime
                // (header_size=40 vs required 88; unpacked entry structs). The harness emits
                // the packed IRPA v0 layout itself — see WhisperExportHarness.writeIrpa.
            }
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.skainet.backend.cpu)
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.skainet.backend.cpu)
                // DSL -> ComputeGraph -> StableHLO export (host tooling) for the
                // MLIR-dump/export tests.
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.compile.hlo)
                implementation(libs.skainet.compile.opt)
            }
        }
    }
}

tasks.withType<JavaExec>().configureEach {
    minHeapSize = "2g"
    maxHeapSize = "8g"
}

tasks.register<JavaExec>("exportWhisper") {
    group = "bridge"
    description = "Export Whisper -> StableHLO MLIR (encoder, prefill, step) + merged params.irpa + manifest.json from a Hugging Face snapshot."
    val jvmMainComp = kotlin.jvm().compilations.getByName("main")
    dependsOn(jvmMainComp.compileTaskProvider)
    classpath = jvmMainComp.output.allOutputs + jvmMainComp.runtimeDependencyFiles
    mainClass.set("sk.ainet.models.whisper.WhisperExportCliKt")
    listOf(
        "WHISPER_MODEL_DIR", "WHISPER_OUT_DIR", "WHISPER_AUDIO_CTX", "WHISPER_MAX_POSITIONS", "WHISPER_LANGUAGE",
    ).forEach { k -> System.getenv(k)?.let { environment(k, it) } }
}
