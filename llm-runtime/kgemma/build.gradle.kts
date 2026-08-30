import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
}

kotlin {
    compilerOptions {
        optIn.add("sk.ainet.lang.memory.ExperimentalMemoryApi")
    }

    android {
        namespace = "sk.ainet.apps.kgemma"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    linuxX64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kgemma.cli.main"
                baseName = "kgemma"
            }
        }
    }

    linuxArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kgemma.cli.main"
                baseName = "kgemma"
            }
        }
    }

    macosArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kgemma.cli.main"
                baseName = "kgemma"
            }
        }
    }

    // Library-only Apple targets: no CLI executable, consumers link the klib
    // into their app (#271). The JVM-only deps (compile-hlo, gemma-iree,
    // llm-providers, …) are confined to jvmMain, so commonMain is iOS-clean.
    iosArm64()
    iosSimulatorArm64()

    jvm {
        mainRun {
            mainClass.set("sk.ainet.apps.kgemma.cli.MainKt")
        }
    }

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(project(":llm-inference:gemma"))
            implementation(project(":llm-core"))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.compile.core)
            implementation(libs.skainet.backend.cpu)
            implementation(libs.skainet.lang.ksp.annotations)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            implementation(libs.skainet.io.safetensors)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
        }

        commonTest.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.kotlin.test)
            implementation(libs.skainet.lang.models)
            implementation(libs.skainet.io.gguf)
        }

        val jvmMain by getting {
            dependencies {
                // FunctionGemma compiled export: DSL -> StableHLO (external params).
                // JVM-only (skainet-compile-hlo/-dag publish no JS).
                implementation(libs.skainet.compile.hlo)
                implementation(libs.skainet.compile.dag)
                // FunctionGemma export moved to :llm-inference:functiongemma (the DSL->StableHLO->IREE
                // module pattern shared with whisper/moonshine); FunctionGemmaExport* here are now
                // deprecated delegating shims (deprecate-don't-delete — exportFunctionGemma keeps working).
                implementation(project(":llm-inference:functiongemma"))
                // FunctionGemma facade: CompactCodec (<tool_N> -> ToolCall) + ToolCall.
                implementation(project(":llm-runtime:gemma-iree"))
                implementation(project(":llm-runtime:kllama"))
                // Direct dep on llm-agent for the --agent CLI flag.
                // kllama's `implementation(project(":llm-agent"))` isn't
                // transitively visible by default.
                implementation(project(":llm-agent"))
                // Spring-AI-style ChatModel surface used by Gemma4ChatModel.
                // llm-providers / llm-api are JVM-only today, so they live
                // here rather than in commonMain.
                implementation(project(":llm-providers"))
                implementation(project(":llm-api"))
                // KernelPacks.install()/FfmRowMajorKernelPack.install() — Gemma4ChatModel's
                // ensureKernelPacksInstalled() (mirrors KLlamaJava's).
                implementation(libs.skainet.backend.api)
                implementation(libs.skainet.backend.nativeCpu)
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(libs.kotlin.test)
                implementation(libs.kotlinx.coroutines.test)
                implementation(libs.skainet.backend.cpu)
                // Needed by Gemma4E2BToolCallSmokeTest for building
                // ToolDefinition parameter schemas inline.
                implementation(libs.kotlinx.serialization.json)
            }
        }
        if (!project.hasProperty("buildFatJar")) {
            val androidMain by getting {
                dependencies {
                    // NEON kernel backend (AAR, engine >= 0.39.0). ServiceLoader
                    // self-registers on ART; runtime classpath is all it needs.
                    runtimeOnly(libs.skainet.backend.jniCpu)
                }
            }
        }
        val wasmJsMain by getting
        wasmJsMain.dependencies {
            implementation(libs.kotlinx.browser)
        }

        val nativeMain by creating {
            dependsOn(commonMain.get())
        }
        val linuxMain by creating { dependsOn(nativeMain) }
        val macosMain by creating { dependsOn(nativeMain) }
        val iosMain by creating { dependsOn(nativeMain) }
        val linuxX64Main by getting { dependsOn(linuxMain) }
        val linuxArm64Main by getting { dependsOn(linuxMain) }
        val macosArm64Main by getting { dependsOn(macosMain) }
        val iosArm64Main by getting { dependsOn(iosMain) }
        val iosSimulatorArm64Main by getting { dependsOn(iosMain) }
    }
}

tasks.withType<Test>().configureEach {
    // HEAP — why 12g and not the root-wide 8192m (see the root build.gradle.kts).
    //
    // The real-checkpoint FunctionGemma-270M tests (FunctionGemmaExportTest, …EagerTest,
    // …Int8QuantTest, …WithPastCpuTest, …WithPastMlirDumpTest) dequantize Q5_K → FP32 and
    // trace the whole model. Measured from the emitted MLIR: 324 weight globals totalling
    // 436,111,680 float elements = 1.62 GiB FP32 (832 MiB as bf16 on disk, matching the
    // 872,248,326-byte gemma.safetensors). `t0` and `t2044` are BOTH 262153x640 — the tied
    // vocab embedding is materialized twice and is 77% of the archive on its own.
    //
    // Peak *live* heap is ~4.3 GiB because up to three near-simultaneous FP32 copies coexist:
    //   (1) the loader's dequantized weights,
    //   (2) TraceToGraphBuilder.extractFloatArray's `buffer.copyOf()`,
    //   (3) the BufferHandle.Owned ByteArrays behind module.externalParameters.
    // 4g cannot hold the live set at all; 8g is ~55% occupancy (G1 thrash); 12g is ~36%.
    // Mirrors :llm-inference:gemma, which defaults to 12g for the same model and reason.
    //
    // Costs CI nothing: -Xmx is a PROT_NONE virtual reservation, not RSS, and with no -Xms
    // the JVM commits ~256 MB initially — these tests abort in microseconds on the absent
    // checkpoint. Reducing the copies (narrow dense storage; dedup of the tied embedding)
    // is tracked upstream; once that lands this override can be deleted outright.
    //
    // DIRECT MEMORY — kept only so the two knobs stay symmetric; these tests do not use it.
    // DirectCpuExecutionContext.create() defaults to the on-heap DenseTensorDataFactory,
    // JvmRandomAccessSource reads via FileChannel into heap ByteArrays (no mmap, no
    // allocateDirect), and the safetensors writer uses ByteBuffer.allocate. Since JDK 8 the
    // default MaxDirectMemorySize already equals -Xmx, so this flag documents intent rather
    // than enforcing anything. NOTE: MemorySegment tensors (Arena.ofAuto, used by
    // Gemma4E2BToolCallSmokeTest and the CLI) are NOT charged against MaxDirectMemorySize at
    // all — only ByteBuffer.allocateDirect is. The previous comment here claimed otherwise.
    //
    // Heavier real-checkpoint runs still override, e.g.
    //   -PkgemmaTestMaxHeap=24g -PkgemmaTestMaxDirect=32g
    val maxDirect = (findProperty("kgemmaTestMaxDirect") as? String) ?: "12g"
    jvmArgs(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector",
        "-XX:MaxDirectMemorySize=$maxDirect",
    )
    maxHeapSize = (findProperty("kgemmaTestMaxHeap") as? String) ?: "12g"
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=36g")
    minHeapSize = "4g"
    maxHeapSize = "24g"
}

// FunctionGemma compiled export entry point (driven by the demo's scripts/compile-gemma.sh):
//   GEMMA_GGUF=…Q5_K_M.gguf GEMMA_OUT_DIR=… \
//     ./gradlew -PuseLocalSkainet=true :llm-runtime:kgemma:exportFunctionGemma
// Inherits the --add-modules jdk.incubator.vector + heap config from withType<JavaExec> above.
tasks.register<JavaExec>("exportFunctionGemma") {
    group = "bridge"
    description = "Export FunctionGemma -> gemma-gen.mlir + bf16 gemma.safetensors from the GGUF."
    val jvmMainComp = kotlin.jvm().compilations.getByName("main")
    dependsOn(jvmMainComp.compileTaskProvider)
    classpath = jvmMainComp.output.allOutputs + jvmMainComp.runtimeDependencyFiles
    mainClass.set("sk.ainet.apps.kgemma.FunctionGemmaExportMainKt")
    // GEMMA_GRAPH selects the graph(s): redecode (default) | prefill | with_past | all.
    // GEMMA_QUANT=int8 quantizes the 2D matmul weights (Phase 5).
    // GEMMA_SENTINEL_PAST=1 rolls the with_past graph back to the legacy sentinel-prime trace (#248).
    listOf("GEMMA_GGUF", "GEMMA_OUT_DIR", "GEN_SEQ", "PARTIAL_ROTARY", "GEMMA_DTYPE", "GEMMA_GRAPH", "GEMMA_QUANT", "GEMMA_SENTINEL_PAST").forEach { k ->
        System.getenv(k)?.let { environment(k, it) }
    }
}
