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
            val androidMain by getting
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
        val linuxX64Main by getting { dependsOn(linuxMain) }
        val linuxArm64Main by getting { dependsOn(linuxMain) }
        val macosArm64Main by getting { dependsOn(macosMain) }
    }
}

tasks.withType<Test>().configureEach {
    // Gemma4E2BToolCallSmokeTest dequantizes Q4_K → FP32 into MemorySegment-backed
    // direct memory (~20 GB for E2B). JDK 21 caps direct memory at ≈ -Xmx by
    // default, so bumping just the heap also lifts the direct cap — but we set
    // both explicitly to document intent. The 4g defaults keep the fast suite
    // cheap; real-checkpoint runs override, e.g.
    //   -PkgemmaTestMaxHeap=24g -PkgemmaTestMaxDirect=32g
    val maxDirect = (findProperty("kgemmaTestMaxDirect") as? String) ?: "4g"
    jvmArgs(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector",
        "-XX:MaxDirectMemorySize=$maxDirect",
    )
    maxHeapSize = (findProperty("kgemmaTestMaxHeap") as? String) ?: "4g"
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
    listOf("GEMMA_GGUF", "GEMMA_OUT_DIR", "GEN_SEQ", "PARTIAL_ROTARY", "GEMMA_DTYPE").forEach { k ->
        System.getenv(k)?.let { environment(k, it) }
    }
}
