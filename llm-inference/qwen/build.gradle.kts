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
        namespace = "sk.ainet.models.qwen"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    macosArm64()
    linuxX64()
    linuxArm64()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            api(project(":llm-inference:llama"))
            implementation(project(":llm-core"))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            implementation(libs.skainet.io.safetensors)
            implementation(libs.skainet.compile.core)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
        }

        commonTest.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.kotlin.test)
            implementation(libs.skainet.backend.cpu)
        }

        // QwenExportHarness / QwenExportCli (SKaiNET-transformers#411): DSL -> ComputeGraph ->
        // StableHLO export, JVM-only publications (compile.hlo/compile.dag), matching
        // :llm-inference:functiongemma's jvmMain shape.
        val jvmMain by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(project(":llm-core"))
                implementation(project(":transformer-core"))
                implementation(libs.skainet.lang.core)
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
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(libs.kotlin.test)
                implementation(libs.junit)
                implementation(libs.kotlinx.coroutines.test)
                implementation(libs.skainet.backend.cpu)
                // Pulls the priority-100 native (FFM) provider onto the
                // jvmTest classpath so KernelRegistry.bestAvailable()
                // hands out the native Q4_K / FP32 kernels for the
                // pipeline test. JVM-only: native-cpu has no Kotlin/
                // Native, JS, or Wasm targets.
                implementation(libs.skainet.backend.nativeCpu)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    // Export tests (QwenExportDumpTest) dequantize a Q8_0 checkpoint and trace the whole model --
    // same rationale as :llm-inference:functiongemma's 12g (FunctionGemma-270M's own tests need
    // that much; Qwen2.5-0.5B/Qwen3-0.6B are 2-2.2x the params). Absent-checkpoint runs abort in
    // microseconds, so this costs CI nothing.
    val maxDirect = (findProperty("qwenTestMaxDirect") as? String) ?: "24g"
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=$maxDirect")
    maxHeapSize = (findProperty("qwenTestMaxHeap") as? String) ?: "12g"
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=36g")
    minHeapSize = "4g"
    maxHeapSize = "24g"
}

// Qwen compiled-export entry point (SKaiNET-transformers#411):
//   QWEN_GGUF=…/qwen3-0.6b-q8_0.gguf QWEN_OUT_DIR=build/mlir QWEN_GRAPH=all \
//     ./gradlew :llm-inference:qwen:exportQwen
tasks.register<JavaExec>("exportQwen") {
    group = "bridge"
    description = "Export Qwen2.5/Qwen3 -> StableHLO MLIR + per-graph safetensors + manifest.json from a GGUF."
    val jvmMainComp = kotlin.jvm().compilations.getByName("main")
    dependsOn(jvmMainComp.compileTaskProvider)
    classpath = jvmMainComp.output.allOutputs + jvmMainComp.runtimeDependencyFiles
    mainClass.set("sk.ainet.models.qwen.QwenExportCliKt")
    listOf("QWEN_GGUF", "QWEN_OUT_DIR", "QWEN_GRAPH", "QWEN_SEQ", "QWEN_CHUNK", "QWEN_DTYPE").forEach { k ->
        System.getenv(k)?.let { environment(k, it) }
    }
}
