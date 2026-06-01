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
        namespace = "sk.ainet.models.gemma"
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
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            implementation(libs.skainet.io.safetensors)
            implementation(libs.skainet.compile.core)
            implementation(project(":llm-core"))
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
        }

        commonTest.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.kotlin.test)
            implementation(libs.skainet.backend.cpu)
            // Test-only: trace gemmaNetwork to a ComputeGraph + lower to StableHLO.
            implementation(libs.skainet.compile.dag)
            implementation(libs.skainet.compile.hlo)
        }

        val jvmTest by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(libs.kotlin.test)
                implementation(libs.junit)
                implementation(libs.kotlinx.coroutines.test)
                implementation(libs.skainet.backend.cpu)
                // Test-only dep so GemmaDslToolCallIntegrationTest can build
                // a real ChatSession around the DSL runtime. Production code
                // in this module keeps no llm-agent coupling.
                implementation(project(":llm-agent"))
                implementation(libs.kotlinx.serialization.json)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = (findProperty("gemmaTestMaxHeap") as? String) ?: "6g"
}

// Kotlin/JS + Kotlin/WASM browser test runners have two separate problems on
// this codebase:
//   1. They don't discover backtick-named commonTest methods reliably, so
//      allTests fails with "did not discover any tests" — `failOnNoDiscoveredTests
//      = false` works around that.
//   2. They need ChromeHeadless at runtime, which isn't installed on headless
//      Linux build agents. The same tests already run on JVM (`jvmTest`) so
//      skipping the browser variants loses no coverage in practice.
// A `-PincludeBrowserTests` escape hatch keeps them available when a browser
// is present.
val includeBrowserTests = project.hasProperty("includeBrowserTests")
tasks.matching { it.name == "jsBrowserTest" || it.name == "wasmJsBrowserTest" }.configureEach {
    (this as? org.jetbrains.kotlin.gradle.targets.js.testing.KotlinJsTest)
        ?.failOnNoDiscoveredTests = false
    enabled = includeBrowserTests
}

// Real-model (FunctionGemma-270M) load test dequantizes ~270M params to FP32 (~1GB).
tasks.withType<Test>().configureEach { maxHeapSize = "8g" }
