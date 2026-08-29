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
            api(project(":llm-core")) // loader API surfaces QuantPolicy (transformers-owned since engine 0.40)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
        }

        commonTest.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.kotlin.test)
            implementation(libs.skainet.backend.cpu)
        }

        val jvmTest by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                implementation(libs.kotlin.test)
                // No JUnit 4: the module runs under the Jupiter engine, and
                // JUnit 4's AssumptionViolatedException is not a Jupiter
                // TestAbortedException, so `org.junit.Assume` records skips as
                // failures (#261). Dropping the dep makes that mistake
                // unrepresentable — use org.junit.jupiter.api.Assumptions.
                implementation(libs.kotlinx.coroutines.test)
                implementation(libs.skainet.backend.cpu)
                // Test-only: trace gemmaNetwork to a ComputeGraph + lower to
                // StableHLO. JVM-only — skainet-compile-hlo/-dag publish no JS
                // variant and the trace/export tests are host tooling anyway, so
                // GemmaTraceTest lives in jvmTest too. Keeping these out of
                // commonTest stops js/wasm from resolving compile-hlo against
                // published SKaiNET.
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.compile.hlo)
                // Test-only dep so GemmaDslToolCallIntegrationTest can build
                // a real ChatSession around the DSL runtime. Production code
                // in this module keeps no llm-agent coupling.
                implementation(project(":llm-agent"))
                implementation(libs.kotlinx.serialization.json)
                // Test-only: write the externalized weights to an IREE .irpa
                // parameter archive (RealGemmaBakeIrpaTest).
                implementation(libs.skainet.io.iree.params)
                // NOTE (0.40.0 closure train investigation, #170/#184): this
                // module deliberately does NOT pull in `skainet.backend.
                // nativeCpu` the way `:llm-inference:llama`'s jvmTest does.
                // Temporarily adding it while validating this closure train
                // confirmed the SKaiNET#951 native (FFM) kernel really is
                // faster (real-checkpoint decode: ~53% higher tok/s) and
                // still byte-identical via the new pre-transposed path — but
                // it also reproducibly zeroed out `GemmaQ5xPackedParityTest`'s
                // synthetic Q5_0/Q5_1 byte-level checks, which exercise the
                // *classic* (non-pre-transposed) packed weight through
                // `linearProject`'s lazy `ops.transpose` branch. Those two
                // tests construct `Q5_{0,1}BlockTensorData` directly — no
                // code this PR touches — so this is a pre-existing upstream
                // engine dispatch gap (native Q5_0/Q5_1 kernel × lazy
                // transpose), not something introduced or fixable here.
                // Production consumers of Gemma (`:llm-runtime:kgemma`)
                // already depend on `skainet.backend.nativeCpu` directly and
                // now get the pre-transposed path by default, which sidesteps
                // this gap entirely — see `packPreTransposed`, `linearProject`.
                // Left un-wired here so this module's own test suite stays
                // green without masking that finding; see the closure-train
                // PR description for the measurement.
            }
        }
    }
}

// Real-model (FunctionGemma-270M) integration tests (run with -PincludeIntegration)
// dequantize ~270M params to FP32, and GemmaQ5KPackedParityTest holds the FP32
// baseline plus both packed decode networks at once; the bake-to-irpa test holds
// weights + serialized bytes simultaneously. 8g OOMs once the real model is
// present, so default to 12g — override via -PgemmaTestMaxHeap (CI without the
// model file self-skips these and never needs the headroom).
tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = (findProperty("gemmaTestMaxHeap") as? String) ?: "12g"
    (findProperty("seqLen") as? String)?.let { systemProperty("seqLen", it) }
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
