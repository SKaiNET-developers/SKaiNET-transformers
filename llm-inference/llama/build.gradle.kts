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
        namespace = "sk.ainet.models.llama"
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
                // Test-only: trace llamaNetwork to a ComputeGraph + lower to
                // StableHLO (RealSmolLm2BakeIrpaTest, transformers#305). Mirrors
                // :llm-inference:gemma's RealGemmaBakeIrpaTest — JVM-only, since
                // skainet-compile-hlo/-dag publish no JS/Wasm variant and the
                // trace/export path is host tooling anyway.
                implementation(libs.skainet.compile.dag)
                implementation(libs.skainet.compile.hlo)
                // Test-only: write the externalized weights to an IREE .irpa
                // parameter archive (RealSmolLm2BakeIrpaTest).
                implementation(libs.skainet.io.iree.params)
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g")
    maxHeapSize = "6g"
    // Opt-in gate for NarrowFloatMatmulBenchmark, which is a measurement rather than a test and
    // stays skipped unless explicitly requested. Gradle does not forward -D to the test JVM.
    System.getProperty("skainet.bench.narrow")?.let { systemProperty("skainet.bench.narrow", it) }
}
