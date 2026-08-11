import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    // Track the public API of the NN primitives here (they live in this module since the 0.31.1
    // extraction). Before this, they were only listed in the stale llm-core.api re-export.
    alias(libs.plugins.binary.compatibility.validator)
}

// Framework NN primitives (attention, KV-cache family, embedding, norms, RoPE, FFNs) extracted from
// llm-core so they build on the FULL target matrix — including androidNative (the 32-bit box + phones).
// Depends ONLY on skainet-lang-core (which has androidNative); no io/compile/backend deps. llm-core
// re-exports this module, so existing consumers are unaffected; ARM-native consumers depend on it directly.
kotlin {
    android {
        namespace = "sk.ainet.lang.nn"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions { jvmTarget.set(JvmTarget.JVM_11) }
    }

    jvm()
    androidNativeArm32()
    androidNativeArm64()
    iosArm64()
    iosSimulatorArm64()
    linuxX64()
    linuxArm64()
    macosArm64()
    js { browser() }
    @OptIn(ExperimentalWasmDsl::class) wasmJs { browser() }
    @OptIn(ExperimentalWasmDsl::class) wasmWasi { nodejs() }

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            api(libs.skainet.lang.core)   // public API is lang-core-typed (Tensor/Module/ExecutionContext)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        val jvmTest by getting {
            dependencies {
                implementation(project.dependencies.platform(project(":llm-bom")))
                // Test-only: a real CPU backend (DirectCpuExecutionContext +
                // ServiceLoader-discovered kernel providers) so the
                // PreTransposedWeight `linearProject` path is proven against
                // the actual packed-quant matmul dispatch. Production code in
                // this module stays lang-core-only.
                implementation(libs.skainet.backend.cpu)
            }
        }
    }
}

// jvmTest runs real packed-quant matmuls (LinearProjectionPreTransposedTest):
// same convention as llm-inference/* — the Vector API module makes the JVM
// backend pick DefaultCpuOpsJvm, which auto-installs the ServiceLoader kernel
// providers the packed dispatch resolves against.
tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
