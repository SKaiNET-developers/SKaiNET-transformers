import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// Typed decision contract (REQ-API): sealed requests, questions, predictions, outcomes, policies,
// provenance and the model-neutral adapter interface. No engine dependency by design.
plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
}

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.decisions"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    linuxX64()
    linuxArm64()
    macosArm64()

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
            // Structured state is carried as JSON; the only dependency of the contract.
            api(libs.kotlinx.serialization.json)
        }
        commonTest.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.kotlin.test)
        }
    }
}
