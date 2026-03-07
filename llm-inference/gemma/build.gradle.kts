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
            implementation("sk.ainet:skainet-lang-core")
            implementation("sk.ainet:skainet-io-core")
            implementation("sk.ainet:skainet-io-gguf")
            implementation("sk.ainet:skainet-io-safetensors")
            implementation("sk.ainet:skainet-compile-core")
            implementation(project(":llm-core"))
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation("sk.ainet:skainet-backend-cpu")
        }

        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.junit)
                implementation(libs.kotlinx.coroutines.test)
                implementation("sk.ainet:skainet-backend-cpu")
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
