import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.kover)
}

kotlin {
    jvm {
        mainRun {
            mainClass.set("sk.ainet.apps.kqwen.cli.MainKt")
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project(":llm-inference:qwen"))
            implementation(project(":llm-core"))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.compile.core)
            implementation(libs.skainet.backend.cpu)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
        }

        val jvmMain by getting {
            dependencies {
                implementation(project(":llm-runtime:kllama"))
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    minHeapSize = "4g"
    maxHeapSize = "24g"
}
