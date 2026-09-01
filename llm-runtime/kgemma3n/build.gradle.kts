import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.kover)
}

kotlin {
    jvm()

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(project(":llm-inference:gemma3n"))
            implementation(project(":llm-inference:gemma"))
            implementation(project(":llm-core"))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            implementation(libs.skainet.io.safetensors)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
        }

        commonTest.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.kotlin.test)
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
