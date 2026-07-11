plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// vec2text embedding inversion (decode sentence embeddings back to text): the inversion
// "hypothesizer" + iterative corrector, built on the :llm-inference:t5 encoder-decoder and
// GTR sentence embedder. Inference only (no training); batch size 1; greedy decoding.
kotlin {
    jvm()
    linuxX64()
    linuxArm64()

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(project(":llm-bom")))
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.safetensors)
            implementation(project(":llm-core"))
            implementation(project(":llm-inference:t5"))
            implementation(libs.kotlinx.io.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.skainet.backend.cpu)
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.skainet.backend.cpu)
            }
        }
    }
}
