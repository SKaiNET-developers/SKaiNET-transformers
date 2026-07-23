plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// T5 encoder-decoder (t5-base / gtr-t5-base) authored as a direct tensor-ops runtime
// (BertRuntime style): per-head attention, T5 relative-position bias, RMS T5LayerNorm,
// un-gated ReLU FFN, tied embeddings. Provides the GTR sentence embedder and the seq2seq
// generation the vec2text (embedding-inversion) module builds on.
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
            implementation(project(":llm-core")) // brings transformer-core transitively
            implementation(libs.kotlinx.io.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.skainet.backend.cpu)
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.kotlinx.coroutines.test)
                implementation(libs.skainet.backend.cpu)
                implementation(libs.skainet.io.safetensors)
                implementation(libs.kotlinx.serialization.json)
            }
        }
    }
}
