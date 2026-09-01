plugins {
    kotlin("jvm")
    application
}

application {
    mainClass.set("sk.ainet.apps.decode.MainKt")
}

dependencies {
    implementation(platform(project(":llm-bom")))

    implementation(project(":llm-core"))
    implementation(project(":llm-inference:llama"))
    implementation(project(":llm-inference:qwen"))
    implementation(project(":llm-inference:bitnet"))

    implementation(libs.skainet.lang.core)
    implementation(libs.skainet.backend.api)
    implementation(libs.skainet.backend.cpu)
    implementation(libs.skainet.backend.nativeCpu)
    implementation(libs.skainet.io.core)
    implementation(libs.skainet.io.gguf)
    implementation(libs.kotlinx.io.core)
    implementation(libs.kotlinx.coroutines)
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
