plugins {
    kotlin("jvm")
    application
}

application {
    mainClass.set("sk.ainet.apps.decode.MainKt")
}

dependencies {
    implementation(platform(project(":llm-bom")))

    implementation(project(":llm-apps:skainet-decode-core"))

    implementation(libs.skainet.lang.core)
    // KernelPacks + FfmRowMajorKernelPack: without the FFM row-major pack on the runtime
    // classpath, MAPPED/keep-packed weights fall to the decoding reference kernel (see the
    // kllama jvmMain note) — JVM-only, so it stays here rather than in -core.
    implementation(libs.skainet.backend.nativeCpu)
    implementation(libs.skainet.io.core)
    implementation(libs.kotlinx.coroutines)
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
