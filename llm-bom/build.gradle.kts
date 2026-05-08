plugins {
    `java-platform`
    id("sk.ainet.transformers.bom-coverage")
    alias(libs.plugins.vanniktech.mavenPublish)
}

version = rootProject.findProperty("VERSION_NAME") ?: "0.1.0"

javaPlatform {
    allowDependencies()
}

bomCoverage {
    // Internal benchmarks; not part of the consumer surface.
    excludePublished.add(":llm-performance")
}

dependencies {
    // Re-export SKaiNET BOM so consumers get aligned SKaiNET versions transitively.
    api(platform("sk.ainet:skainet-bom:${libs.versions.skainet.get()}"))
}
