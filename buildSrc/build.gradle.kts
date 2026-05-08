plugins {
    `kotlin-dsl`
}

repositories {
    gradlePluginPortal()
}

gradlePlugin {
    plugins {
        register("bom-coverage") {
            id = "sk.ainet.transformers.bom-coverage"
            implementationClass = "sk.ainet.transformers.build.BomCoveragePlugin"
        }
    }
}
