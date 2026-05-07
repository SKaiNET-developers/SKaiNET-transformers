plugins {
    `java-platform`
    alias(libs.plugins.vanniktech.mavenPublish)
}

version = rootProject.findProperty("VERSION_NAME") ?: "0.1.0"

javaPlatform {
    allowDependencies()
}

// Single source of truth: every published subproject in this repo must appear
// here, except those listed in `bomExcludedPublishedPaths`. The
// `verifyBomCoverage` task below enforces this.
val bomModules = listOf(
    ":llm-api",
    ":llm-core",
    ":llm-agent",
    ":llm-providers",
    ":llm-inference:apertus",
    ":llm-inference:bert",
    ":llm-inference:gemma",
    ":llm-inference:llama",
    ":llm-inference:qwen",
    ":llm-inference:voxtral",
    ":llm-runtime:kgemma",
    ":llm-runtime:kllama",
)

// Published subprojects deliberately kept out of the BOM. Add a comment when
// you add an entry here so the next reader knows why.
val bomExcludedPublishedPaths = setOf(
    ":llm-bom",         // self
    ":llm-performance", // internal benchmarks, not part of the consumer surface
)

dependencies {
    // Re-export SKaiNET BOM so consumers get aligned SKaiNET versions transitively.
    api(platform("sk.ainet:skainet-bom:${libs.versions.skainet.get()}"))

    constraints {
        bomModules.forEach { api(project(it)) }
    }
}

// Drift guard. Captures the set of published subproject paths once all
// subprojects have evaluated, then fails `:check` if it diverges from
// `bomModules ∪ bomExcludedPublishedPaths`.
val publishedPaths = objects.setProperty(String::class.java)
gradle.projectsEvaluated {
    publishedPaths.set(
        rootProject.subprojects
            .filter { it.plugins.hasPlugin("com.vanniktech.maven.publish") }
            .map { it.path }
            .toSortedSet()
    )
    publishedPaths.disallowChanges()
}

val verifyBomCoverage by tasks.registering {
    val expected = bomModules.toSet()
    val excluded = bomExcludedPublishedPaths
    val publishedPathsInput = publishedPaths
    inputs.property("publishedPaths", publishedPathsInput)
    inputs.property("expectedInBom", expected)
    inputs.property("excluded", excluded)

    doLast {
        val paths = publishedPathsInput.get()
        val missing = (paths - expected - excluded).toSortedSet()
        val unknown = (expected - paths).toSortedSet()
        if (missing.isEmpty() && unknown.isEmpty()) return@doLast

        val msg = buildString {
            appendLine("BOM drift detected (llm-bom/build.gradle.kts):")
            if (missing.isNotEmpty()) {
                appendLine("  Published modules missing from bomModules:")
                missing.forEach { appendLine("    - $it") }
                appendLine("  Either add them to bomModules, or add them to bomExcludedPublishedPaths with a comment.")
            }
            if (unknown.isNotEmpty()) {
                appendLine("  bomModules entries that are not published:")
                unknown.forEach { appendLine("    - $it") }
                appendLine("  Either remove them from bomModules, or apply the maven-publish plugin to them.")
            }
        }
        throw GradleException(msg)
    }
}

tasks.named("check") { dependsOn(verifyBomCoverage) }
