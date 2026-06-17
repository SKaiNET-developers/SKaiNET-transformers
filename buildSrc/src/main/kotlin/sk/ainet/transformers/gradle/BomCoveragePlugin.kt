package sk.ainet.transformers.gradle

import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project

private const val PUBLISH_PLUGIN_ID = "com.vanniktech.maven.publish"

class BomCoveragePlugin : Plugin<Project> {
    override fun apply(project: Project) {
        require(project.plugins.hasPlugin("java-platform")) {
            "sk.ainet.transformers.bom-coverage requires the java-platform plugin to be applied first"
        }

        val ext = project.extensions.create(
            "bomCoverage",
            BomCoverageExtension::class.java,
        )
        ext.excludePublished.convention(emptySet())

        project.rootProject.subprojects
            .filter { it.path != project.path }
            .forEach { project.evaluationDependsOn(it.path) }

        project.afterEvaluate {
            val excluded = ext.excludePublished.get() + project.path
            val publishedPaths = project.rootProject.subprojects
                .filter { it.plugins.hasPlugin(PUBLISH_PLUGIN_ID) }
                .map { it.path }
                .filterNot { it in excluded }
                .sorted()

            if (publishedPaths.isEmpty()) {
                throw GradleException(
                    "[bom-coverage] No published subprojects found for ${project.path}. " +
                        "At least one sibling must apply '$PUBLISH_PLUGIN_ID'."
                )
            }

            // Fail fast (at configuration time, not at Maven Central deploy time) when a NEW published
            // module forgot its gradle.properties. Without POM_ARTIFACT_ID the artifact silently defaults
            // to the bare project name (wrong coordinates / not the skainet-transformers-* convention);
            // without POM_NAME, Maven Central rejects the deploy. This recurs on every new module — catch it.
            val pomProblems = publishedPaths.mapNotNull { path ->
                val p = project.project(path)
                val missing = buildList {
                    if (p.findProperty("POM_ARTIFACT_ID")?.toString().isNullOrBlank()) add("POM_ARTIFACT_ID")
                    if (p.findProperty("POM_NAME")?.toString().isNullOrBlank()) add("POM_NAME")
                }
                if (missing.isEmpty()) null else "$path — missing ${missing.joinToString(" + ")}"
            }
            if (pomProblems.isNotEmpty()) {
                throw GradleException(
                    "[bom-coverage] Published module(s) are missing required POM properties — the Maven " +
                        "Central deploy would fail:\n" +
                        pomProblems.joinToString("\n") { "  - $it" } +
                        "\nAdd a `gradle.properties` to each module with POM_ARTIFACT_ID + POM_NAME " +
                        "(see `llm-core/gradle.properties`)."
                )
            }

            project.dependencies.constraints {
                publishedPaths.forEach { add("api", project.project(it)) }
            }
        }
    }
}
