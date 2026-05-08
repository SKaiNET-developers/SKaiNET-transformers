package sk.ainet.transformers.gradle

import org.gradle.api.provider.SetProperty

abstract class BomCoverageExtension {
    abstract val excludePublished: SetProperty<String>
}
