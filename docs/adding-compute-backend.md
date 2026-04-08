# Adding a Compute Backend

This guide explains how to implement and register a new compute backend
(e.g. Metal GPU, MLX, CUDA, Vulkan) for the SKaiNET inference engine.

## Architecture Overview

```
llm-core (commonMain)
  └── BackendProvider          ← interface you implement
  └── BackendRegistry          ← expect object (discovery)

llm-core (jvmMain)
  └── BackendRegistry.jvm      ← ServiceLoader-based discovery

llm-core (registryBasedMain)
  └── BackendRegistry          ← manual registry (native, JS, Wasm, Android)

llm-runtime/kllama
  └── CpuBackendProvider       ← reference implementation
  └── META-INF/services/...    ← JVM SPI registration
  └── BackendActual.kt         ← native registration per platform
```

Backends are discovered differently per target:

| Target            | Mechanism                                            |
|-------------------|------------------------------------------------------|
| JVM               | `java.util.ServiceLoader` — auto-discovers from JAR  |
| Native (macOS, Linux, iOS) | `registerPlatformBackends()` at startup     |
| Android, JS, Wasm | `BackendRegistry.register()` called by host app      |

The registry auto-selects the **highest priority available** backend.
Users can override with `--backend=NAME` on the CLI.

## Step 1: Implement `BackendProvider`

Create a class that implements `BackendProvider` from
`llm-core/src/commonMain/kotlin/sk/ainet/apps/llm/backend/BackendProvider.kt`:

```kotlin
package sk.ainet.apps.mybackend

import sk.ainet.apps.llm.backend.BackendProvider
import sk.ainet.context.ExecutionContext

class MetalBackendProvider : BackendProvider {
    override val name: String = "metal"
    override val displayName: String = "Metal GPU"
    override val priority: Int = 100          // GPU > CPU (0)

    override fun isAvailable(): Boolean {
        // Runtime check: is the hardware/driver present?
        // Return false if not — the registry will skip this backend.
        return try {
            MetalExecutionContext()
            true
        } catch (_: Throwable) {
            false
        }
    }

    override fun createContext(): ExecutionContext {
        return MetalExecutionContext()
    }
}
```

**Priority guidelines:**

| Backend    | Priority |
|------------|----------|
| CPU        | 0        |
| GPU (Metal, Vulkan) | 100 |
| Specialized accelerator | 200 |

`isAvailable()` must be safe to call on any platform — return `false`
if the hardware or native library is not present.

## Step 2: Register the backend

Registration differs by target.

### JVM — ServiceLoader (automatic)

Create a service file in your module's JVM resources:

```
src/jvmMain/resources/META-INF/services/sk.ainet.apps.llm.backend.BackendProvider
```

Contents (one fully-qualified class name per line):

```
sk.ainet.apps.mybackend.MetalBackendProvider
```

That's it. Adding the JAR to the classpath makes it discoverable.
The Shadow JAR's `mergeServiceFiles()` handles combining service files
from multiple JARs.

### Native — manual registration

In the platform-specific `BackendActual.kt`, add a `register()` call
inside `registerPlatformBackends()`:

```kotlin
// llm-runtime/kllama/src/macosMain/kotlin/.../BackendActual.kt

internal actual fun registerPlatformBackends() {
    BackendRegistry.register(CpuBackendProvider())
    BackendRegistry.register(MetalBackendProvider())   // ← add this
}
```

This is called once at CLI startup before backend selection happens.

### Android / JS / Wasm

Call `BackendRegistry.register()` from your application's initialization
code before any inference calls:

```kotlin
// In your app's startup
BackendRegistry.register(CpuBackendProvider())
BackendRegistry.register(MyGpuBackendProvider())
```

## Step 3: Add the dependency

### As a separate module

If the backend lives in its own Gradle module or external JAR:

```kotlin
// build.gradle.kts of the consuming module
sourceSets {
    val jvmMain by getting {
        dependencies {
            implementation("sk.ainet.core:skainet-backend-metal:0.18.0")
        }
    }
    val macosMain by getting {
        dependencies {
            implementation("sk.ainet.core:skainet-backend-metal:0.18.0")
        }
    }
}
```

### Native bridge libraries

If the backend wraps a native C/C++ library (Metal, MLX, Vulkan),
configure linker opts in the native binary block:

```kotlin
macosArm64 {
    binaries {
        executable {
            linkerOpts(
                "-L/path/to/bridge", "-lmetal_bridge",
                "-framework", "Metal",
                "-framework", "MetalPerformanceShaders",
                "-framework", "Accelerate",
            )
        }
    }
}
```

## Step 4: Verify

### List backends

```bash
# JVM
./gradlew :llm-runtime:kllama:runJvm --args="--list-backends"

# Native (macOS)
./llm-runtime/kllama/build/bin/macosArm64/debugExecutable/kllama.kexe --list-backends
```

Expected output:

```
Available backends:
  metal        Metal GPU (priority=100, available)
  cpu          CPU (SIMD) (priority=0, available)
```

### Run with a specific backend

```bash
./gradlew :llm-runtime:kllama:runJvm \
  --args="--backend=metal -m model.gguf 'Hello'"
```

### Auto-selection

Without `--backend`, the registry picks the highest-priority available
backend automatically (Metal over CPU in this example).

## Reference: Existing implementation

The CPU backend serves as the reference implementation:

- **Provider:** `llm-runtime/kllama/src/commonMain/kotlin/sk/ainet/apps/kllama/CpuBackendProvider.kt`
- **JVM SPI file:** `llm-runtime/kllama/src/jvmMain/resources/META-INF/services/sk.ainet.apps.llm.backend.BackendProvider`
- **Native registration:** `llm-runtime/kllama/src/{macosMain,linuxMain,iosMain}/kotlin/.../BackendActual.kt`
- **Interface:** `llm-core/src/commonMain/kotlin/sk/ainet/apps/llm/backend/BackendProvider.kt`
- **Registry:** `llm-core/src/commonMain/kotlin/sk/ainet/apps/llm/backend/BackendRegistry.kt`

## File checklist for a new backend

```
[ ] BackendProvider implementation class
[ ] JVM: META-INF/services file listing the provider class
[ ] Native: register() call in platform BackendActual.kt
[ ] build.gradle.kts: dependency declaration
[ ] Native: linkerOpts if wrapping C/C++ bridge
[ ] Verify: --list-backends shows the new backend
[ ] Verify: --backend=NAME runs inference with it
```
