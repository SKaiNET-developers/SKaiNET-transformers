# SKaiNET-transformers

SKaiNET-transformers is a high-performance LLM (Large Language Model) application layer built on top of the [SKaiNET](https://github.com/SKaiNET-developers/SKaiNET.git) engine. It provides a set of runtimes and CLI tools for various transformer-based models, optimized for Kotlin Multiplatform.

## Key Features

- **Multi-Model Support**: Implementations for popular architectures including Llama, Gemma, Qwen, and BERT.
- **Engineered for Performance**: Uses the SKaiNET library as its core inference engine, leveraging hardware acceleration where available.
- **Kotlin Multiplatform**: Designed to run across different platforms (JVM, Native, Android, etc.).
- **Efficient Weights Loading**: Support for `safetensors` format for fast and safe model loading.

## Project Structure

- `llm-core`: Core abstractions and base classes for LLM components.
- `llm-inference`: Model-specific inference logic (Llama, BERT, Gemma, Qwen).
- `llm-runtime`: Platform-specific runtime implementations.
- `llm-apps`: Ready-to-use CLI applications for model interaction and testing.
- `llm-agent`: High-level agentic capabilities (in development).

## Current Release

The current release is **0.16.0**. To use SKaiNET-transformers in your project, add the following dependency:

```kotlin
dependencies {
    implementation("sk.ainet.transformers:llm-core:0.16.0")
}
```

Make sure to use a matching version of the SKaiNET engine (`sk.ainet.core:skainet-lang-core:0.16.0`).

## Getting Started

### Prerequisites

- JDK 17 or higher
- Gradle

### Running the CLI Tools

You can run the provided CLI tools using Gradle. For example, to run the BERT CLI:

```bash
./gradlew :llm-apps:kbert-cli:run --args="/path/to/model-dir 'query text'"
```

Replace `/path/to/model-dir` with a directory containing `model.safetensors`, `vocab.txt`, and `config.json`.

## Engine

This project uses **SKaiNET** as its underlying execution engine. 
GitHub: [https://github.com/SKaiNET-developers/SKaiNET](https://github.com/SKaiNET-developers/SKaiNET.git)

## License

[Add License Information Here]
