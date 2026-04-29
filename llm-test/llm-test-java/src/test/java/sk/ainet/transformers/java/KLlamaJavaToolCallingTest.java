package sk.ainet.transformers.java;

import org.junit.jupiter.api.Test;
import sk.ainet.apps.kllama.chat.AgentConfig;
import sk.ainet.apps.kllama.chat.ModelMetadata;
import sk.ainet.apps.kllama.chat.ToolDefinition;
import sk.ainet.apps.kllama.chat.java.JavaAgentLoop;
import sk.ainet.apps.kllama.chat.java.JavaTool;
import sk.ainet.apps.kllama.chat.java.JavaTools;
import sk.ainet.apps.kllama.java.KLlamaJava;
import sk.ainet.apps.kllama.java.KLlamaSession;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end check that the kllama Java surface (KLlamaJava + KLlamaSession +
 * JavaAgentLoop + JavaTool + JavaTools) compiles and runs from pure Java
 * against a real TinyLlama checkpoint.
 *
 * Gated on the {@code TINYLLAMA_MODEL_PATH} env var so CI without a model
 * checkpoint still passes — the test reports as skipped (assumption failure)
 * rather than failed.
 *
 * Set the env var to a TinyLlama instruct GGUF, e.g.:
 * <pre>
 *   export TINYLLAMA_MODEL_PATH=~/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
 *   ./gradlew :llm-test:llm-test-java:test
 * </pre>
 */
class KLlamaJavaToolCallingTest {

    @Test
    void calculatorToolIsInvokedFromJavaAgentLoop() throws Exception {
        String envPath = System.getenv("TINYLLAMA_MODEL_PATH");
        assumeTrue(envPath != null && !envPath.isBlank(),
                "TINYLLAMA_MODEL_PATH not set — skipping Java end-to-end test.");

        Path modelPath = Path.of(envPath);
        assumeTrue(Files.exists(modelPath),
                "TINYLLAMA_MODEL_PATH=" + envPath + " does not exist — skipping.");

        // Build a Java tool that records each invocation. The model only has
        // to reach into our `execute` once for the assertion to pass; the
        // exact final answer is not constrained because TinyLlama 1.1B is
        // small and may be noisy on arithmetic.
        AtomicInteger callCount = new AtomicInteger(0);

        JavaTool calculator = new JavaTool() {
            @Override
            public ToolDefinition getDefinition() {
                return JavaTools.definition(
                        "calculator",
                        "Evaluate a simple arithmetic expression like '17 * 23'.",
                        "{\"type\":\"object\","
                                + "\"properties\":{\"expression\":{\"type\":\"string\","
                                + "\"description\":\"Arithmetic expression to evaluate\"}},"
                                + "\"required\":[\"expression\"]}"
                );
            }

            @Override
            public String execute(Map<String, ?> arguments) {
                callCount.incrementAndGet();
                Object expr = arguments.get("expression");
                if (expr == null) return "error: missing expression";
                String s = expr.toString().trim();
                // Tiny eval: only handles "<int> * <int>" — enough for the
                // smoke prompt below; real implementations would use a
                // proper expression parser.
                String[] parts = s.split("\\s*\\*\\s*");
                if (parts.length != 2) {
                    return "error: only 'a * b' is supported in this smoke test";
                }
                try {
                    long a = Long.parseLong(parts[0]);
                    long b = Long.parseLong(parts[1]);
                    return String.valueOf(a * b);
                } catch (NumberFormatException e) {
                    return "error: " + e.getMessage();
                }
            }
        };

        try (KLlamaSession session = KLlamaJava.loadGGUF(modelPath, /* systemPrompt */ null)) {
            JavaAgentLoop agent = JavaAgentLoop.builder()
                    .session(session)
                    .tool(calculator)
                    .systemPrompt("You are a helpful assistant. When asked an arithmetic "
                            + "question, call the calculator tool exactly once.")
                    .config(new AgentConfig())
                    .template("llama3")
                    .metadata(new ModelMetadata())
                    .build();

            String response = agent.chat("What is 17 * 23?");

            assertNotNull(response, "agent.chat returned null");
            assertTrue(callCount.get() >= 1,
                    "Expected calculator.execute() to be invoked at least once; "
                            + "got " + callCount.get() + " invocations. Response: " + response);
        }
    }
}
