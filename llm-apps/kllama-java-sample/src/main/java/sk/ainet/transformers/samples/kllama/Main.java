package sk.ainet.transformers.samples.kllama;

import sk.ainet.apps.kllama.chat.AgentConfig;
import sk.ainet.apps.kllama.chat.ModelMetadata;
import sk.ainet.apps.kllama.chat.ToolDefinition;
import sk.ainet.apps.kllama.chat.java.JavaAgentLoop;
import sk.ainet.apps.kllama.chat.java.JavaTool;
import sk.ainet.apps.kllama.chat.java.JavaTools;
import sk.ainet.apps.kllama.java.KLlamaJava;
import sk.ainet.apps.kllama.java.KLlamaSession;

import java.nio.file.Path;
import java.util.Map;

/**
 * Pure-Java sample showing how to load a TinyLlama / Llama 3 GGUF and run a
 * tool-calling conversation through the kllama Java surface.
 *
 * <p>Run with:
 * <pre>
 *   ./gradlew :llm-apps:kllama-java-sample:run \
 *       --args="/absolute/path/to/tinyllama-1.1b-chat-v1.0.Q8_0.gguf 'What is 17 * 23?'"
 * </pre>
 */
public final class Main {

    private Main() {
        // utility entry point
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: kllama-java-sample <model.gguf> [prompt]");
            System.exit(1);
        }

        Path modelPath = Path.of(args[0]);
        String prompt = args.length >= 2 ? args[1] : "What is 17 * 23?";

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
                Object exprObj = arguments.get("expression");
                if (exprObj == null) return "error: missing expression";
                String s = exprObj.toString().trim();
                String[] parts = s.split("\\s*\\*\\s*");
                if (parts.length != 2) {
                    return "error: only 'a * b' is supported in this sample";
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

        try (KLlamaSession session = KLlamaJava.loadGGUF(modelPath, null)) {
            JavaAgentLoop agent = JavaAgentLoop.builder()
                    .session(session)
                    .tool(calculator)
                    .systemPrompt("You are a helpful assistant. When asked an arithmetic "
                            + "question, call the calculator tool exactly once.")
                    .config(new AgentConfig())
                    .template("llama3")
                    .metadata(new ModelMetadata())
                    .build();

            // Stream tokens to stdout as they arrive.
            String finalResponse = agent.chat(prompt, System.out::print);

            System.out.println();
            System.out.println("---");
            System.out.println("Final assistant response:");
            System.out.println(finalResponse);
        }
    }
}
