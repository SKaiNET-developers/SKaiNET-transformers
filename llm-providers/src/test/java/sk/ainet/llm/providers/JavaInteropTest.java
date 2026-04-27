package sk.ainet.llm.providers;

import org.junit.jupiter.api.Test;
import sk.ainet.llm.api.ChatOptions;
import sk.ainet.llm.api.ChatRequest;
import sk.ainet.llm.api.ChatResponse;
import sk.ainet.llm.api.Embedding;
import sk.ainet.llm.api.EmbeddingRequest;
import sk.ainet.llm.api.EmbeddingResponse;
import sk.ainet.llm.api.FinishReason;
import sk.ainet.llm.api.Generation;
import sk.ainet.llm.api.Message;
import sk.ainet.llm.api.Role;
import sk.ainet.llm.api.ToolCall;
import sk.ainet.llm.api.ToolDefinition;
import sk.ainet.llm.api.Usage;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Compile-and-run check that the public llm-api SPI is reachable from plain Java
 * with the ergonomics we care about: static factory methods on Message, JvmField on
 * ChatOptions.DEFAULTS, and JvmOverloads-generated short constructors.
 */
class JavaInteropTest {

    @Test
    void messageStaticFactories() {
        Message sys = Message.system("you are helpful");
        Message usr = Message.user("hi");
        Message asst = Message.assistant("hello back");
        Message tool = Message.tool("42", "call-1");

        assertEquals(Role.SYSTEM, sys.getRole());
        assertEquals(Role.USER, usr.getRole());
        assertEquals(Role.ASSISTANT, asst.getRole());
        assertEquals(Role.TOOL, tool.getRole());
        assertEquals("call-1", tool.getToolCallId());
    }

    @Test
    void chatOptionsDefaultsIsStaticField() {
        ChatOptions defaults = ChatOptions.DEFAULTS;
        assertNotNull(defaults);
        assertEquals(0.7f, defaults.getTemperature(), 1e-6);
        assertEquals(512, defaults.getMaxTokens());
    }

    @Test
    void chatOptionsBuiltViaJvmOverloads() {
        // No-args constructor should exist via @JvmOverloads on the primary constructor.
        ChatOptions empty = new ChatOptions();
        assertNull(empty.getTemperature());
        assertEquals(Collections.emptyList(), empty.getStopSequences());
    }

    @Test
    void chatRequestSinglePromptConstructor() {
        ChatRequest r = new ChatRequest("hello");
        assertEquals(1, r.getMessages().size());
        assertEquals("hello", r.getMessages().get(0).getContent());
        assertNull(r.getOptions());
    }

    @Test
    void embeddingResponseShape() {
        Embedding e = new Embedding(0, new float[] { 0.1f, 0.2f, 0.3f });
        EmbeddingResponse resp = new EmbeddingResponse(
            Collections.singletonList(e),
            new Usage(2, 0),
            "test-model"
        );
        assertEquals(1, resp.getEmbeddings().size());
        assertEquals(3, resp.getEmbeddings().get(0).getVector().length);
    }

    @Test
    void chatResponseAndGenerationConstructible() {
        Message asst = Message.assistant("done");
        Generation gen = new Generation(asst, FinishReason.STOP);
        ChatResponse resp = new ChatResponse(Collections.singletonList(gen));
        assertEquals("done", resp.getText());
        assertSame(FinishReason.STOP, resp.getGenerations().get(0).getFinishReason());
    }

    @Test
    void toolDefinitionAndToolCallConstructible() {
        ToolDefinition def = new ToolDefinition(
            "search",
            "Web search",
            "{\"type\":\"object\"}"
        );
        ToolCall call = new ToolCall("tc-1", "search", "{\"q\":\"hello\"}");
        assertEquals("search", def.getName());
        assertEquals("tc-1", call.getId());
    }

    @Test
    void embeddingRequestSingleStringConstructor() {
        EmbeddingRequest r = new EmbeddingRequest("hello");
        assertEquals(1, r.getInputs().size());
        assertEquals("hello", r.getInputs().get(0));
    }
}
