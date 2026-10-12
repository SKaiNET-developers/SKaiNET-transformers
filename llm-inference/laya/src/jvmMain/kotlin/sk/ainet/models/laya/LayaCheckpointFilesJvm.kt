package sk.ainet.models.laya

import java.nio.file.Files
import java.nio.file.Path

/**
 * Reads the text files of a checkpoint directory laid out as the published checkpoints are:
 * `rl_agent_config.json`, `encoder/config.json`, `tokenizer/tokenizer.json`,
 * `tokenizer/tokenizer_config.json`.
 */
public fun LayaCheckpointFiles.Companion.read(directory: Path): LayaCheckpointFiles = LayaCheckpointFiles(
    agentConfigJson = Files.readString(directory.resolve("rl_agent_config.json")),
    encoderConfigJson = Files.readString(directory.resolve("encoder").resolve("config.json")),
    tokenizerJson = Files.readString(directory.resolve("tokenizer").resolve("tokenizer.json")),
    tokenizerConfigJson = Files.readString(directory.resolve("tokenizer").resolve("tokenizer_config.json")),
)
