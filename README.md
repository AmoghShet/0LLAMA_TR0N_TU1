# 0LLAMA_TR0N_TU1
Rudimentary Python based Ollama TUI inspired by Tron

A compact Textual-based terminal UI that lets you interact with local Ollama models in a Tron-inspired “ARES” interface.
The app streams model thinking (cognitive trace) and assistant responses, supports simple persistent memories, and persists the active module + memories to `~/.tron_ares_state.json`.

---

## Features

* List local Ollama models and select an active “thinking module”
* Streamed chat completions from Ollama with a dedicated “thinking” pane
* Shared conversation history (session-only) + persistent ARES memory items
* Keyboard shortcuts and slash commands for quick control
* Minimal, Tron-themed UI using `textual` + `rich`

---

## Requirements

* Python 3.10+ (3.11 recommended)
* Local Ollama HTTP API accessible (default `http://localhost:11434`)
* Python packages:

  * `textual`
  * `rich`
  * `httpx`

---

## Files

Place the provided script (example name `tron_ares.py`) in a project folder. The app persists lightweight state to:

```
~/.tron_ares_state.json
```

This file stores:

```json
{
  "active_model": "<model-name>",
  "ares_memory": ["fact 1", "fact 2", ...]
}
```

---

## Running

Ensure your Ollama server is running and has models available (e.g. with `ollama pull <model>` from the Ollama CLI). Then:

```bash
python tron_ares.py
```

If you need to point the client at a non-default Ollama base URL, modify the `TronChatApp` instantiation in `__main__`:

```py
if __name__ == "__main__":
    app = TronChatApp(base_url="http://<host>:<port>")
    app.run()
```

---

## UI / Keybindings

* `Ctrl+P` — Focus prompt input
* `Ctrl+M` — Focus module list (left pane)
* `Ctrl+L` — Clear chat
* `Ctrl+C` — Quit

Use arrow keys + Enter to switch module in the left pane.

---

## Chat commands (slash-style)

Type commands in the prompt (start with `/`). Key commands implemented:

Navigation / modules

```
/switch model <name|prefix>    — switch active module by name/prefix
```

Chat & session

```
/clear, /cls                   — clear current chat
/bye, /exit, /quit, /terminate — close the TUI
```

Memory

```
/remember <fact>               — store persistent memory
/mem, /memory, /memories       — list memories
/delete mem <index>            — delete memory (1-based)
/delete memory <index>         — same as above
/forget <index>                — shortcut for delete
/yes, /y                       — confirm pending delete
/no, /n                        — cancel pending delete
/reset mem|memories            — wipe all persistent memories
```

System

```
/system <text>                 — append extra system instructions (session-only)
```

Help

```
/help or /?                    — shows available commands
```

Unknown or malformed commands will be logged to the system pane.

---

## Behavior & Persistence

* Conversation messages are **session-only** (cleared on restart unless you export them yourself).
* `ARES` persistent memory and the last active module are saved to `~/.tron_ares_state.json`.
* The app updates the ARES system prompt dynamically to include persistent memories and the active module.

---

## Error handling / Troubleshooting

* If the app cannot reach Ollama it will log an error in the UI: `Unable to talk to Ollama API`.

  * Confirm Ollama is running and accessible at `http://localhost:11434` (or change the base URL).
  * Ensure at least one model is pulled locally (use the Ollama CLI: `ollama pull <model>`).
* If no models are found the UI suggests: `Use ollama pull in another terminal`.
* State file is best-effort: corrupted or unreadable state will be ignored (no crash).

---

## Customization

* UI styling: modify the `CSS` string in the `TronChatApp` class.
* Default base URL: change `TronChatApp(base_url=...)` in `__main__` or extend the app to read an environment variable / CLI flag.
* To add features such as exporting conversation history, extend `TronChatApp` methods where conversation messages are managed.

---

## Development notes

* `OllamaClient` uses `httpx.AsyncClient` and streams the Ollama `/api/chat` endpoint.
* The app uses `textual.work` for background tasks (`refresh_models`, `send_message`) so the UI stays responsive while streaming.
* System prompt is built in `_build_ares_system_prompt` and inserted as the first system message of the session.
