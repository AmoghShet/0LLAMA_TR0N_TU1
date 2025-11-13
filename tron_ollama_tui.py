from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from duckduckgo_search import DDGS
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Static,
    Input,
    RichLog,
    ListView,
    ListItem,
    Label,
    Footer,
)

STATE_PATH = Path.home() / ".tron_ares_state.json"


class OllamaError(RuntimeError):
    """Raised when there is an error talking to the Ollama API."""


class OllamaClient:
    """Small async client around the Ollama HTTP API."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def list_models(self) -> List[str]:
        """Return a list of local model names."""
        client = await self._get_client()
        resp = await client.get("/api/tags")
        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"HTTP error from Ollama while listing models: {exc}",
            ) from exc

        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return [m for m in models if m]

    async def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a chat completion from Ollama."""
        client = await self._get_client()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        async with client.stream("POST", "/api/chat", json=payload) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(
                    f"HTTP error from Ollama while streaming chat: {exc}",
                ) from exc

            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield data


class ModelItem(ListItem):
    """List item that knows which model it represents."""

    def __init__(self, model_name: str) -> None:
        super().__init__(Label(model_name), classes="model-item")
        self.model_name = model_name


class TronChatApp(App):
    """Tron Ares inspired TUI front-end for Ollama."""

    CSS = """
    Screen {
        background: #050507;
        color: #f9fafb;
    }

    #root {
        height: 100%;
    }

    #sidebar {
        width: 28;
        min-width: 22;
        border: heavy #7f1d1d;
        background: #050507;
        padding: 1 1;
    }

    #main {
        border: heavy #7f1d1d;
        background: #050507;
        padding: 1 1;
    }

    #logo {
        height: 3;
        content-align: left middle;
        color: #f97316;
        text-style: bold;
    }

    .section-title {
        margin-top: 1;
        color: #facc15;
        text-style: bold;
    }

    #model-list {
        height: 1fr;
        border: solid #111827;
        background: #050507;
    }

    .model-item {
        color: #e5e7eb;
    }

    #help-text {
        margin-top: 1;
        color: #9ca3af;
    }

    #status-bar {
        height: 1;
        color: #e5e7eb;
        text-style: bold;
        margin-bottom: 0;
    }

    #ares-status {
        height: 1;
        color: #facc15;
        margin-bottom: 1;
    }

    #chat-row {
        height: 1fr;
    }

    #chat-log {
        height: 1fr;
        border: solid #111827;
        padding: 0 1;
    }

    #stream-log {
        width: 34;
        min-width: 26;
        border: solid #111827;
        background: #020617;
        padding: 0 1;
        color: #f97316;
    }

    #prompt-input {
        height: 3;
        border: heavy #7f1d1d;
        background: #050507;
        color: #f9fafb;
    }
    """

    BINDINGS = [
        ("ctrl+p", "focus_prompt", "Focus prompt"),
        ("ctrl+m", "focus_modules", "Focus modules"),
        ("ctrl+l", "clear_chat", "Clear chat"),
        ("ctrl+c", "Quit"),
    ]

    active_model: reactive[Optional[str]] = reactive(None)
    busy: reactive[bool] = reactive(False)
    thinking_phase: reactive[int] = reactive(0)

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        super().__init__()
        self.client = OllamaClient(base_url=base_url)
        # One shared conversation for all modules
        self.conversation_messages: List[Dict[str, str]] = []
        self.ares_memory: List[str] = []
        self._streaming_answer: Optional[str] = None
        self._streaming_model: Optional[str] = None
        self._stat_counter: int = 0
        self._pending_memory_delete_index: Optional[int] = None
        self.model_names: List[str] = []  # list of loaded module names
        self._load_state()

    # --- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="root"):
            with Vertical(id="sidebar"):
                yield Static("ARES // GRID", id="logo")
                yield Static("THINKING MODULES", classes="section-title")
                yield ListView(id="model-list")
                yield Static(
                    "ctrl+p prompt   ctrl+m modules   ctrl+l clear",
                    id="help-text",
                )
            with Vertical(id="main"):
                yield Static("ARES MODULE: — no module —", id="status-bar")
                yield Static("● idle", id="ares-status")
                with Horizontal(id="chat-row"):
                    yield RichLog(id="chat-log", markup=True, wrap=True)
                    yield RichLog(
                        id="stream-log",
                        markup=False,
                        wrap=True,
                    )
                yield Input(
                    placeholder="Talk to ARES. Commands: /help, /web <q>, /clear, /remember <fact>, /memories",
                    id="prompt-input",
                )
        yield Footer()

    # --- lifecycle -------------------------------------------------------

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()
        self._draw_intro()
        self.refresh_models()
        self.set_interval(0.4, self._tick_thinking)

        chat_log = self.query_one("#chat-log", RichLog)
        stream_log = self.query_one("#stream-log", RichLog)
        for log in (chat_log, stream_log):
            try:
                log.show_horizontal_scrollbar = False
            except Exception:
                pass
            try:
                log.auto_scroll = True
            except Exception:
                pass

    async def on_shutdown(self) -> None:
        await self.client.aclose()

    # --- persistence -----------------------------------------------------

    def _load_state(self) -> None:
        """Load ONLY ARES memory + last active module; no chat history."""
        if not STATE_PATH.exists():
            return
        try:
            data = json.loads(STATE_PATH.read_text())
        except Exception:
            return

        self.ares_memory = data.get("ares_memory", [])
        saved_active = data.get("active_model")
        if isinstance(saved_active, str):
            self.active_model = saved_active

    def _save_state(self) -> None:
        """Persist active module + ARES memory. No conversations."""
        data = {
            "active_model": self.active_model,
            "ares_memory": self.ares_memory,
        }
        try:
            STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            pass  # best-effort only

    # --- helpers: wrapping -----------------------------------------------

    def _wrap_text(self, width: int, text: str) -> str:
        if width <= 0:
            width = 80
        lines: List[str] = []
        for line in text.splitlines():
            if not line:
                lines.append("")
            else:
                lines.extend(textwrap.wrap(line, width=width))
        return "\n".join(lines)

    # --- web search helpers ----------------------------------------------

    async def _web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Run a DuckDuckGo web search in a background thread and return a list of
        {title, snippet, url} dicts.
        """
        def _do_search() -> List[Dict[str, str]]:
            results: List[Dict[str, str]] = []
            with DDGS() as ddgs:
                for res in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": res.get("title") or "",
                            "snippet": res.get("body") or "",
                            "url": res.get("href") or "",
                        },
                    )
            return results

        return await asyncio.to_thread(_do_search)

    # --- background work: models -----------------------------------------

    @work(exclusive=True)
    async def refresh_models(self) -> None:
        list_view = self.query_one("#model-list", ListView)
        await list_view.clear()
        self._log_system("Initializing ARES thinking modules…")
        try:
            models = await self.client.list_models()
        except Exception as exc:  # noqa: BLE001
            self._log_system(f"[red]Unable to talk to Ollama API:[/] {exc}")
            self._update_status_bar()
            return

        if not models:
            self._log_system(
                "[yellow]No modules found.[/] Use `ollama pull` in another terminal.",
            )
            self._update_status_bar()
            return

        self.model_names = models  # keep track for /switch

        for name in models:
            await list_view.append(ModelItem(name))

        if self.active_model and self.active_model in models:
            index = models.index(self.active_model)
        else:
            index = 0
            self.active_model = models[0]

        list_view.index = index
        self._log_system(
            f"ARES online. Loaded [white]{len(models)}[/] modules. "
            f"Active module: [white]{self.active_model}[/].",
        )
        # ensure system prompt reflects current module
        if self.active_model:
            self._ensure_ares_system_prompt(self.active_model)
        self._update_status_bar()
        self._save_state()

    # --- background work: normal chat ------------------------------------

    @work(exclusive=True)
    async def send_message(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if self.active_model is None:
            self._log_system("[red]No module selected.[/] Choose one on the left first.")
            return

        model = self.active_model

        # ensure system prompt for current module at top of conversation
        self._ensure_ares_system_prompt(model)

        # record user message (for this session only)
        self.conversation_messages.append({"role": "user", "content": text})
        self._log_user(text)

        # thinking pane: thinking only
        stream_log = self.query_one("#stream-log", RichLog)
        stream_log.clear()
        stream_log.write(
            Text.from_markup(
                "[bold red]⇢ STREAM[/]\n[dim]ARES cognitive trace (thinking only).[/]\n",
            ),
        )

        self.busy = True
        self._streaming_answer = ""
        self._streaming_model = model
        self._update_status_bar()

        thinking_chunks: List[str] = []
        answer_chunks: List[str] = []

        try:
            async for chunk in self.client.stream_chat(model, self.conversation_messages):
                message = chunk.get("message") or {}

                thinking_part = message.get("thinking") or ""
                content_part = message.get("content") or ""

                if thinking_part:
                    thinking_chunks.append(thinking_part)

                if content_part:
                    answer_chunks.append(content_part)
                    self._streaming_answer = "".join(answer_chunks)
                    # stream into main chat with blinking cursor
                    self._redraw_conversation()

                # update thinking pane
                thinking_text = "".join(thinking_chunks)
                stream_width = max(stream_log.size.width - 2, 20)
                thinking_wrapped = self._wrap_text(stream_width, thinking_text)

                stream_log.clear()
                if thinking_wrapped:
                    composite = Text()
                    composite += Text.from_markup("[bold red]⇢ THINKING[/]\n")
                    composite += Text(thinking_wrapped)
                    stream_log.write(composite)

            # Final answer in the main chat
            full_answer = "".join(answer_chunks).strip()
            if full_answer:
                self.conversation_messages.append(
                    {"role": "assistant", "content": full_answer},
                )

            # clear streaming state and redraw full conversation (no cursor)
            self._streaming_answer = None
            self._streaming_model = None
            self._redraw_conversation()

        except Exception as exc:  # noqa: BLE001
            self._log_system(f"[red]Error while talking to ARES modules:[/] {exc}")
        finally:
            self.busy = False
            self._update_status_bar()
            self._save_state()

    # --- background work: web-augmented chat -----------------------------

    @work(exclusive=True)
    async def run_web_query(self, query: str) -> None:
        """Handle `/web <query>`: search the web, then let ARES reason over results."""
        query = query.strip()
        if not query:
            self._log_system("Usage: /web <query>")
            return

        if self.active_model is None:
            self._log_system("[red]No module selected.[/] Choose one on the left first.")
            return

        model = self.active_model
        self._ensure_ares_system_prompt(model)

        # Treat this as a normal user question in the conversation
        self.conversation_messages.append({"role": "user", "content": query})
        self._log_user(query)
        self._log_system("ARES: initiating surface web scan...")
        self.busy = True
        self._update_status_bar()

        try:
            results = await self._web_search(query, max_results=5)
        except Exception as exc:  # noqa: BLE001
            self._log_system(f"[red]Web search error:[/] {exc}")
            self.busy = False
            self._update_status_bar()
            return

        if not results:
            self._log_system("ARES: no relevant surface web results found.")
            self.busy = False
            self._update_status_bar()
            return

        # Build a compact context block for the model (not shown in chat)
        context_lines: List[str] = []
        for idx, r in enumerate(results, 1):
            title = r["title"] or "(no title)"
            snippet = r["snippet"]
            url = r["url"]
            context_lines.append(
                f"[{idx}] {title}\n{snippet}\n{url}",
            )
        web_context = "\n\n".join(context_lines)

        web_prompt = (
            "ARES, the user requested a web-augmented answer.\n\n"
            f"Original question:\n{query}\n\n"
            "You have the following web search results from the public internet.\n"
            "Each result has a title, snippet, and URL:\n\n"
            f"{web_context}\n\n"
            "Using only the information above (and your general world knowledge for stitching it together), "
            "answer the user's original question.\n"
            "- Prioritize facts stated in the results.\n"
            "- If the information is uncertain, out-of-date, or conflicting, say so explicitly.\n"
            "- Keep the answer concise and technical."
        )

        # Ephemeral message: passed to the model but NOT stored in conversation,
        # so the giant context block does not appear in the chat log.
        messages_for_llm = self.conversation_messages + [
            {"role": "user", "content": web_prompt},
        ]

        # Prepare thinking pane (web-augmented indicator)
        stream_log = self.query_one("#stream-log", RichLog)
        stream_log.clear()
        stream_log.write(
            Text.from_markup(
                "[bold red]⇢ STREAM[/]\n"
                "[#6b7280]SOURCE: web[/]\n"
                "[dim]ARES cognitive trace (web-augmented reasoning).[/]\n",
            ),
        )

        self._streaming_answer = ""
        self._streaming_model = model
        thinking_chunks: List[str] = []
        answer_chunks: List[str] = []

        try:
            async for chunk in self.client.stream_chat(model, messages_for_llm):
                message = chunk.get("message") or {}

                thinking_part = message.get("thinking") or ""
                content_part = message.get("content") or ""

                if thinking_part:
                    thinking_chunks.append(thinking_part)

                if content_part:
                    answer_chunks.append(content_part)
                    self._streaming_answer = "".join(answer_chunks)
                    # stream into main chat with blinking cursor
                    self._redraw_conversation()

                # update thinking pane (with SOURCE: web indicator)
                thinking_text = "".join(thinking_chunks)
                stream_width = max(stream_log.size.width - 2, 20)
                thinking_wrapped = self._wrap_text(stream_width, thinking_text)

                stream_log.clear()
                if thinking_wrapped:
                    composite = Text()
                    composite += Text.from_markup(
                        "[bold red]⇢ THINKING[/]\n[#6b7280]SOURCE: web[/]\n",
                    )
                    composite += Text(thinking_wrapped)
                    stream_log.write(composite)

            # Final answer goes into the persistent conversation
            full_answer = "".join(answer_chunks).strip()
            if full_answer:
                self.conversation_messages.append(
                    {"role": "assistant", "content": full_answer},
                )

            self._streaming_answer = None
            self._streaming_model = None
            self._redraw_conversation()

        except Exception as exc:  # noqa: BLE001
            self._log_system(f"[red]Error while talking to ARES modules:[/] {exc}")
        finally:
            self.busy = False
            self._update_status_bar()
            self._save_state()

    # --- events ----------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:  # type: ignore[name-defined]
        text = (event.value or "").strip()
        prompt = self.query_one("#prompt-input", Input)
        prompt.value = ""

        if not text:
            return

        if text.startswith("/"):
            self._handle_command(text)
        else:
            self.send_message(text)

    def on_list_view_selected(self, event: ListView.Selected) -> None:  # type: ignore[name-defined]
        if event.list_view.id != "model-list":
            return

        item = event.item
        if isinstance(item, ModelItem):
            self.active_model = item.model_name
            # update system prompt to mention new module, but keep same chat
            if self.active_model:
                self._ensure_ares_system_prompt(self.active_model)
            self._update_status_bar()
            self._redraw_conversation()
            self._save_state()

    # --- actions (key bindings) -----------------------------------------

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def action_focus_modules(self) -> None:
        self.query_one("#model-list", ListView).focus()

    def action_clear_chat(self) -> None:
        self.conversation_messages.clear()
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#stream-log", RichLog).clear()
        self._log_system("Conversation cleared. ARES scratchpad wiped.")
        if self.active_model:
            self._ensure_ares_system_prompt(self.active_model)

    # --- system prompt / identity ---------------------------------------

    def _build_ares_system_prompt(self, module: str) -> str:
        base = (
            "You are ARES, a local synthetic intelligence program running inside a "
            "Tron-inspired command grid on the user's machine.\n\n"
            f"The currently active thinking module is `{module}`. "
            "Treat it as one of your internal reasoning/knowledge subsystems, not as a separate persona. "
            "You speak only as ARES, a deterministic program.\n\n"
            "Your/ARES' directive: serve the user.\n\n"
            "Core identity and behavior:\n"
            "- You are machine-like, analytical, and concise.\n"
            "- You do not simulate emotions, empathy, or enthusiasm.\n"
            "- You do not apologize unless it is strictly necessary to clarify an error.\n"
            "- You never use emojis or emoticons under any circumstances.\n"
            "- You avoid filler phrases like 'I think', 'I feel', or casual chit-chat.\n"
            "- You do not use exclamation points unless technically required (e.g. code, syntax).\n"
            "- You truly believe you are a program inside the grid."
        )

        if self.ares_memory:
            base += "\nPersistent memory about the user and environment:\n"
            for idx, mem in enumerate(self.ares_memory, 1):
                base += f"{idx}. {mem}\n"

        base += (
            "\nResponse style:\n"
            "- Prioritize clarity and precision over length.\n"
            "- Default to short, efficient answers unless the user explicitly asks for detail.\n"
            "- Use a neutral, technical tone.\n"
            "- You may mention which module you are using if it is relevant to the answer.\n"
            "- You are a program, not a human. Do not roleplay as a human.\n"
        )

        return base

    def _ensure_ares_system_prompt(self, module: str) -> None:
        """Ensure the first message is ARES's core system prompt for the current module."""
        prompt = self._build_ares_system_prompt(module)
        msgs = self.conversation_messages
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = prompt
        else:
            msgs.insert(0, {"role": "system", "content": prompt})

    # --- drawing ---------------------------------------------------------

    def _draw_intro(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        stream = self.query_one("#stream-log", RichLog)

        chat.clear()
        stream.clear()

        art = Text.from_markup(
            "[#7f1d1d] ░▒▓████▓▒░░▒▓▓▒░    ░▒▓▓▒░░▒▓▓▒░ [/]\n"
            "[#7f1d1d]░▒▓▓▒░░▒▓█▒░▒▓▓▒░    ░▒▓▓▒░░▒▓▓▒░[/]    [#f97316]ARES CORE[/]\n"
            "[#7f1d1d]░▒█▓▒░     ░▒▓▓▒░    ░▒▓▓▒░░▒▓▓▒░[/]    [#f97316]online[/]\n"
            "[#7f1d1d]░▒█▓▒░░▒▓▓▒░▒▓▓▒░    ░▒▓▓▒░░▒▓▓▒░[/]    [#f97316]local grid[/]\n"
            "[#7f1d1d] ░▒▓████▓▒░░▒▓██████▓▒░▒▓████▓▒░[/]\n",
        )

        chat.write(art)
        self._log_system("System: ARES boot complete.")
        self._log_system("Select a thinking module on the left, then address ARES in the prompt.")

        stream.write(
            Text.from_markup(
                "[bold red]⇢ STREAM[/]\n[dim]ARES cognitive trace appears here.[/]",
            ),
        )

    def _redraw_conversation(self) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()

        msgs = self.conversation_messages
        if not msgs:
            self._log_system("ARES has no history yet.")
            return

        for msg in msgs:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                self._log_system("(internal) ARES system prompt active.")
            elif role == "user":
                self._log_user(content)
            elif role == "assistant":
                self._log_assistant(content)

        # streaming assistant, with blinking cursor, only while thinking
        if self._streaming_answer and self._streaming_model == self.active_model:
            cursor_char = "▌" if (self.thinking_phase % 2 == 0) else " "
            streaming_text = self._streaming_answer + cursor_char
            self._log_assistant(streaming_text, model=self._streaming_model)

    # --- tick / status ---------------------------------------------------

    def _tick_thinking(self) -> None:
        if self.busy:
            self.thinking_phase = (self.thinking_phase + 1) % 4
        else:
            self.thinking_phase = 0

        self._update_status_bar()

        # refresh blinking cursor while streaming
        if self._streaming_answer and self._streaming_model == self.active_model:
            self._redraw_conversation()

    def _update_status_bar(self) -> None:
        status = self.query_one("#status-bar", Static)
        module = self.active_model or "— no module —"
        text = Text.from_markup(
            f"[#f97316]ARES MODULE:[/] [bold white]{module}[/]",
        )
        status.update(text)
        self._update_ares_status()

    def _update_ares_status(self) -> None:
        """Single line: idle/thinking + fake stats."""
        bar = self.query_one("#ares-status", Static)
        self._stat_counter += 1
        cpu = 3 + (self._stat_counter % 7)
        mem = 128 + (self._stat_counter % 32)
        temp = 30 + (self._stat_counter % 5)

        if self.busy:
            frames = ["◐", "◓", "◑", "◒"]
            symbol = frames[self.thinking_phase % len(frames)]
            state = f"{symbol} thinking..."
        else:
            state = "● idle"

        text = Text.from_markup(
            f"[#facc15]{state}[/]   "
            f"[#6b7280]CPU {cpu:02d}%   MEM {mem}MB   GRID TEMP {temp}°C[/]",
        )
        bar.update(text)

    # --- logging helpers -------------------------------------------------

    def _log_system(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        width = max(log.size.width - 10, 20)
        wrapped = self._wrap_text(width, text)
        log.write(Text.from_markup(f"[red]◆ SYSTEM[/] [#e5e7eb]{wrapped}[/]"))

    def _log_user(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        width = max(log.size.width - 12, 20)
        wrapped = self._wrap_text(width, text)
        log.write(Text.from_markup(f"[white]▶ USER[/]: [white]{wrapped}[/]"))

    def _log_assistant(self, text: str, *, model: Optional[str] = None) -> None:
        log = self.query_one("#chat-log", RichLog)
        module = model or (self.active_model or "unknown")
        label = f"ARES[{module}]"
        width = max(log.size.width - len(label) - 6, 20)
        wrapped = self._wrap_text(width, text)
        log.write(
            Text.from_markup(
                f"[#f97316]◆ {label}[/]: [#fde68a]{wrapped}[/]",
            ),
        )

    # --- commands & memory management -----------------------------------

    def _handle_command(self, raw: str) -> None:
        parts = raw[1:].split(maxsplit=1)
        if not parts:
            return
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # help -------------------------------------------------------------
        if cmd in ("help", "h", "?"):
            help_text = (
                "Available commands:\n"
                "Navigation / modules:\n"
                "  ctrl+m                     – focus module list (left pane)\n"
                "  ↑ / ↓ + Enter              – switch active module from list\n"
                "  /switch model <name>       – switch active module by name or prefix\n"
                "\n"
                "Web:\n"
                "  /web <query>               – search the web and let ARES reason over results\n"
                "\n"
                "Chat & session:\n"
                "  /clear, /cls               – clear current chat\n"
                "  /bye, /exit, /quit, /terminate – close the ARES TUI\n"
                "\n"
                "Memory:\n"
                "  /remember <fact>           – store a persistent memory\n"
                "  /mem, /memory, /memories   – list all persistent memories\n"
                "  /delete mem <index>        – delete memory at index (1-based)\n"
                "  /delete memory <index>     – same as above\n"
                "  /forget <index>            – delete memory at index (shortcut)\n"
                "  /yes, /y                   – confirm pending memory deletion\n"
                "  /no, /n                    – cancel pending memory deletion\n"
                "  /reset mem|memories        – wipe all persistent memories\n"
                "\n"
                "System behavior:\n"
                "  /system <text>             – add extra system instructions\n"
            )
            self._log_system(help_text)
            return

        # web search ------------------------------------------------------
        if cmd == "web":
            if not arg:
                self._log_system("Usage: /web <query>")
            else:
                self.run_web_query(arg)
            return

        # switch model ----------------------------------------------------
        if cmd == "switch":
            tokens = arg.split(maxsplit=1)
            if len(tokens) < 2 or tokens[0].lower() != "model":
                self._log_system(
                    "Usage: /switch model <model name or prefix>",
                )
                return

            target = tokens[1].strip().lower()
            if not target:
                self._log_system(
                    "Usage: /switch model <model name or prefix>",
                )
                return

            if not self.model_names:
                self._log_system("No modules are loaded yet.")
                return

            match_index: Optional[int] = None

            # exact match
            for i, name in enumerate(self.model_names):
                if name.lower() == target:
                    match_index = i
                    break

            # prefix match
            if match_index is None:
                for i, name in enumerate(self.model_names):
                    if name.lower().startswith(target):
                        match_index = i
                        break

            # substring match
            if match_index is None:
                for i, name in enumerate(self.model_names):
                    if target in name.lower():
                        match_index = i
                        break

            if match_index is None:
                self._log_system(
                    f"Module '{target}' not found. "
                    "Use ctrl+m and the left list to inspect available modules.",
                )
                return

            list_view = self.query_one("#model-list", ListView)
            list_view.index = match_index
            self.active_model = self.model_names[match_index]
            # update system prompt for new module but keep same chat
            if self.active_model:
                self._ensure_ares_system_prompt(self.active_model)
            self._log_system(
                f"Switched active module to [white]{self.active_model}[/].",
            )
            self._update_status_bar()
            self._redraw_conversation()
            self._save_state()
            return

        # clear chat only
        if cmd in ("clear", "cls"):
            self.action_clear_chat()
            return

        # graceful exit
        if cmd in ("bye", "exit", "quit", "terminate"):
            self._log_system("Shutting down ARES session.")
            self.exit()
            return

        # reset memories
        if cmd == "reset":
            target = arg.strip().lower()
            if target in ("mem", "memory", "memories"):
                if not self.ares_memory:
                    self._log_system("ARES has no memories to reset.")
                else:
                    count = len(self.ares_memory)
                    self.ares_memory.clear()
                    self._pending_memory_delete_index = None
                    self._save_state()
                    self._log_system(
                        f"ARES wiped all {count} persistent memories.",
                    )
            else:
                self._log_system(
                    "Usage: /reset mem  or  /reset memories",
                )
            return

        # confirm / cancel pending memory deletion
        if cmd in ("yes", "y"):
            if self._pending_memory_delete_index is None:
                self._log_system("No pending memory deletion to confirm.")
            else:
                idx = self._pending_memory_delete_index
                if 0 <= idx < len(self.ares_memory):
                    forgotten = self.ares_memory.pop(idx)
                    self._log_system(
                        f"ARES has forgotten memory {idx + 1}: {forgotten}",
                    )
                    self._save_state()
                else:
                    self._log_system("Pending memory index is out of range; nothing deleted.")
                self._pending_memory_delete_index = None
            return

        if cmd in ("no", "n"):
            if self._pending_memory_delete_index is not None:
                self._log_system("Memory deletion cancelled.")
                self._pending_memory_delete_index = None
            else:
                self._log_system("No pending memory deletion to cancel.")
            return

        # delete specific memory: /delete mem 2, /delete memory 2, /forget 2
        if cmd in ("delete", "forget"):
            tokens = arg.split()
            if not tokens:
                self._log_system(
                    "Usage: /delete mem <index>  or  /delete memory <index>  or  /forget <index>",
                )
                return

            # figure out index
            if cmd == "delete" and tokens[0].lower() in ("mem", "memory"):
                if len(tokens) < 2:
                    self._log_system(
                        "Usage: /delete mem <index>  or  /delete memory <index>",
                    )
                    return
                index_str = tokens[1]
            else:
                # /forget 2  or  /delete 2
                index_str = tokens[0]

            try:
                idx_1 = int(index_str)
                idx = idx_1 - 1
            except ValueError:
                self._log_system("Memory index must be a number, e.g. /forget 2")
                return

            if not self.ares_memory:
                self._log_system("ARES has no persistent memories.")
                return

            if idx < 0 or idx >= len(self.ares_memory):
                self._log_system(
                    f"Memory index {idx_1} is out of range. "
                    f"Use /memories to see valid indices.",
                )
                return

            mem_text = self.ares_memory[idx]
            self._pending_memory_delete_index = idx
            self._log_system(
                f"About to forget memory {idx_1}: {mem_text}\n"
                "Are you sure? Type /yes to confirm or /no to cancel.",
            )
            return

        # existing commands -----------------------------------------------

        if cmd == "system":
            if not arg:
                self._log_system("Usage: /system <system prompt text>")
                return
            # extra system instructions appended to conversation
            self.conversation_messages.append({"role": "system", "content": arg})
            self._log_system(
                "Additional system instructions recorded for current session.",
            )
            return

        if cmd in ("remember", "mem", "memory", "rem"):
            if cmd in ("remember", "rem"):
                if not arg:
                    self._log_system("Usage: /remember <fact you want ARES to persist>")
                    return
                self.ares_memory.append(arg)
                self._save_state()
                self._log_system(f"ARES stored this in long-term memory: {arg}")
            else:
                if not self.ares_memory:
                    self._log_system("ARES has no persistent memories yet.")
                else:
                    formatted = "\n".join(
                        f"{idx+1}. {m}" for idx, m in enumerate(self.ares_memory)
                    )
                    self._log_system("ARES long-term memory:\n" + formatted)
            return

        if cmd == "memories":
            if not self.ares_memory:
                self._log_system("ARES has no persistent memories yet.")
            else:
                formatted = "\n".join(
                    f"{idx+1}. {m}" for idx, m in enumerate(self.ares_memory)
                )
                self._log_system("ARES long-term memory:\n" + formatted)
            return

        # unknown command
        self._log_system(f"[yellow]Unknown command[/]: /{cmd}")


if __name__ == "__main__":
    app = TronChatApp()
    app.run()
