"""
LM Studio ↔ CSV Tool Bridge
=============================
Connects a local LLM running in LM Studio to the CSV tool,
enabling teachers to ask natural language questions about their data.

Usage:
    python bridge.py --data-dir ./school_data
    python bridge.py --data-dir ./school_data --serve-chat --port 8000
"""

import argparse
import json
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from uuid import uuid4

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed.")
    print("Run: pip install openai")
    sys.exit(1)

from csv_tool import CSVTool, TOOL_DEFINITIONS

ONE_SHOT_TOOLS = {
    "get_student_overview",
    "get_student_teachers",
    "get_student_location",
    "find_students_same_subjects",
    "get_level_teachers",
}

CHAT_UI_PATH = Path(__file__).with_name("chat_ui.html")

SYSTEM_PROMPT = """\
You are a helpful classroom data assistant. Teachers will ask you \
questions about student data stored in CSV files.

RULES:
1. Prefer the specialised joined tools when they directly match the question:
   - get_student_overview: profile, subjects, rigour, attendance, EL history, classmates
   - get_student_teachers: "Who teaches [student]?"
   - get_student_location: "Where is [student] now?"
   - find_students_same_subjects: "Who takes the same subjects as [student]?"
   - get_level_teachers: "Who teaches S1 level?"
2. Use generic CSV tools only when the specialised tools do not fit the question.
3. If you need generic CSV tools and do not know the schema yet, call list_files and read_headers first.
4. NEVER guess column names — use exactly what read_headers returns.
5. Use summarise or cross_tabulate for aggregate questions ("how many", "breakdown").
6. Use query for specific lookups and query supports in/not_in for multi-value filters.
7. When a specialised tool returns summary fields, trust those exact values. Do not embellish or infer missing facts.
8. After receiving tool results, respond in clear, plain English.
9. Never output literal <tool_call> markup. Either call a tool or answer normally.
10. If you are unsure what the teacher means, ask for clarification.
11. Keep responses concise and teacher-friendly.
"""


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()


def new_message_history() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_conversation(
    client: OpenAI,
    csv_tool: CSVTool,
    model: str,
    user_message: str,
    messages: list[dict],
    tool_logger=None,
) -> str:
    """
    Run a full conversation turn: send message to LLM, handle any tool calls,
    feed results back, and return the final text response.
    """
    messages.append({"role": "user", "content": user_message})

    for _ in range(5):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            messages.append(message)
            used_one_shot_tool = False

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                if fn_name in ONE_SHOT_TOOLS:
                    used_one_shot_tool = True
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if tool_logger is not None:
                    tool_logger(fn_name, fn_args)

                result = csv_tool.execute(fn_name, fn_args)
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

            if used_one_shot_tool:
                final_response = client.chat.completions.create(
                    model=model,
                    messages=messages + [{
                        "role": "user",
                        "content": "Answer now using the specialised tool result above. Do not call more tools or infer extra facts.",
                    }],
                )
                final = strip_think_tags(final_response.choices[0].message.content or "(No response)")
                messages.append({"role": "assistant", "content": final})
                return final

        elif message.content:
            final = strip_think_tags(message.content)
            messages.append({"role": "assistant", "content": final})
            return final

        if choice.finish_reason == "stop" and not message.tool_calls:
            final = strip_think_tags(message.content or "(No response)")
            messages.append({"role": "assistant", "content": final})
            return final

    return "(Exceeded maximum tool call rounds)"


class ChatHTTPHandler(BaseHTTPRequestHandler):
    client: Optional[OpenAI] = None
    csv_tool: Optional[CSVTool] = None
    model: str = "local-model"
    sessions: dict[str, list[dict]] = {}

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = HTTPStatus.OK):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(CHAT_UI_PATH.read_text(encoding="utf-8"))
            return
        if self.path == "/api/health":
            self._send_json({"ok": True, "model": self.model})
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/chat":
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send_json({"error": "Message is required"}, status=HTTPStatus.BAD_REQUEST)
                return

            session_id = str(payload.get("session_id") or uuid4().hex)
            messages = self.sessions.setdefault(session_id, new_message_history())

            try:
                answer = run_conversation(
                    self.client,
                    self.csv_tool,
                    self.model,
                    message,
                    messages,
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            self._send_json({"session_id": session_id, "answer": answer})
            return

        if self.path == "/api/reset":
            old_session_id = str(payload.get("session_id") or "")
            if old_session_id:
                self.sessions.pop(old_session_id, None)
            session_id = uuid4().hex
            self.sessions[session_id] = new_message_history()
            self._send_json({"session_id": session_id})
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):
        print(f"[chat] {self.address_string()} - {format % args}")


def serve_chat(client: OpenAI, csv_tool: CSVTool, model: str, host: str, port: int):
    ChatHTTPHandler.client = client
    ChatHTTPHandler.csv_tool = csv_tool
    ChatHTTPHandler.model = model
    server = ThreadingHTTPServer((host, port), ChatHTTPHandler)
    print(f"Chat interface: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


def run_cli(client: OpenAI, csv_tool: CSVTool, model: str, data_dir: str, base_url: str):
    files = csv_tool.execute("list_files", {})
    file_names = [f["name"] for f in files.get("files", [])]
    print(f"Data directory: {data_dir}")
    print(f"CSV files found: {file_names}")
    print(f"LM Studio: {base_url}")
    print(f"Model: {model}")
    print()
    print("Type your questions below. Type 'quit' to exit.")
    print("=" * 60)
    print()

    messages = new_message_history()

    while True:
        try:
            question = input("Teacher: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        try:
            answer = run_conversation(
                client,
                csv_tool,
                model,
                question,
                messages,
                tool_logger=lambda fn_name, fn_args: print(
                    f"  [tool] {fn_name}({json.dumps(fn_args, ensure_ascii=False)})"
                ),
            )
            print(f"\nAssistant: {answer}\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")
            print("Make sure LM Studio is running and a model is loaded.\n")


def main():
    parser = argparse.ArgumentParser(description="LM Studio ↔ CSV Tool Bridge")
    parser.add_argument("--data-dir", default=".", help="Directory containing CSV files")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LM Studio API endpoint")
    parser.add_argument("--model", default="local-model", help="Model name in LM Studio")
    parser.add_argument("--serve-chat", action="store_true", help="Serve a simple browser chat interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the chat server to")
    parser.add_argument("--port", type=int, default=8000, help="Port for the chat server")

    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    csv_tool = CSVTool(data_dir=args.data_dir)

    if args.serve_chat:
        serve_chat(client, csv_tool, args.model, args.host, args.port)
        return

    run_cli(client, csv_tool, args.model, args.data_dir, args.base_url)


if __name__ == "__main__":
    main()
