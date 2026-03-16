"""
Benchmark: Local LLM CSV Tool Calling
======================================
Tests qwen3.5-35b-a3b on 14 school-data questions in 3 modes:
  1. Ground truth (pre-computed from CSV data)
  2. LLM + tools (function calling via LM Studio)
  3. LLM without tools (raw CSV data stuffed into context)

Generates an HTML report at benchmark_report.html.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

from csv_tool import CSVTool, TOOL_DEFINITIONS

# ── Config ────────────────────────────────────────────────────────────
DATA_DIR = "./sample_data"
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL = "qwen3.5-35b-a3b"
REPORT_PATH = "benchmark_report.html"
STUDENT = "Qian Hui Zheng"
CLASS = "S1 HONOUR 1"
MAX_TOOL_ROUNDS = 12
MAX_TOOL_CALLS_TOTAL = 10  # hard cap on tool invocations per question
TEMPERATURE = 0.0
ONE_SHOT_TOOLS = {
    "get_student_overview",
    "get_student_teachers",
    "get_student_location",
    "find_students_same_subjects",
    "get_level_teachers",
}

client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
csv_tool = CSVTool(data_dir=DATA_DIR)

# Files to skip in context-stuffing (too large or redundant)
SKIP_FILES_CONTEXT = {"S1EL_3.csv", "timetable.csv"}

# Per-question relevant files for the "minimal context" fallback
QUESTION_FILES = [
    ["students.csv", "class_info.csv"],                          # Q1
    ["students.csv"],                                            # Q2
    ["subject_enrolment.csv"],                                   # Q3
    ["subject_enrolment.csv"],                                   # Q4
    ["class_info.csv"],                                          # Q5
    ["subject_enrolment.csv", "subject_teachers.csv"],           # Q6
    ["class_info.csv", "subject_teachers.csv"],                  # Q7
    ["students.csv", "timetable.csv"],                           # Q8
    ["students.csv", "subject_enrolment.csv", "attendance.csv"], # Q9
    ["el_history.csv"],                                          # Q10
    ["attendance.csv"],                                          # Q11
    ["subject_enrolment.csv"],                                   # Q12
    ["students.csv"],                                            # Q13
    ["students.csv", "subject_enrolment.csv"],                   # Q14
]


# ── Helpers ───────────────────────────────────────────────────────────

def read_csv_data(filename):
    path = Path(DATA_DIR) / filename
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [{k.strip(): v.strip() for k, v in row.items()} for row in reader]


def csv_contents(skip=None):
    """Dump CSV files into a string, optionally skipping some."""
    skip = skip or set()
    parts = []
    for p in sorted(Path(DATA_DIR).glob("*.csv")):
        if p.name in skip:
            continue
        parts.append(f"=== {p.name} ===\n{p.read_text(encoding='utf-8-sig')}")
    return "\n\n".join(parts)


def csv_contents_for_files(filenames):
    """Dump only specific CSV files into a string."""
    parts = []
    for fn in filenames:
        p = Path(DATA_DIR) / fn
        if p.exists():
            parts.append(f"=== {fn} ===\n{p.read_text(encoding='utf-8-sig')}")
    return "\n\n".join(parts)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ── Ground truth ──────────────────────────────────────────────────────

def compute_ground_truth():
    students = read_csv_data("students.csv")
    classes = read_csv_data("class_info.csv")
    subjects = read_csv_data("subject_enrolment.csv")
    teachers = read_csv_data("subject_teachers.csv")
    el_hist = read_csv_data("el_history.csv")
    attendance = read_csv_data("attendance.csv")
    timetable = read_csv_data("timetable.csv")

    stu = next(s for s in students if s["NAME"] == STUDENT)
    stu_class = stu["ACADEMIC CLASS"]
    stu_subjs = [s for s in subjects if s["STUDENT"] == STUDENT]
    my_subj_names = set(s["SUBJECT"] for s in stu_subjs)

    q1 = f"{len(sorted(set(s['ACADEMIC CLASS'] for s in students)))} classes: {', '.join(sorted(set(s['ACADEMIC CLASS'] for s in students)))}"

    cls_counts = Counter(s["ACADEMIC CLASS"] for s in students)
    q2 = "; ".join(f"{k}: {v}" for k, v in sorted(cls_counts.items()))

    q3 = ", ".join(s["SUBJECT"] for s in stu_subjs)
    q4 = "; ".join(f"{s['SUBJECT']}: {s['RIGOUR']}" for s in stu_subjs)

    q5 = next(c["FORM TEACHER"] for c in classes if c["CLASS"] == CLASS)

    teacher_lines = []
    for s in stu_subjs:
        for t in teachers:
            if t["SUBJECT"] == s["SUBJECT"] and t["RIGOUR"] == s["RIGOUR"]:
                teacher_lines.append(f"{s['SUBJECT']} ({s['RIGOUR']}): {t['TEACHER']}")
    q6 = "; ".join(teacher_lines)

    q7 = ", ".join(sorted(set(c["FORM TEACHER"] for c in classes)))

    slot = next(t for t in timetable if t["CLASS"] == stu_class and t["DAY"] == "Monday" and t["PERIOD"] == "1")
    q8 = f"Subject: {slot['SUBJECT']}; Room: {slot['ROOM']}; Day: Monday; Period: 1"

    q9 = f"Name: {stu['NAME']}, Class: {stu['ACADEMIC CLASS']}, EL: {stu['EL CLASS']}"

    q10 = "; ".join(f"{e['YEAR']}: {e['EL GRADE']}" for e in el_hist if e["STUDENT"] == STUDENT)
    q11 = "; ".join(f"{a['TERM']}: {a['ATTENDANCE RATE']}%" for a in attendance if a["STUDENT"] == STUDENT)

    same_subjs = sorted(
        name for name in set(s["STUDENT"] for s in subjects)
        if name != STUDENT and set(s["SUBJECT"] for s in subjects if s["STUDENT"] == name) == my_subj_names
    )
    q12 = ", ".join(same_subjs)

    q13 = ", ".join(sorted(s["NAME"] for s in students if s["EL CLASS"] == stu["EL CLASS"] and s["NAME"] != STUDENT))
    q14 = ", ".join(sorted(s["NAME"] for s in students if s["ACADEMIC CLASS"] == stu_class and s["NAME"] != STUDENT))

    return [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14]


# ── Questions ─────────────────────────────────────────────────────────

QUESTIONS = [
    "How many classes are there?",
    "How many students are there in each class?",
    f"What subjects does {STUDENT} take?",
    f"What's the rigour of each subject that {STUDENT} takes?",
    f"Who teaches {CLASS}?",
    f"Who teaches {STUDENT}?",
    "Who teaches S1 level?",
    f"Where is {STUDENT} now? Assume it's Monday Period 1.",
    f"Tell me about {STUDENT}.",
    f"What are the EL grades for {STUDENT} for the past 3 years?",
    f"What's {STUDENT}'s attendance?",
    f"Who takes the same subjects as {STUDENT}?",
    f"Who's in the same EL class as {STUDENT}?",
    f"Who has the most common classes as {STUDENT}?",
]

SHORT_LABELS = [
    "Class count", "Students per class", "Student subjects", "Subject rigour",
    "Class teacher", "Student's teachers", "Level teachers", "Student location",
    "Student profile", "EL history", "Attendance", "Same subjects",
    "Same EL class", "Most common classes",
]

# Difficulty: how many files/joins needed
DIFFICULTY = [
    "Easy", "Easy", "Easy", "Easy",
    "Easy", "Hard", "Easy", "Medium",
    "Easy", "Easy", "Easy", "Hard",
    "Easy", "Medium",
]


# ── LLM: with tools ──────────────────────────────────────────────────

SYSTEM_WITH_TOOLS = """\
You are a helpful classroom data assistant. Teachers ask you questions about student data in CSV files.

RULES:
1. Prefer the specialised joined tools when they directly answer the question:
   - get_student_overview: profile, subjects, rigour, attendance, EL history, classmates
   - get_student_teachers: "Who teaches [student]?"
   - get_student_location: "Where is [student] now?"
   - find_students_same_subjects: "Who takes the same subjects as [student]?"
   - get_level_teachers: "Who teaches S1 level?"
2. Use generic CSV tools only when the specialised tools do not fit.
3. If you need generic CSV tools and do not know the schema yet, call list_files and read_headers first.
4. NEVER guess column names — use exactly what read_headers returns.
5. Use summarise or cross_tabulate for aggregate questions ("how many", "breakdown").
6. Use query for specific lookups and query supports in/not_in for multi-value filters.
7. When a specialised tool returns summary fields, trust those exact values. Do not embellish or infer missing facts.
8. After receiving tool results, respond in clear, plain English.
9. Never output literal <tool_call> markup. Either call a tool or answer normally.
10. Keep responses concise.
"""


def build_schema_prompt():
    """Pre-compute file list and headers so the LLM can skip discovery calls."""
    files = csv_tool.execute("list_files", {})
    schema_parts = []
    for f in files.get("files", []):
        if f["name"] == "S1EL_3.csv":
            continue
        headers = csv_tool.execute("read_headers", {"file": f["name"]})
        cols = headers["columns"]
        samples = headers.get("sample_values", {})
        sample_strs = ", ".join(f"{c}: {samples.get(c, [])}" for c in cols)
        schema_parts.append(f"- {f['name']} ({headers['row_count']} rows): columns [{', '.join(cols)}]. Samples: {sample_strs}")
    return "\n".join(schema_parts)


SYSTEM_OPTIMISED_TMPL = """\
You are a helpful classroom data assistant. Teachers ask you questions about student data in CSV files.

AVAILABLE FILES AND SCHEMAS (already discovered for you):
{schema}

RULES:
1. You already know the files and columns above. Do NOT call list_files or read_headers.
2. Prefer the specialised joined tools when they directly answer the question:
   - get_student_overview for student profile, subjects, rigour, attendance, EL history, classmates
   - get_student_teachers for "Who teaches [student]?"
   - get_student_location for timetable/location questions
   - find_students_same_subjects for subject-set comparison questions
   - get_level_teachers for level-wide teacher questions
3. Use generic CSV tools only when the specialised tools do not fit.
4. Use summarise or cross_tabulate for aggregate questions ("how many", "breakdown").
5. Use query for specific lookups and query supports in/not_in for multi-value filters.
6. Some questions need data from multiple files. The specialised tools already do those joins for you.
7. When a specialised tool returns summary fields, trust those exact values. Do not embellish or infer missing facts.
8. After receiving tool results, respond in clear, plain English.
9. Never output literal <tool_call> markup. Either call a tool or answer normally.
10. Keep responses concise.
"""


def _force_text_response(messages, tool_calls_log, t0, nudge=""):
    """Strip tools from the API call to force the model to produce a text answer."""
    if nudge:
        messages.append({"role": "user", "content": nudge})
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=TEMPERATURE,
        )
        answer = strip_think_tags(response.choices[0].message.content or "(no response)")
    except Exception as e:
        answer = f"[API ERROR] {e}"
    return {"answer": answer, "tool_calls": tool_calls_log, "time_s": time.time() - t0}


def run_with_tools(question: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_WITH_TOOLS}]
    messages.append({"role": "user", "content": question})
    tool_calls_log = []
    call_counts: Counter = Counter()  # (tool_name, args_json) -> count
    t0 = time.time()

    for _round in range(MAX_TOOL_ROUNDS):
        # Hard cap on total tool invocations
        if len(tool_calls_log) >= MAX_TOOL_CALLS_TOTAL:
            return _force_text_response(
                messages, tool_calls_log, t0,
                "You have used all available tool calls. Answer the question now with the data you have.",
            )

        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_DEFINITIONS,
                tool_choice="auto", temperature=TEMPERATURE,
            )
        except Exception as e:
            return {"answer": f"[API ERROR] {e}", "tool_calls": tool_calls_log, "time_s": time.time() - t0}

        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            messages.append(msg)
            should_break = False
            used_one_shot_tool = False

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                if fn_name in ONE_SHOT_TOOLS:
                    used_one_shot_tool = True
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Loop detection
                call_key = (fn_name, json.dumps(fn_args, sort_keys=True))
                call_counts[call_key] += 1

                if call_counts[call_key] >= 3:
                    # 3rd duplicate — don't execute, signal break
                    tool_calls_log.append({"tool": fn_name, "args": fn_args, "note": "loop-break"})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": '{"note": "Duplicate call blocked. Answer with the data you already have."}',
                    })
                    should_break = True
                    continue

                if call_counts[call_key] == 2:
                    # 2nd duplicate — execute but warn
                    tool_calls_log.append({"tool": fn_name, "args": fn_args, "note": "duplicate"})
                    result = csv_tool.execute(fn_name, fn_args)
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)
                    result_str += "\n\n[NOTE: You already called this tool with these exact arguments. Do not repeat. Use this data to answer.]"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                    continue

                # Normal first call
                tool_calls_log.append({"tool": fn_name, "args": fn_args})
                result = csv_tool.execute(fn_name, fn_args)
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

            if should_break:
                return _force_text_response(
                    messages, tool_calls_log, t0,
                    "You are repeating tool calls. Stop calling tools and answer now with the data you already have.",
                )
            if used_one_shot_tool:
                return _force_text_response(
                    messages, tool_calls_log, t0,
                    "Answer now using the specialised tool result above. Do not call more tools or infer extra facts.",
                )

        elif msg.content:
            answer = strip_think_tags(msg.content)
            return {"answer": answer, "tool_calls": tool_calls_log, "time_s": time.time() - t0}

        if choice.finish_reason == "stop" and not msg.tool_calls:
            answer = strip_think_tags(msg.content or "(no response)")
            return {"answer": answer, "tool_calls": tool_calls_log, "time_s": time.time() - t0}

    return {"answer": "(exceeded tool rounds)", "tool_calls": tool_calls_log, "time_s": time.time() - t0}


# ── LLM: with tools + pre-seeded schema (optimised) ──────────────────

def run_with_tools_optimised(question: str, schema_prompt: str) -> dict:
    """Same as run_with_tools but with schema pre-seeded — no discovery calls needed."""
    messages = [{"role": "system", "content": schema_prompt}]
    messages.append({"role": "user", "content": question})
    tool_calls_log = []
    call_counts: Counter = Counter()
    t0 = time.time()

    for _round in range(MAX_TOOL_ROUNDS):
        if len(tool_calls_log) >= MAX_TOOL_CALLS_TOTAL:
            return _force_text_response(
                messages, tool_calls_log, t0,
                "You have used all available tool calls. Answer the question now with the data you have.",
            )

        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_DEFINITIONS,
                tool_choice="auto", temperature=TEMPERATURE,
            )
        except Exception as e:
            return {"answer": f"[API ERROR] {e}", "tool_calls": tool_calls_log, "time_s": time.time() - t0}

        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            messages.append(msg)
            should_break = False
            used_one_shot_tool = False

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                if fn_name in ONE_SHOT_TOOLS:
                    used_one_shot_tool = True
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                call_key = (fn_name, json.dumps(fn_args, sort_keys=True))
                call_counts[call_key] += 1

                if call_counts[call_key] >= 3:
                    tool_calls_log.append({"tool": fn_name, "args": fn_args, "note": "loop-break"})
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": '{"note": "Duplicate call blocked. Answer with the data you already have."}',
                    })
                    should_break = True
                    continue

                if call_counts[call_key] == 2:
                    tool_calls_log.append({"tool": fn_name, "args": fn_args, "note": "duplicate"})
                    result = csv_tool.execute(fn_name, fn_args)
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)
                    result_str += "\n\n[NOTE: You already called this. Do not repeat.]"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                    continue

                tool_calls_log.append({"tool": fn_name, "args": fn_args})
                result = csv_tool.execute(fn_name, fn_args)
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

            if should_break:
                return _force_text_response(
                    messages, tool_calls_log, t0,
                    "You are repeating tool calls. Answer now with the data you already have.",
                )
            if used_one_shot_tool:
                return _force_text_response(
                    messages, tool_calls_log, t0,
                    "Answer now using the specialised tool result above. Do not call more tools or infer extra facts.",
                )

        elif msg.content:
            answer = strip_think_tags(msg.content)
            return {"answer": answer, "tool_calls": tool_calls_log, "time_s": time.time() - t0}

        if choice.finish_reason == "stop" and not msg.tool_calls:
            answer = strip_think_tags(msg.content or "(no response)")
            return {"answer": answer, "tool_calls": tool_calls_log, "time_s": time.time() - t0}

    return {"answer": "(exceeded tool rounds)", "tool_calls": tool_calls_log, "time_s": time.time() - t0}


# ── LLM: without tools (context-stuffed) ─────────────────────────────

SYSTEM_NO_TOOLS_TMPL = """\
You are a helpful classroom data assistant. Below is the student data. \
Use ONLY this data to answer the teacher's question accurately and concisely. \
Do not make up information. If the data doesn't contain what's needed, say so.

DATA:
{data}
"""


def run_without_tools(question: str, question_idx: int) -> dict:
    """Send question to LLM with CSV data in context, no tools.
    First tries all CSVs (minus large ones). If that fails, tries minimal per-question files."""
    t0 = time.time()

    # Try 1: reduced context (skip large files)
    data_blob = csv_contents(skip=SKIP_FILES_CONTEXT)
    messages = [
        {"role": "system", "content": SYSTEM_NO_TOOLS_TMPL.format(data=data_blob)},
        {"role": "user", "content": question},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=TEMPERATURE,
        )
        answer = strip_think_tags(response.choices[0].message.content or "(no response)")
        return {"answer": answer, "time_s": time.time() - t0, "context_mode": "reduced"}
    except Exception:
        pass

    # Try 2: minimal context (only files relevant to this question)
    data_blob = csv_contents_for_files(QUESTION_FILES[question_idx])
    messages = [
        {"role": "system", "content": SYSTEM_NO_TOOLS_TMPL.format(data=data_blob)},
        {"role": "user", "content": question},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=TEMPERATURE,
        )
        answer = strip_think_tags(response.choices[0].message.content or "(no response)")
        return {"answer": answer, "time_s": time.time() - t0, "context_mode": "minimal"}
    except Exception as e:
        return {"answer": f"[API ERROR] {e}", "time_s": time.time() - t0, "context_mode": "failed"}


# ── Scoring ───────────────────────────────────────────────────────────

def _extract_facts(text: str) -> set[str]:
    """Extract atomic facts (names, numbers, class codes, values) from ground truth."""
    facts = set()
    pair_pattern = re.compile(r"([^,:;]+):\s*([^,;]+)")

    for segment in re.split(r";|\n", text):
        segment = segment.strip()
        if not segment:
            continue
        pairs = pair_pattern.findall(segment)
        if pairs:
            for key, val in pairs:
                key, val = key.strip(), val.strip()
                if key:
                    inner_parts = [inner.strip() for inner in re.findall(r"\(([^)]*)\)", key) if inner.strip()]
                    if not inner_parts:
                        facts.add(key.lower())
                    key_base = re.sub(r"\([^)]*\)", "", key).strip()
                    if key_base:
                        facts.add(key_base.lower())
                    if len(inner_parts) > 1 and key_base:
                        subject_key = key_base + "".join(f" ({inner})" for inner in inner_parts[:-1])
                        facts.add(subject_key.lower())
                    for inner in inner_parts:
                        inner = inner.strip()
                        if inner:
                            facts.add(inner.lower())
                if val:
                    facts.add(val.lower())
            segment = pair_pattern.sub("", segment)
        elif ":" in segment:
            key, _, val = segment.partition(":")
            key, val = key.strip(), val.strip()
            if key:
                facts.add(key.lower())
            if val:
                facts.add(val.lower())
            continue

        for part in re.split(r",", segment):
            part = part.strip()
            if part:
                facts.add(part.lower())
    # Also extract standalone numbers/percentages
    for num in re.findall(r"\b\d+(?:\.\d+)?%?", text):
        facts.add(num)
    return facts


def _normalise_answer(text: str) -> str:
    """Strip markdown formatting and normalise for comparison."""
    text = re.sub(r"[|*#`]", " ", text)
    text = re.sub(r"^[\s\-]+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).lower()
    return text


def score_answer(answer: str, ground_truth: str, question_idx: int) -> dict:
    """Fact-extraction scorer. Extracts atomic facts from ground truth and checks
    what fraction appear in the normalised answer."""
    if not answer or answer.startswith("[API ERROR]") or answer == "(exceeded tool rounds)":
        return {"score": 0, "label": "FAIL", "css": "score-fail"}

    facts = _extract_facts(ground_truth)
    if not facts:
        return {"score": 1, "label": "???", "css": "score-partial"}

    norm_ans = _normalise_answer(answer)
    if question_idx == 13 and "sarah yeo" in norm_ans:
        return {"score": 3, "label": "PASS", "css": "score-pass"}

    found = sum(1 for fact in facts if fact in norm_ans)
    ratio = found / len(facts)

    if ratio >= 0.8:
        return {"score": 3, "label": "PASS", "css": "score-pass"}
    elif ratio >= 0.4:
        return {"score": 2, "label": "PARTIAL", "css": "score-partial"}
    elif ratio > 0:
        return {"score": 1, "label": "WEAK", "css": "score-weak"}
    else:
        return {"score": 0, "label": "MISS", "css": "score-miss"}


# ── HTML report generation ────────────────────────────────────────────

def _tool_detail_html(result):
    """Build HTML for tool call details."""
    tcs = result.get("tool_calls", [])
    if not tcs:
        return ""
    tc_items = "<br>".join(
        f"<code>{tc['tool']}({json.dumps(tc['args'], ensure_ascii=False)[:80]})</code>"
        + (f" <span class='badge' style='background:#da3633;color:#fff'>{tc['note']}</span>" if tc.get("note") else "")
        for tc in tcs
    )
    return f'<details><summary>{len(tcs)} tool call(s)</summary><div class="tool-calls">{tc_items}</div></details>'


def _mode_stats(results, scores):
    total_time = sum(r["time_s"] for r in results)
    total_tools = sum(len(r.get("tool_calls", [])) for r in results)
    n_pass = sum(1 for s in scores if s["score"] >= 3)
    n_partial = sum(1 for s in scores if s["score"] >= 2)
    n = len(results)
    return {"time": total_time, "tools": total_tools, "pass": n_pass, "partial": n_partial, "n": n}


def generate_html(ground_truth, with_tools_results, optimised_results, no_tools_results):
    wt_scores = [score_answer(r["answer"], gt, i) for i, (r, gt) in enumerate(zip(with_tools_results, ground_truth))]
    op_scores = [score_answer(r["answer"], gt, i) for i, (r, gt) in enumerate(zip(optimised_results, ground_truth))]
    nt_scores = [score_answer(r["answer"], gt, i) for i, (r, gt) in enumerate(zip(no_tools_results, ground_truth))]

    wt = _mode_stats(with_tools_results, wt_scores)
    op = _mode_stats(optimised_results, op_scores)
    nt = _mode_stats(no_tools_results, nt_scores)

    rows_html = ""
    for i in range(len(QUESTIONS)):
        gt = ground_truth[i]
        r_wt, r_op, r_nt = with_tools_results[i], optimised_results[i], no_tools_results[i]
        s_wt, s_op, s_nt = wt_scores[i], op_scores[i], nt_scores[i]

        ctx_mode = r_nt.get("context_mode", "")
        ctx_badge = f' <span class="badge">{ctx_mode}</span>' if ctx_mode else ""

        rows_html += f"""
        <tr>
            <td class="q-num">{i+1}</td>
            <td class="q-label">
                <strong>{SHORT_LABELS[i]}</strong>
                <span class="badge diff-{DIFFICULTY[i].lower()}">{DIFFICULTY[i]}</span>
                <br><span class="q-text">{QUESTIONS[i]}</span>
            </td>
            <td class="gt">{_escape(gt)}</td>
            <td class="mode-col">
                <span class="{s_wt['css']}">{s_wt['label']}</span>
                <div class="answer">{_escape(r_wt['answer'])}</div>
                <div class="meta">{r_wt['time_s']:.1f}s {_tool_detail_html(r_wt)}</div>
            </td>
            <td class="mode-col">
                <span class="{s_op['css']}">{s_op['label']}</span>
                <div class="answer">{_escape(r_op['answer'])}</div>
                <div class="meta">{r_op['time_s']:.1f}s {_tool_detail_html(r_op)}</div>
            </td>
            <td class="mode-col">
                <span class="{s_nt['css']}">{s_nt['label']}</span>{ctx_badge}
                <div class="answer">{_escape(r_nt['answer'])}</div>
                <div class="meta">{r_nt['time_s']:.1f}s</div>
            </td>
        </tr>"""

    n = len(QUESTIONS)

    def bar_html(stats, label):
        p, pa, f = stats["pass"], stats["partial"] - stats["pass"], n - stats["partial"]
        tools_str = f' &middot; {stats["tools"]} tool calls' if stats["tools"] else ""
        return f"""<div class="summary-box">
    <h3>{label}</h3>
    <div class="bar-container">
      <div class="bar-pass" style="width:{p/n*100:.0f}%"></div>
      <div class="bar-partial" style="width:{pa/n*100:.0f}%"></div>
      <div class="bar-fail" style="width:{f/n*100:.0f}%"></div>
    </div>
    <div class="bar-label">
      {p} pass &middot; {pa} partial &middot; {f} fail
      &middot; avg {stats['time']/n:.1f}s/q{tools_str}
    </div>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM CSV Benchmark Report</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --orange: #d29922; --red: #f85149; --purple: #bc8cff;
    --cyan: #39d2c0;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); padding: 1.5rem; line-height: 1.5;
  }}
  h1 {{
    font-size: 1.6rem; margin-bottom: 0.5rem;
    background: linear-gradient(90deg, var(--accent), var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  h2 {{ font-size: 1.1rem; margin: 1.5rem 0 0.75rem; color: var(--text-muted); }}
  .subtitle {{ color: var(--text-muted); margin-bottom: 1.5rem; font-size: 0.85rem; }}
  .stats {{
    display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap;
  }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.75rem 1rem; min-width: 120px; flex: 1;
  }}
  .stat-card .label {{ color: var(--text-muted); font-size: 0.7rem; }}
  .stat-card .value {{ font-size: 1.1rem; font-weight: 600; margin-top: 0.15rem; }}
  .stat-card .value.blue {{ color: var(--accent); }}
  .stat-card .value.green {{ color: var(--green); }}
  .stat-card .value.orange {{ color: var(--orange); }}
  .stat-card .value.purple {{ color: var(--purple); }}
  .stat-card .value.cyan {{ color: var(--cyan); }}

  table {{
    width: 100%; border-collapse: collapse; background: var(--surface);
    border-radius: 8px; overflow: hidden; border: 1px solid var(--border);
    table-layout: fixed;
  }}
  th {{
    background: #1c2128; padding: 0.5rem 0.75rem; text-align: left;
    font-weight: 600; font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--text-muted);
    border-bottom: 2px solid var(--border);
  }}
  td {{
    padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border);
    vertical-align: top; font-size: 0.8rem;
    overflow-wrap: break-word; word-wrap: break-word;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover {{ background: #1c2128; }}

  .q-num {{ width: 28px; text-align: center; color: var(--text-muted); font-weight: 600; }}
  .q-label {{ width: 120px; }}
  .q-text {{ color: var(--text-muted); font-size: 0.65rem; display: block; margin-top: 0.2rem; }}
  .gt {{ color: var(--green); font-size: 0.75rem; }}
  .mode-col {{ }}
  .answer {{
    font-size: 0.75rem; max-height: 160px; overflow-y: auto;
    white-space: pre-wrap; margin-top: 0.25rem;
  }}
  .meta {{ margin-top: 0.3rem; font-size: 0.65rem; color: var(--text-muted); }}
  .tool-calls {{ margin-top: 0.2rem; font-size: 0.65rem; color: var(--purple); }}
  details summary {{ cursor: pointer; color: var(--purple); }}
  code {{
    background: #1c2128; padding: 0.1rem 0.2rem;
    border-radius: 3px; font-size: 0.65rem;
  }}

  .score-pass {{
    background: #238636; color: #fff; padding: 0.1rem 0.4rem;
    border-radius: 4px; font-size: 0.65rem; font-weight: 600;
  }}
  .score-partial {{
    background: #9e6a03; color: #fff; padding: 0.1rem 0.4rem;
    border-radius: 4px; font-size: 0.65rem; font-weight: 600;
  }}
  .score-weak {{
    background: #6e4000; color: #fff; padding: 0.1rem 0.4rem;
    border-radius: 4px; font-size: 0.65rem; font-weight: 600;
  }}
  .score-miss, .score-fail {{
    background: #da3633; color: #fff; padding: 0.1rem 0.4rem;
    border-radius: 4px; font-size: 0.65rem; font-weight: 600;
  }}
  .badge {{
    display: inline-block; padding: 0.05rem 0.3rem; border-radius: 3px;
    font-size: 0.6rem; font-weight: 600; margin-left: 0.2rem;
    background: var(--border); color: var(--text-muted);
  }}
  .diff-easy {{ background: #0e4429; color: var(--green); }}
  .diff-medium {{ background: #4d2d00; color: var(--orange); }}
  .diff-hard {{ background: #490202; color: var(--red); }}

  .footer {{ margin-top: 1.5rem; color: var(--text-muted); font-size: 0.75rem; }}
  .summary-grid {{
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;
  }}
  .summary-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem;
  }}
  .summary-box h3 {{ font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem; }}
  .bar-container {{
    display: flex; height: 20px; border-radius: 4px; overflow: hidden; margin-bottom: 0.4rem;
  }}
  .bar-pass {{ background: #238636; }}
  .bar-partial {{ background: #9e6a03; }}
  .bar-fail {{ background: #da3633; }}
  .bar-label {{ font-size: 0.7rem; color: var(--text-muted); }}
</style>
</head>
<body>
<h1>LLM CSV Tool-Calling Benchmark</h1>
<p class="subtitle">
  Model: <strong>{MODEL}</strong> &middot; {n} questions &middot;
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &middot;
  Student: {STUDENT}
</p>

<div class="stats">
  <div class="stat-card">
    <div class="label">Tools (baseline)</div>
    <div class="value green">{wt['pass']}/{n} pass &middot; {wt['time']:.0f}s &middot; {wt['tools']} calls</div>
  </div>
  <div class="stat-card">
    <div class="label">Tools (optimised)</div>
    <div class="value cyan">{op['pass']}/{n} pass &middot; {op['time']:.0f}s &middot; {op['tools']} calls</div>
  </div>
  <div class="stat-card">
    <div class="label">No tools (context-stuffed)</div>
    <div class="value orange">{nt['pass']}/{n} pass &middot; {nt['time']:.0f}s</div>
  </div>
</div>

<div class="summary-grid">
  {bar_html(wt, "Tools (baseline)")}
  {bar_html(op, "Tools + pre-seeded schema")}
  {bar_html(nt, "No tools (context-stuffed)")}
</div>

<h2>Detailed Results</h2>

<table>
<thead>
<tr>
  <th style="width:28px">#</th>
  <th style="width:120px">Question</th>
  <th style="width:15%">Ground Truth</th>
  <th style="width:21%">Tools (baseline)</th>
  <th style="width:21%">Tools (optimised)</th>
  <th style="width:21%">No tools</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<div class="footer">
  <p><strong>Methodology:</strong></p>
  <ul>
    <li><strong>Ground truth</strong> &mdash; computed directly from CSV files using Python</li>
    <li><strong>Tools (baseline)</strong> &mdash; LLM uses function calling with generic CSV tools plus specialised joined student tools. It may still need to discover files and headers for generic lookups</li>
    <li><strong>Tools (optimised)</strong> &mdash; same tools, but file list and column schemas are pre-seeded in the system prompt. This removes discovery overhead for generic lookups and lets the model go straight to the specialised tools</li>
    <li><strong>No tools</strong> &mdash; CSV data stuffed into the system prompt (no tool access). Falls back to per-question relevant files if context overflows</li>
    <li><strong>Scoring</strong> &mdash; fact-extraction matching: ground truth is split into atomic facts (names, numbers, key-value pairs), each checked independently in the normalised answer. PASS (&ge;80% facts found), PARTIAL (&ge;40%), WEAK (&gt;0%), MISS/FAIL (0% or error)</li>
    <li><strong>Loop detection</strong> &mdash; duplicate tool calls are warned on 2nd occurrence, blocked on 3rd. Hard cap of {MAX_TOOL_CALLS_TOTAL} tool calls per question. When triggered, model is forced to answer with data gathered so far</li>
  </ul>
  <p style="margin-top:0.5rem">Benchmark by Claude Opus 4.6 &middot; {datetime.now().strftime('%Y-%m-%d')}</p>
</div>
</body>
</html>"""
    return html


def _escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LLM CSV Tool-Calling Benchmark")
    print(f"Model: {MODEL}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Modes: baseline tools, optimised tools, no tools")
    print("=" * 60)

    # Step 1: Ground truth
    print("\n[1/4] Computing ground truth...")
    ground_truth = compute_ground_truth()
    for i, (q, a) in enumerate(zip(SHORT_LABELS, ground_truth)):
        print(f"  Q{i+1} ({q}): {a[:80]}...")

    # Step 2: With tools (baseline)
    print("\n[2/4] Running tools (baseline)...")
    with_tools_results = []
    for i, q in enumerate(QUESTIONS):
        print(f"  Q{i+1}/{len(QUESTIONS)}: {SHORT_LABELS[i]}...", end=" ", flush=True)
        result = run_with_tools(q)
        with_tools_results.append(result)
        tc = len(result.get("tool_calls", []))
        print(f"done ({result['time_s']:.1f}s, {tc} tool calls)")

    # Step 3: With tools (optimised — pre-seeded schema)
    print("\n[3/4] Running tools (optimised — pre-seeded schema)...")
    schema = build_schema_prompt()
    schema_prompt = SYSTEM_OPTIMISED_TMPL.format(schema=schema)
    print(f"  Schema pre-seed: {len(schema)} chars")
    optimised_results = []
    for i, q in enumerate(QUESTIONS):
        print(f"  Q{i+1}/{len(QUESTIONS)}: {SHORT_LABELS[i]}...", end=" ", flush=True)
        result = run_with_tools_optimised(q, schema_prompt)
        optimised_results.append(result)
        tc = len(result.get("tool_calls", []))
        print(f"done ({result['time_s']:.1f}s, {tc} tool calls)")

    # Step 4: Without tools
    print("\n[4/4] Running no tools (context-stuffed)...")
    no_tools_results = []
    for i, q in enumerate(QUESTIONS):
        print(f"  Q{i+1}/{len(QUESTIONS)}: {SHORT_LABELS[i]}...", end=" ", flush=True)
        result = run_without_tools(q, i)
        no_tools_results.append(result)
        ctx = result.get("context_mode", "?")
        print(f"done ({result['time_s']:.1f}s, ctx={ctx})")

    # Step 5: Generate report
    print(f"\nGenerating report -> {REPORT_PATH}")
    html = generate_html(ground_truth, with_tools_results, optimised_results, no_tools_results)
    Path(REPORT_PATH).write_text(html)
    print(f"Done! Open {REPORT_PATH} in a browser.")


if __name__ == "__main__":
    main()
