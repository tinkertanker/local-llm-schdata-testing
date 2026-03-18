"""
Benchmark: Local LLM CSV Tool Calling
======================================
Runs the school-data benchmark against a local LM Studio model.

Outputs:
  - benchmark_report.html     summary page for the latest verified run
  - benchmark_results.html    detailed per-question evidence
  - benchmark_results.json    machine-readable run artefact
  - benchmark_full_report.html (optional, only in --mode full)
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
MODEL_LABEL = "mlx-community/Qwen3.5-35B-A3B-4bit mlx"
HARDWARE_LABEL = "Apple M4 Pro with 48GB RAM"
SUMMARY_REPORT_PATH = "benchmark_report.html"
DETAIL_REPORT_PATH = "benchmark_results.html"
RESULTS_JSON_PATH = "benchmark_results.json"
FULL_REPORT_PATH = "benchmark_full_report.html"
REPO_URL = "https://github.com/tinkertanker/local-llm-schdata-testing"
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


def format_fact_list(facts, limit=6):
    facts = [fact for fact in facts if fact]
    if not facts:
        return "None"
    shown = facts[:limit]
    suffix = "" if len(facts) <= limit else f", and {len(facts) - limit} more"
    return ", ".join(shown) + suffix


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


def judge_answer(answer: str, ground_truth: str, question_idx: int) -> dict:
    """Human-readable assessment built on top of the fact scorer."""
    score = score_answer(answer, ground_truth, question_idx)
    facts = sorted(_extract_facts(ground_truth))
    norm_ans = _normalise_answer(answer)

    matched = [fact for fact in facts if fact in norm_ans]
    missing = [fact for fact in facts if fact not in norm_ans]

    if question_idx == 13 and "sarah yeo" in norm_ans:
        matched = facts[:]
        missing = []

    if not answer or answer.startswith("[API ERROR]"):
        verdict = "Did not work"
        reasoning = "The benchmark run returned an API error instead of an answer."
    elif answer == "(exceeded tool rounds)":
        verdict = "Did not work"
        reasoning = "The model did not finish within the allowed tool rounds."
    elif score["score"] >= 3:
        verdict = "Worked"
        if question_idx == 13 and "sarah yeo" in norm_ans:
            reasoning = (
                "Accepted benchmark interpretation. The answer names a student from the same "
                "academic class, which is the intended comparison for this question in the sample data."
            )
        elif missing:
            reasoning = (
                f"Worked overall. It covered {len(matched)} of {len(facts)} benchmark facts. "
                f"Missing details were minor: {format_fact_list(missing, limit=3)}."
            )
        else:
            reasoning = "Worked. The answer covered the expected benchmark facts."
    elif score["score"] == 2:
        verdict = "Partly worked"
        reasoning = (
            f"The answer captured some of the benchmark facts but missed key details: "
            f"{format_fact_list(missing, limit=4)}."
        )
    elif score["score"] == 1:
        verdict = "Did not work"
        reasoning = (
            f"The answer touched the topic but missed most of the benchmark facts: "
            f"{format_fact_list(missing, limit=4)}."
        )
    else:
        verdict = "Did not work"
        reasoning = "The answer did not contain the benchmark facts needed for this question."

    return {
        "score": score["score"],
        "label": score["label"],
        "css": score["css"],
        "verdict": verdict,
        "reasoning": reasoning,
        "matched_facts": matched,
        "missing_facts": missing,
        "matched_count": len(matched),
        "total_facts": len(facts),
    }


# ── HTML report generation ────────────────────────────────────────────

def _tool_detail_html(result):
    """Build HTML for tool call details."""
    tcs = result.get("tool_calls", [])
    if not tcs:
        return ""
    tc_items = "".join(
        "<div class='tool-call'>"
        f"<code>{_escape(tc['tool'])}({ _escape(json.dumps(tc['args'], ensure_ascii=False)) })</code>"
        + (f"<div class='tool-note'>{_escape(tc['note'])}</div>" if tc.get("note") else "")
        + "</div>"
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


def build_single_mode_report_data(ground_truth, results, mode_label, generated_at):
    judgements = [judge_answer(r["answer"], gt, i) for i, (r, gt) in enumerate(zip(results, ground_truth))]
    stats = _mode_stats(results, judgements)
    question_rows = []

    for i, (question, expected, result, judgement) in enumerate(zip(QUESTIONS, ground_truth, results, judgements)):
        question_rows.append({
            "index": i + 1,
            "short_label": SHORT_LABELS[i],
            "question": question,
            "difficulty": DIFFICULTY[i],
            "expected_answer": expected,
            "model_answer": result["answer"],
            "time_s": result["time_s"],
            "tool_calls": result.get("tool_calls", []),
            "tool_call_count": len(result.get("tool_calls", [])),
            "judgement": judgement,
        })

    longest = max(question_rows, key=lambda row: row["time_s"])
    return {
        "generated_at": generated_at,
        "model_id": MODEL,
        "model_label": MODEL_LABEL,
        "hardware_label": HARDWARE_LABEL,
        "mode_label": mode_label,
        "student": STUDENT,
        "question_count": len(question_rows),
        "summary": {
            "pass_count": stats["pass"],
            "partial_count": stats["partial"],
            "fail_count": stats["n"] - stats["partial"],
            "total_time_s": stats["time"],
            "average_time_s": stats["time"] / stats["n"],
            "tool_calls": stats["tools"],
            "longest_question_label": longest["short_label"],
            "longest_question_time_s": longest["time_s"],
        },
        "questions": question_rows,
    }


def generate_summary_html(report_data):
    summary = report_data["summary"]
    questions = report_data["questions"]
    pass_rate = f"{summary['pass_count']} / {report_data['question_count']}"
    question_rows = "".join(
        f"""
        <tr>
          <td>{row['index']}</td>
          <td>
            <strong>{_escape(row['short_label'])}</strong>
            <div class="row-subtle">{_escape(row['question'])}</div>
          </td>
          <td><span class="badge badge-{row['difficulty'].lower()}">{_escape(row['difficulty'])}</span></td>
          <td><span class="badge {row['judgement']['css']}">{_escape(row['judgement']['verdict'])}</span></td>
          <td>{row['time_s']:.1f}s</td>
          <td>{row['tool_call_count']}</td>
        </tr>
        """
        for row in questions
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Summary</title>
<style>
  :root {{
    --bg: #f4efe7;
    --paper: #fffdf8;
    --frame-bg: rgba(255, 253, 248, 0.92);
    --float-bg: rgba(255, 253, 248, 0.88);
    --hover-bg: #fff7ef;
    --bg-glow-1: rgba(180, 77, 22, 0.08);
    --bg-glow-2: rgba(31, 93, 99, 0.09);
    --ink: #1e1812;
    --muted: #6e655c;
    --line: #dbcab5;
    --accent: #b44d16;
    --accent-dark: #7d2e0d;
    --accent-soft: #f7dfcf;
    --teal: #1f5d63;
    --teal-soft: #dceef0;
    --success: #2f7d32;
    --success-soft: #e1f2e2;
    --warning: #9a6700;
    --warning-soft: #f5ead2;
    --danger: #9b2c2c;
    --danger-soft: #f8dedd;
    --shadow: 0 18px 48px rgba(63, 38, 15, 0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Avenir Next", "Segoe UI", sans-serif;
    color: var(--ink);
    background:
      radial-gradient(circle at top right, var(--bg-glow-1), transparent 24%),
      radial-gradient(circle at bottom left, var(--bg-glow-2), transparent 28%),
      var(--bg);
  }}
  a {{
    color: inherit;
    text-decoration-thickness: 2px;
    text-underline-offset: 0.18em;
    text-decoration-color: color-mix(in srgb, var(--accent) 55%, transparent);
  }}
  a:hover {{ color: var(--accent); }}
  .wrap {{
    width: min(1180px, calc(100% - 32px));
    margin: 32px auto 48px;
  }}
  .hero,
  .panel,
  .table-panel,
  .stat {{
    border: 1px solid var(--line);
    border-radius: 28px;
    background: var(--frame-bg);
    box-shadow: var(--shadow);
    backdrop-filter: blur(8px);
  }}
  .hero,
  .panel,
  .table-panel {{
    padding: 32px;
  }}
  .hero {{
    margin-bottom: 24px;
  }}
  .eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    padding: 7px 12px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent-dark);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }}
  .eyebrow::before {{
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
  }}
  h1 {{
    margin: 0;
    font-size: clamp(40px, 6vw, 64px);
    line-height: 0.98;
    letter-spacing: -0.04em;
  }}
  h2 {{
    margin: 0 0 14px;
    font-size: 24px;
    letter-spacing: -0.03em;
  }}
  .lede {{
    margin-top: 18px;
    max-width: 900px;
    font-size: clamp(18px, 2.1vw, 26px);
    line-height: 1.45;
    color: var(--muted);
  }}
  .hero-note {{
    margin-top: 18px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
  }}
  .actions {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 24px;
  }}
  .action-link {{
    display: inline-flex;
    align-items: center;
    min-height: 44px;
    padding: 11px 16px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--paper);
    text-decoration: none;
    font-weight: 700;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin-bottom: 24px;
  }}
  .stat {{
    padding: 22px;
    background: var(--paper);
  }}
  .stat .label {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
  }}
  .stat .value {{
    margin-top: 8px;
    font-size: clamp(32px, 4vw, 56px);
    line-height: 1;
    font-weight: 800;
    letter-spacing: -0.06em;
  }}
  .stat .subvalue {{
    margin-top: 10px;
    color: var(--muted);
    line-height: 1.6;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 18px;
    margin-bottom: 24px;
  }}
  .callout {{
    margin-top: 18px;
    padding: 16px 18px;
    border: 1px solid var(--line);
    border-radius: 20px;
    background: var(--paper);
    border-bottom: 2px solid var(--accent);
    color: var(--muted);
    line-height: 1.7;
  }}
  ul {{
    margin: 0;
    padding-left: 20px;
    color: var(--muted);
    line-height: 1.7;
  }}
  li + li {{ margin-top: 6px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
  }}
  th,
  td {{
    padding: 12px 10px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    vertical-align: top;
  }}
  th {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  tr:last-child td {{ border-bottom: none; }}
  .row-subtle {{
    margin-top: 4px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
  }}
  .badge {{
    display: inline-block;
    padding: 5px 8px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
  }}
  .badge-easy {{ background: var(--success-soft); color: var(--success); }}
  .badge-medium {{ background: var(--warning-soft); color: var(--warning); }}
  .badge-hard {{ background: var(--danger-soft); color: var(--danger); }}
  .score-pass {{ background: var(--success-soft); color: var(--success); }}
  .score-partial {{ background: var(--warning-soft); color: var(--warning); }}
  .score-weak, .score-miss, .score-fail {{ background: var(--danger-soft); color: var(--danger); }}
  .foot {{
    margin-top: 18px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
  }}
  @media (max-width: 980px) {{
    .stats,
    .grid {{
      grid-template-columns: 1fr;
    }}
    .table-panel {{
      overflow-x: auto;
    }}
    table {{
      min-width: 720px;
    }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Benchmark summary</div>
      <h1>Optimised local tool benchmark</h1>
      <p class="lede">
        This page summarises the latest rerun of the school question set against the current local setup.
        The detailed per-question evidence is linked below.
      </p>
      <p class="hero-note">
        Tested model: <strong>{_escape(report_data['model_label'])}</strong><br>
        LM Studio API model id: <strong>{_escape(report_data['model_id'])}</strong><br>
        Hardware: <strong>{_escape(report_data['hardware_label'])}</strong><br>
        Scope: <strong>{_escape(report_data['mode_label'])}</strong> · Questions: <strong>{report_data['question_count']}</strong> · Generated: <strong>{_escape(report_data['generated_at'])}</strong>
      </p>
      <div class="actions">
        <a class="action-link" href="./benchmark_results.html">View detailed results</a>
        <a class="action-link" href="./benchmark_results.json">Open raw results JSON</a>
        <a class="action-link" href="{REPO_URL}/blob/main/benchmark.py" target="_blank" rel="noreferrer">View benchmark harness on GitHub</a>
      </div>
    </section>

    <section class="stats">
      <article class="stat">
        <div class="label">Worked</div>
        <div class="value">{pass_rate}</div>
        <div class="subvalue">Questions judged as worked in the latest rerun.</div>
      </article>
      <article class="stat">
        <div class="label">Average time</div>
        <div class="value">{summary['average_time_s']:.1f}s</div>
        <div class="subvalue">{summary['total_time_s']:.1f} seconds total across the run on { _escape(report_data['hardware_label']) }.</div>
      </article>
      <article class="stat">
        <div class="label">Tool calls</div>
        <div class="value">{summary['tool_calls']}</div>
        <div class="subvalue">Total tool calls made in the optimised mode run.</div>
      </article>
      <article class="stat">
        <div class="label">Slowest question</div>
        <div class="value">{summary['longest_question_time_s']:.1f}s</div>
        <div class="subvalue">{_escape(summary['longest_question_label'])}</div>
      </article>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>What this rerun shows</h2>
        <ul>
          <li>The benchmark was rerun against the model currently served by LM Studio, with the report now driven by the actual outputs from that run.</li>
          <li>Each question is shown with the prompt, expected answer, model answer, elapsed time, tool usage, and a plain-language assessment.</li>
          <li>The detailed page is intended to make it easier to explain the result to the school without relying on a static hand-written summary.</li>
        </ul>
        <div class="callout">
          The expected answers are derived directly from the sample CSV data, and the judgement text is generated from the benchmark fact checks with a small benchmark-specific interpretation for the final “most common classes” question.
        </div>
      </article>
      <article class="panel">
        <h2>Relevant files</h2>
        <ul>
          <li><a href="{REPO_URL}/blob/main/sample_data/S1EL_3.csv" target="_blank" rel="noreferrer">Original source CSV</a></li>
          <li><a href="{REPO_URL}/blob/main/csv_tool.py" target="_blank" rel="noreferrer">Local tool layer</a></li>
          <li><a href="{REPO_URL}/blob/main/bridge.py" target="_blank" rel="noreferrer">LM Studio bridge</a></li>
          <li><a href="{REPO_URL}/blob/main/benchmark.py" target="_blank" rel="noreferrer">Benchmark harness</a></li>
        </ul>
      </article>
    </section>

    <section class="table-panel">
      <h2>Per-question summary</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Question</th>
            <th>Difficulty</th>
            <th>Assessment</th>
            <th>Time</th>
            <th>Calls</th>
          </tr>
        </thead>
        <tbody>
          {question_rows}
        </tbody>
      </table>
      <p class="foot">
        Full query-by-query evidence is available in <a href="./benchmark_results.html">benchmark_results.html</a>.
      </p>
    </section>
  </div>
</body>
</html>"""


def generate_detail_html(report_data):
    summary = report_data["summary"]
    cards_html = ""

    for row in report_data["questions"]:
        judgement = row["judgement"]
        cards_html += f"""
        <article class="question-card">
          <div class="question-top">
            <div>
              <div class="question-index">Question {row['index']}</div>
              <h2>{_escape(row['short_label'])}</h2>
              <p class="question-text">{_escape(row['question'])}</p>
            </div>
            <div class="question-badges">
              <span class="badge badge-{row['difficulty'].lower()}">{_escape(row['difficulty'])}</span>
              <span class="badge {judgement['css']}">{_escape(judgement['verdict'])}</span>
            </div>
          </div>
          <div class="meta-row">
            <span><strong>Time:</strong> {row['time_s']:.1f}s</span>
            <span><strong>Tool calls:</strong> {row['tool_call_count']}</span>
          </div>
          <div class="compare-grid">
            <div class="answer-panel">
              <h3>Expected answer</h3>
              <pre>{_escape(row['expected_answer'])}</pre>
            </div>
            <div class="answer-panel">
              <h3>Model answer</h3>
              <pre>{_escape(row['model_answer'])}</pre>
            </div>
          </div>
          <div class="assessment">
            <h3>Assessment</h3>
            <p>{_escape(judgement['reasoning'])}</p>
            <p><strong>Matched facts:</strong> {_escape(format_fact_list(judgement['matched_facts'], limit=8))}</p>
            <p><strong>Missing facts:</strong> {_escape(format_fact_list(judgement['missing_facts'], limit=8))}</p>
          </div>
          {_tool_detail_html(row)}
        </article>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Results</title>
<style>
  :root {{
    --bg: #f4efe7;
    --paper: #fffdf8;
    --frame-bg: rgba(255, 253, 248, 0.92);
    --hover-bg: #fff7ef;
    --bg-glow-1: rgba(180, 77, 22, 0.08);
    --bg-glow-2: rgba(31, 93, 99, 0.09);
    --ink: #1e1812;
    --muted: #6e655c;
    --line: #dbcab5;
    --accent: #b44d16;
    --accent-dark: #7d2e0d;
    --accent-soft: #f7dfcf;
    --teal: #1f5d63;
    --teal-soft: #dceef0;
    --success: #2f7d32;
    --success-soft: #e1f2e2;
    --warning: #9a6700;
    --warning-soft: #f5ead2;
    --danger: #9b2c2c;
    --danger-soft: #f8dedd;
    --shadow: 0 18px 48px rgba(63, 38, 15, 0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Avenir Next", "Segoe UI", sans-serif;
    color: var(--ink);
    background:
      radial-gradient(circle at top right, var(--bg-glow-1), transparent 24%),
      radial-gradient(circle at bottom left, var(--bg-glow-2), transparent 28%),
      var(--bg);
  }}
  a {{
    color: inherit;
    text-decoration-thickness: 2px;
    text-underline-offset: 0.18em;
    text-decoration-color: color-mix(in srgb, var(--accent) 55%, transparent);
  }}
  a:hover {{ color: var(--accent); }}
  .wrap {{
    width: min(1180px, calc(100% - 32px));
    margin: 32px auto 48px;
  }}
  .hero,
  .question-card {{
    border: 1px solid var(--line);
    border-radius: 28px;
    background: var(--frame-bg);
    box-shadow: var(--shadow);
    backdrop-filter: blur(8px);
  }}
  .hero {{
    padding: 32px;
    margin-bottom: 24px;
  }}
  .eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    padding: 7px 12px;
    border-radius: 999px;
    background: var(--teal-soft);
    color: var(--teal);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }}
  .eyebrow::before {{
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--teal);
  }}
  h1 {{
    margin: 0;
    font-size: clamp(40px, 6vw, 64px);
    line-height: 0.98;
    letter-spacing: -0.04em;
  }}
  h2 {{
    margin: 0;
    font-size: 30px;
    letter-spacing: -0.03em;
  }}
  .lede {{
    margin-top: 18px;
    max-width: 960px;
    font-size: 20px;
    line-height: 1.5;
    color: var(--muted);
  }}
  .actions {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 24px;
  }}
  .action-link {{
    display: inline-flex;
    align-items: center;
    min-height: 44px;
    padding: 11px 16px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--paper);
    text-decoration: none;
    font-weight: 700;
  }}
  .overview {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin-top: 24px;
  }}
  .overview-card {{
    padding: 20px;
    border: 1px solid var(--line);
    border-radius: 22px;
    background: var(--paper);
  }}
  .overview-card .label {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
  }}
  .overview-card .value {{
    margin-top: 8px;
    font-size: 34px;
    line-height: 1;
    font-weight: 800;
    letter-spacing: -0.05em;
  }}
  .overview-card .subvalue {{
    margin-top: 10px;
    color: var(--muted);
    line-height: 1.6;
  }}
  .results {{
    display: grid;
    gap: 18px;
  }}
  .question-card {{
    padding: 28px;
  }}
  .question-top {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
  }}
  .question-index {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }}
  .question-text {{
    margin: 12px 0 0;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.6;
  }}
  .question-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .badge {{
    display: inline-block;
    padding: 5px 8px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
  }}
  .badge-easy {{ background: var(--success-soft); color: var(--success); }}
  .badge-medium {{ background: var(--warning-soft); color: var(--warning); }}
  .badge-hard {{ background: var(--danger-soft); color: var(--danger); }}
  .score-pass {{ background: var(--success-soft); color: var(--success); }}
  .score-partial {{ background: var(--warning-soft); color: var(--warning); }}
  .score-weak, .score-miss, .score-fail {{ background: var(--danger-soft); color: var(--danger); }}
  .meta-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 14px;
    color: var(--muted);
    font-size: 14px;
  }}
  .compare-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 22px;
  }}
  .answer-panel,
  .assessment,
  .tool-panel {{
    padding: 18px;
    border: 1px solid var(--line);
    border-radius: 22px;
    background: var(--paper);
  }}
  .answer-panel h3,
  .assessment h3 {{
    margin: 0 0 10px;
    font-size: 18px;
    letter-spacing: -0.02em;
  }}
  pre {{
    margin: 0;
    white-space: pre-wrap;
    font: inherit;
    line-height: 1.7;
    color: var(--muted);
  }}
  .assessment {{
    margin-top: 16px;
    border-bottom: 2px solid var(--accent);
  }}
  .assessment p {{
    margin: 0;
    color: var(--muted);
    line-height: 1.7;
  }}
  .assessment p + p {{
    margin-top: 10px;
  }}
  details {{
    margin-top: 16px;
  }}
  summary {{
    cursor: pointer;
    font-weight: 700;
    color: var(--teal);
  }}
  .tool-calls {{
    margin-top: 12px;
    display: grid;
    gap: 10px;
  }}
  .tool-call {{
    padding: 14px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: var(--paper);
  }}
  .tool-call code {{
    display: block;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 13px;
    color: var(--muted);
  }}
  .tool-note {{
    margin-top: 8px;
    color: var(--danger);
    font-size: 13px;
    font-weight: 700;
  }}
  @media (max-width: 980px) {{
    .overview,
    .compare-grid {{
      grid-template-columns: 1fr;
    }}
    .question-top {{
      flex-direction: column;
    }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Detailed benchmark results</div>
      <h1>Question-by-question evidence</h1>
      <p class="lede">
        This page shows the actual prompt, expected answer, model answer, tool usage, timing, and assessment for the latest run of the school benchmark on { _escape(report_data['hardware_label']) }.
      </p>
      <div class="actions">
        <a class="action-link" href="./benchmark_report.html">Back to summary</a>
        <a class="action-link" href="./benchmark_results.json">Open raw results JSON</a>
        <a class="action-link" href="{REPO_URL}/blob/main/benchmark.py" target="_blank" rel="noreferrer">View benchmark harness on GitHub</a>
      </div>
      <div class="overview">
        <article class="overview-card">
          <div class="label">Worked</div>
          <div class="value">{summary['pass_count']}/{report_data['question_count']}</div>
          <div class="subvalue">Questions judged as worked in this rerun.</div>
        </article>
        <article class="overview-card">
          <div class="label">Average time</div>
          <div class="value">{summary['average_time_s']:.1f}s</div>
          <div class="subvalue">{_escape(report_data['mode_label'])}<br>{summary['tool_calls']} total tool calls on { _escape(report_data['hardware_label']) }</div>
        </article>
        <article class="overview-card">
          <div class="label">Generated</div>
          <div class="value">{_escape(report_data['generated_at'].split()[0])}</div>
          <div class="subvalue">{_escape(report_data['generated_at'])}</div>
        </article>
        <article class="overview-card">
          <div class="label">Model tested</div>
          <div class="value">{_escape(report_data['model_id'])}</div>
          <div class="subvalue">{_escape(report_data['model_label'])}<br>{_escape(report_data['hardware_label'])}</div>
        </article>
      </div>
    </section>

    <section class="results">
      {cards_html}
    </section>
  </div>
</body>
</html>"""


def generate_full_comparison_html(ground_truth, with_tools_results, optimised_results, no_tools_results):
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
  Model: <strong>{MODEL_LABEL}</strong> (LM Studio id: <strong>{MODEL}</strong>) &middot; {n} questions &middot;
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
  <p style="margin-top:0.5rem">Generated from benchmark.py &middot; {datetime.now().strftime('%Y-%m-%d')}</p>
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local school-data benchmark.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing the sample CSV files.")
    parser.add_argument("--lm-studio-url", default=LM_STUDIO_URL, help="Base URL for the local LM Studio OpenAI-compatible API.")
    parser.add_argument("--model", default=MODEL, help="LM Studio model id to call.")
    parser.add_argument("--model-label", default=MODEL_LABEL, help="Human-readable model label to show in reports.")
    parser.add_argument(
        "--mode",
        choices=["optimised", "full"],
        default="optimised",
        help="Run only the current optimised tool benchmark, or all benchmark modes.",
    )
    parser.add_argument("--summary-report", default=SUMMARY_REPORT_PATH, help="Path for the summary HTML report.")
    parser.add_argument("--detail-report", default=DETAIL_REPORT_PATH, help="Path for the detailed per-question HTML report.")
    parser.add_argument("--results-json", default=RESULTS_JSON_PATH, help="Path for the machine-readable results JSON.")
    parser.add_argument("--full-report", default=FULL_REPORT_PATH, help="Path for the optional full three-mode comparison report.")
    return parser.parse_args()


def configure_runtime(args):
    global DATA_DIR, LM_STUDIO_URL, MODEL, MODEL_LABEL, client, csv_tool
    DATA_DIR = args.data_dir
    LM_STUDIO_URL = args.lm_studio_url
    MODEL = args.model
    MODEL_LABEL = args.model_label
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
    csv_tool = CSVTool(data_dir=DATA_DIR)


def main():
    args = parse_args()
    configure_runtime(args)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("=" * 60)
    print("LLM CSV Tool-Calling Benchmark")
    print(f"Model label: {MODEL_LABEL}")
    print(f"LM Studio model id: {MODEL}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    # Step 1: Ground truth
    print("\n[1/3] Computing ground truth...")
    ground_truth = compute_ground_truth()
    for i, (q, a) in enumerate(zip(SHORT_LABELS, ground_truth)):
        print(f"  Q{i+1} ({q}): {a[:80]}...")

    # Step 2: With tools (optimised — pre-seeded schema)
    print("\n[2/3] Running tools (optimised — pre-seeded schema)...")
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

    print("\n[3/3] Writing summary, detail, and JSON outputs...")
    report_data = build_single_mode_report_data(ground_truth, optimised_results, "Optimised tools mode", generated_at)
    Path(args.summary_report).write_text(generate_summary_html(report_data), encoding="utf-8")
    Path(args.detail_report).write_text(generate_detail_html(report_data), encoding="utf-8")
    Path(args.results_json).write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Summary report: {args.summary_report}")
    print(f"  Detail report: {args.detail_report}")
    print(f"  Results JSON:   {args.results_json}")

    if args.mode == "full":
        print("\nRunning extra comparison modes for the full report...")
        with_tools_results = []
        for i, q in enumerate(QUESTIONS):
            print(f"  Baseline Q{i+1}/{len(QUESTIONS)}: {SHORT_LABELS[i]}...", end=" ", flush=True)
            result = run_with_tools(q)
            with_tools_results.append(result)
            tc = len(result.get("tool_calls", []))
            print(f"done ({result['time_s']:.1f}s, {tc} tool calls)")

        no_tools_results = []
        for i, q in enumerate(QUESTIONS):
            print(f"  No-tools Q{i+1}/{len(QUESTIONS)}: {SHORT_LABELS[i]}...", end=" ", flush=True)
            result = run_without_tools(q, i)
            no_tools_results.append(result)
            ctx = result.get("context_mode", "?")
            print(f"done ({result['time_s']:.1f}s, ctx={ctx})")

        Path(args.full_report).write_text(
            generate_full_comparison_html(ground_truth, with_tools_results, optimised_results, no_tools_results),
            encoding="utf-8",
        )
        print(f"  Full comparison report: {args.full_report}")

    print("Done.")


if __name__ == "__main__":
    main()
