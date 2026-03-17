# CSV Tool for Local LLMs (via LM Studio)

Tools that let teachers ask questions about student data in natural language, with all data staying on the local machine. Nothing goes to the cloud.

## How it works

```
Teacher types a question
        |
Local LLM (LM Studio) interprets the question
        |
LLM calls a CSV tool (read_headers, query, summarise, etc.)
        |
Tool runs against the CSV file locally
        |
Results go back to the LLM
        |
LLM writes a plain English answer
```

## Setup

### 1. Install LM Studio

Download from https://lmstudio.ai.

Recommended models for tool calling:

- **Qwen 2.5 7B Instruct** (~5 GB) — best tool-calling accuracy
- **Phi-3 Medium Instruct** (~8 GB) — good balance
- **Phi-3 Mini Instruct** (~2.5 GB) — for lower-spec machines

After loading a model, go to the Developer tab and start the local server (runs on http://localhost:1234).

### 2. Install Python dependency

```bash
pip install openai
```

That's it. The CSV tool uses Python's built-in libraries only.

### 3. Put your CSV files in a folder

Create a folder and drop your CSV files in:

```
school_data/
  S1EL_2.csv
  math_results.csv
```

### 4. Run

```bash
python bridge.py --data-dir ./school_data
```

Then type questions at the prompt:

```
Teacher: How many students does each teacher have?
Teacher: Show me the breakdown of EL levels across classes
Teacher: Which classes does Mdm Ang teach?
```

### 5. Optional: browser chat interface

If you want a simple local chat UI instead of the terminal prompt:

```bash
python bridge.py --data-dir ./school_data --serve-chat --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.

The web UI keeps the conversation local, reuses the same tool-calling bridge, and supports resetting the current chat session.

## Available tools

The LLM can call these functions:

| Tool | What it does | Example question |
|------|-------------|------------------|
| get_student_overview | One-call student profile join across roster, subjects, attendance, EL history, and classmates | "Tell me about Qian Hui Zheng" |
| get_student_teachers | Automatically joins subject enrolments to teacher assignments | "Who teaches Qian Hui Zheng?" |
| get_student_location | Joins student → class → timetable for a specific slot | "Where is Qian Hui Zheng now?" |
| find_students_same_subjects | Finds students with the same subject set as a target student | "Who takes the same subjects as Qian Hui Zheng?" |
| get_level_teachers | Returns form teachers for a level such as S1 | "Who teaches S1 level?" |
| list_files | Shows available CSV files | "What data do we have?" |
| read_headers | Shows columns and sample values | (called automatically when needed) |
| query | Filter and select rows, including multi-value `in` filters | "Show me G1 students in Honour 1" |
| summarise | Group and count/sum/avg | "How many students per teacher?" |
| distinct | Unique values in a column | "What EL levels exist?" |
| cross_tabulate | Pivot table of two columns | "Teacher vs EL level breakdown" |

## Benchmark

`benchmark.py` tests how well a local LLM answers 14 school-data questions across three modes:

| Mode | Description |
|------|-------------|
| **Tools (baseline)** | LLM uses function calling; must discover files and headers each time |
| **Tools (optimised)** | Same tools, but file list and column schemas are pre-seeded in the system prompt |
| **No tools** | Raw CSV data stuffed into the system prompt; no tool access |

Latest verified result on **16 March 2026** with `qwen3.5-35b-a3b`:

- **Tools (optimised): 14/14 PASS**
- Achieved by combining pre-seeded schema, one-shot joined tools, one-shot answer forcing, and a stricter fact scorer that handles compound facts correctly

### Running the benchmark

```bash
# Edit MODEL at the top of benchmark.py to match your LM Studio model ID
python benchmark.py
# Opens benchmark_report.html when done
```

### Sample data

The `sample_data/` directory contains 7 interconnected CSV files (32 students, 8 classes, 6 teachers) covering:

| File | Purpose |
|------|---------|
| `students.csv` | Student roster (name, class, EL level) |
| `class_info.csv` | Class → form teacher mapping |
| `subject_enrolment.csv` | Who takes what subject at what rigour |
| `subject_teachers.csv` | Subject + rigour → teacher mapping |
| `timetable.csv` | Day/period/class/subject/room schedule |
| `el_history.csv` | 3 years of EL grade progression |
| `attendance.csv` | 3 terms of attendance records |

### 14 benchmark questions

Questions range from simple lookups ("How many classes are there?") to multi-file joins ("Who teaches Qian Hui Zheng?" requires subject_enrolment → subject_teachers). Each is tagged Easy/Medium/Hard.

### Benchmark features

- **Fact-extraction scoring** — ground truth is split into atomic facts (names, numbers, key-value pairs) and each is checked independently in the model's answer, regardless of formatting
- **Loop detection** — duplicate tool calls are warned on 2nd occurrence, blocked on 3rd; hard cap of 10 tool calls per question; model is forced to answer with data gathered so far
- **Context fallback** — "no tools" mode tries all CSVs first, falls back to per-question relevant files if context overflows
- **HTML report** — generates a styled dark-theme report with score badges, progress bars, and expandable tool call logs

See [FINDINGS.md](FINDINGS.md) for the earlier benchmark history, the joined-tool redesign, and the latest verified pass result.

## School-facing materials

If you need to explain the project to non-technical stakeholders:

- [docs/school_presentation.html](docs/school_presentation.html) — hostable static presentation deck
- [docs/school_demo_recording_guide.md](docs/school_demo_recording_guide.md) — suggested recording flow and voiceover

## Testing the CSV tool directly

You can test the tool without LM Studio:

```bash
# List files
python csv_tool.py --data-dir ./school_data --tool list_files

# Read headers
python csv_tool.py --data-dir ./school_data --tool read_headers \
  --params '{"file": "S1EL_2.csv"}'

# Cross-tabulate
python csv_tool.py --data-dir ./school_data --tool cross_tabulate \
  --params '{"file": "S1EL_2.csv", "row_col": "TEACHER", "col_col": "EL"}'
```

## Deterministic regression tests

The joined tools and scorer also have a deterministic regression suite:

```bash
python3 -m unittest test_csv_tool.py
```

## Presentation assets

For stakeholder-facing materials, the repo now includes:

- [docs/school-presentation.html](docs/school-presentation.html) — a hostable slide deck for the school briefing
- [docs/demo-recording.md](docs/demo-recording.md) — a practical runbook for recording a short demo video

## School-facing assets

Two project artefacts are included for explaining the work to non-technical stakeholders:

- `school_briefing.html` — a hostable static presentation covering the starting point, synthetic data build-out, tooling, benchmark approach, current result, and pilot framing
- `demo_recording_guide.md` — a suggested script for recording a short walkthrough if you do not want to host a live demo
- `school_self_test_guide.md` — instructions for the school if they want to try the prototype in their own LM Studio instance or wire the tools into their own app

## Running as an HTTP server

If you want other applications to call the tool over HTTP:

```bash
python csv_tool.py --serve --port 5555 --data-dir ./school_data
```

Then POST to it:

```bash
curl -X POST http://localhost:5555 \
  -H "Content-Type: application/json" \
  -d '{"tool": "summarise", "params": {"file": "S1EL_2.csv", "group_by": "TEACHER", "metric": "count"}}'
```

## Security notes

- The tool restricts file access to the specified data directory only
- No directory traversal is possible (filenames are sanitised)
- The LLM never sees raw data unless the tool returns it
- No data leaves the machine

## Adapting for other data

The tool works with any CSV. Teachers can swap in different files (exam results, attendance, etc.) and the LLM will discover the schema automatically via read_headers. No configuration needed when the data changes.
