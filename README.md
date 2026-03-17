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

`benchmark.py` tests how well a local LLM answers 14 school-data questions. The current reporting flow is focused on the optimised tool setup and writes:

- `benchmark_report.html` — summary page for the latest run
- `benchmark_results.html` — detailed per-question evidence, including prompt, expected answer, model answer, timing, and assessment
- `benchmark_results.json` — machine-readable result bundle

An optional `--mode full` rerun also writes `benchmark_full_report.html` with the older three-mode comparison.

The full comparison modes are:

| Mode | Description |
|------|-------------|
| **Tools (baseline)** | LLM uses function calling; must discover files and headers each time |
| **Tools (optimised)** | Same tools, but file list and column schemas are pre-seeded in the system prompt |
| **No tools** | Raw CSV data stuffed into the system prompt; no tool access |

Latest rerun on **17 March 2026** with `mlx-community/Qwen3.5-35B-A3B-4bit mlx` via LM Studio model id `qwen3.5-35b-a3b`:

- **Optimised tools mode: 14/14 worked**
- **Average time:** 55.7s per question
- **Total tool calls:** 15
- The generated HTML now includes a summary page plus a detailed question-by-question evidence page

### Running the benchmark

```bash
python benchmark.py \
  --mode optimised \
  --model qwen3.5-35b-a3b \
  --model-label "mlx-community/Qwen3.5-35B-A3B-4bit mlx"

# Outputs benchmark_report.html, benchmark_results.html, and benchmark_results.json
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

## School-facing assets

Two project artefacts are included for explaining the work to non-technical stakeholders:

- `index.html` — the GitHub Pages entrypoint and hostable presentation deck for the school briefing
- `school_briefing.html` — a lightweight redirect to `index.html` for backwards compatibility
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
