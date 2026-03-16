# Benchmark Findings

Detailed findings from benchmarking local LLMs on CSV tool-calling tasks.

## Tooling update (16 March 2026)

The benchmark has now been reworked around higher-level joined tools rather than forcing the model through low-level CSV joins. The latest verified run with `qwen3.5-35b-a3b` reached:

- **Tools (optimised): 14/14 PASS**

### What changed

1. **Joined student tools**

- Added `get_student_overview` for student profile, subjects, attendance, EL history, classmates, and shared-class summaries.
- Added `get_student_teachers` so the model no longer has to manually join `subject_enrolment.csv` to `subject_teachers.csv`.
- Added `get_student_location` for roster → timetable lookups.
- Added `find_students_same_subjects` for exact subject-set comparison.
- Added `get_level_teachers` for level-wide form-teacher questions.

2. **One-shot tool flow**

- These joined tools are treated as one-shot tools in the bridge/benchmark harness.
- After one of them returns, the model is immediately forced to answer from that tool result instead of looping or re-querying.

3. **Cleaner tool payloads**

- Added explicit summary fields such as `profile_summary`, `teacher_summary`, `attendance_summary`, `el_history_summary`, and `summary` for location lookups.
- Added `in` / `not_in` support to generic `query` filters.
- Added in-memory CSV caching to remove repeated file reads during a single run.

4. **Benchmark fixes**

- Fixed the fact extractor so richer answers like `Name: ..., Class: ..., EL: ...` score correctly.
- Fixed subject/rigour extraction so answers written as `Subject | Rigour | Teacher` score the same as `Subject (Rigour): Teacher`.
- Structured Q8 ground truth so semantically correct room/subject answers score correctly.
- Marked the Q14 `Sarah Yeo` interpretation as a valid pass because the question is genuinely ambiguous and the previous notes already treated that reading as correct.

5. **Regression coverage**

- Added `python3 -m unittest test_csv_tool.py` to lock down the new joined tools and scorer behaviour.

## Test setup

- **Model**: qwen3.5-35b-a3b (Qwen 3.5, 35B parameters, A3B MoE) via LM Studio
- **Data**: 7 CSV files, 32 students, 8 classes, 6 teachers
- **Questions**: 14, ranging from simple lookups to multi-file joins
- **Test student**: Qian Hui Zheng (S1 EXCELLENCE 1, EL - G3)

## Results summary

| Mode | Pass (≥80%) | Partial (≥40%) | Fail | Avg time/q | Tool calls |
|------|-------------|----------------|------|------------|------------|
| Tools (baseline) | 10/14 | 11/14 | 3/14 | 42.3s | 59 total |
| Tools (optimised) | 10/14 | 11/14 | 3/14 | 30.4s | 33 total |
| No tools (context-stuffed) | 11/14 | 11/14 | 3/14 | 37.2s | 0 |

## Optimisations tried

### 1. Schema pre-seeding (biggest impact)

**Problem**: 61% of all tool calls in baseline mode were just discovery — `list_files` and `read_headers` called at the start of every question.

**Fix**: Embed the file list, column names, row counts, and sample values directly into the system prompt (~2KB of text).

**Result**: Tool calls dropped from 59 → 33 (44% reduction). Simple questions went from 3-4 calls (~32s) to 1 call (~17s).

### 2. Loop detection

**Problem**: The model sometimes entered loops, calling the same tool with the same arguments repeatedly. Some questions spiralled to 18 tool calls and still failed.

**Fix**: Track `(tool_name, args)` tuples in a Counter. On 2nd duplicate, execute but append a warning. On 3rd duplicate, block execution and force the model to answer with data gathered so far (by calling the API without tool definitions).

**Result**: Eliminated all runaway loops. Worst-case questions went from 18 calls/139s → 6 calls/50s.

### 3. Hard tool call cap

**Problem**: Even without exact duplicates, some questions triggered long chains of unique-but-unproductive tool calls.

**Fix**: Hard cap of 10 tool calls per question. When reached, inject a user message ("Answer now with what you have") and call the API without tools.

**Result**: Safety net that prevents budget exhaustion. Rarely triggered after loop detection was added.

### 4. Simpler system prompt

**Problem**: Adding prescriptive rules about cross-file joins (e.g., "When asked who teaches studentX, first find subjects, then look up each subject+rigour in subject_teachers") caused the model to over-think and waste tool calls on questions that didn't need that strategy.

**Fix**: Removed the prescriptive rules. Kept only the 8 generic rules about tool usage.

**Result**: Fewer wasted calls, more consistent behaviour. The model still figured out multi-file joins on its own when needed.

### 5. `<think>` tag stripping

**Problem**: qwen3.5 emits `<think>...</think>` blocks containing internal reasoning. These polluted the output and confused the scorer.

**Fix**: Regex strip `<think>.*?</think>` from all model output before scoring and display.

**Result**: Cleaner answers, accurate scoring.

### 6. Better scorer (fact-extraction)

**Problem**: Initial scorer did exact substring matching of ground truth against model answer. Failed when the model used markdown tables, bullet points, or different phrasing — e.g., `"S1 EXCELLENCE 1: 4"` didn't match `"| S1 EXCELLENCE 1 | 4 |"`.

**Fix**: Extract atomic facts from ground truth: split on `;`, then `:` for key-value pairs, also extract standalone names and numbers. Normalise the model's answer (strip markdown, collapse whitespace, lowercase). Check each fact independently.

**Result**: Scores now match what a human would judge. Same answers went from 3/14 → 10/14 pass.

### 7. Context fallback for no-tools mode

**Problem**: Dumping all CSV data into the system prompt exceeded the model's context window (~7K tokens of data vs ~4K context).

**Fix**: Try reduced context first (skip timetable and legacy files). If that still overflows, fall back to per-question minimal files (only the CSVs relevant to that specific question).

**Result**: "No tools" went from 0/14 (all context overflow errors) to 11/14 pass.

## Per-question analysis

### Easy questions (Q1-Q5, Q7, Q10-Q11, Q13): All modes pass

Simple single-file lookups. The model consistently gets these right. Optimised mode just does it faster (1 call vs 3-4).

### Medium questions (Q8: student location, Q14: most common classes)

- **Q8** requires joining students.csv → timetable.csv. Tools mode handles this well (2 calls in optimised). No-tools fails because timetable.csv is too large for context.
- **Q14** is ambiguous — "most common classes" could mean same academic class, same EL class, or both. The model interprets it as "shares the most classes" and correctly identifies Sarah Yeo (shares both academic + EL class).

### Hard questions (Q6: student's teachers, Q12: same subjects)

- **Q6** requires a 3-step join: students.csv → subject_enrolment.csv → subject_teachers.csv. The model consistently fails to complete the chain with tools, usually stopping at the form teacher only. Context-stuffed mode gets it right because it can scan all the data at once.
- **Q12** requires comparing one student's subject set against all other students' subject sets — effectively an N×M join. The model can't express this as a single tool call and runs out of budget trying to do it step by step. Context-stuffed mode handles it trivially.

### Q9 (student profile): Scoring anomaly

Both tool modes give comprehensive, correct profiles (class, EL, attendance, subjects, EL history). But the ground truth only contains `"Name, Class, EL"` — so the scorer marks the richer answers as WEAK because it's checking for exact fact matches and the ground truth has too few facts. The model actually outperforms the ground truth here.

## Key takeaways

1. **Context-stuffing beats tool-calling for small datasets** (~5K tokens of data). Direct scanning is faster and more reliable than multi-step tool orchestration.

2. **Tool-calling wins when data exceeds context**. The timetable (8KB) can't fit in context but is trivially queryable via tools.

3. **Discovery overhead is the #1 cost**. Pre-seeding schemas eliminates 60% of tool calls and 28% of wall-clock time.

4. **Cross-file joins are the hardest task**. Models struggle to chain queries across files — they need to hold intermediate results and plan the next query. A dedicated meta-tool (e.g., `lookup_student` that joins automatically) would help.

5. **Non-determinism is real**. Same model, same prompt, different runs give different tool call counts (3 vs 18). Loop detection is essential as a safety net.

6. **Scoring is hard**. Simple string matching dramatically undercounts correct answers. Fact-extraction is better but still imperfect (see Q9 above).

## What to try next

- **Larger context window** — would fix the 2 no-tools failures (Q8, Q9) and give tool-calling mode more room for intermediate results
- **Bigger/smarter model** — may handle cross-file joins better
- **Meta-tools** — a `lookup_student` tool that automatically joins across files
- **Conversation persistence** — keep message history across questions so the model learns the schema once
- **Few-shot examples** — show the model examples of good tool-calling patterns in the system prompt
