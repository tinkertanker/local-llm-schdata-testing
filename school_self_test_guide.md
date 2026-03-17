# School Self-Test Guide

This guide is for the school if they want to try the prototype in their own environment, using their own LM Studio instance.

There are two sensible routes.

## Route A: easiest path

Use the existing local chat interface exactly as provided.

### What they need

1. LM Studio installed locally
2. A local model loaded in LM Studio
3. The LM Studio local server enabled
4. This project checked out locally
5. A safe dataset:

- either the included synthetic `sample_data/`
- or an approved/redacted school CSV export

### Recommended setup

1. Start LM Studio and load the agreed local model.
2. In LM Studio, enable the local OpenAI-compatible server at:

```text
http://localhost:1234/v1
```

3. Run the local browser chat:

```bash
python3 bridge.py --data-dir ./sample_data --serve-chat --port 8000 --model qwen3.5-35b-a3b
```

4. Open:

```text
http://127.0.0.1:8000
```

5. Ask the same benchmark questions the school originally supplied.

### If they want to try their own CSV export

Put the approved CSV files into a folder and point the bridge at that folder instead:

```bash
python3 bridge.py --data-dir ./school_data --serve-chat --port 8000 --model qwen3.5-35b-a3b
```

## Route B: attach the tools into their own app

This is for a more technical team that already has an app which can call an OpenAI-compatible model and supports tool/function calling.

### Option B1: run the tools as a local HTTP service

Start the CSV tool service:

```bash
python3 csv_tool.py --serve --port 5555 --data-dir ./school_data
```

This exposes a local endpoint that accepts requests like:

```bash
curl -X POST http://127.0.0.1:5555 \
  -H "Content-Type: application/json" \
  -d '{"tool": "summarise", "params": {"file": "students.csv", "group_by": "ACADEMIC CLASS", "metric": "count"}}'
```

Their app can then:

1. send the user’s prompt to LM Studio
2. allow the model to choose from the tool definitions
3. execute the chosen tool locally against the CSV tool service
4. pass the tool result back to the model

### Option B2: embed the tool layer directly in Python

If their application is Python-based, it can import the tool layer directly:

```python
from csv_tool import CSVTool, TOOL_DEFINITIONS

tool = CSVTool(data_dir="./school_data")
result = tool.execute("get_student_teachers", {"student": "Qian Hui Zheng"})
```

This avoids the extra HTTP hop and uses the same tool logic as the browser chat and benchmark.

## Where the tool definitions live

The OpenAI-style tool schema is already defined in:

- `csv_tool.py` via `TOOL_DEFINITIONS`

That means a team building its own interface does not need to redesign the function schema from scratch.

## Which questions are currently the best fit

The strongest supported starting point is the original school question set:

- class counts
- students per class
- student subjects
- subject rigour
- class teacher
- student teachers
- level teachers
- current location
- student summary
- EL history
- attendance
- same subjects
- same EL class
- most common classes

## Suggested test sequence for the school

If the school wants to trial this independently, ask them to test in this order:

1. One simple count question
2. One student profile question
3. One teacher-join question
4. One timetable/location question
5. One relationship question

This gives a quick check across the full range of behaviour without turning the first trial into an open-ended exploration session.

## What to say about readiness

Recommended wording:

> You can test this locally in your own LM Studio environment today. The safest way to begin is to run the existing bridge against a safe extract and validate the original question set in your own setting before integrating it into a wider school app.

## Important caveats

- The benchmark evidence is currently based on the optimised tool setup.
- The strongest evidence is on the exact question set already supplied by the school.
- A redacted or approved school export should still be used before any real pilot claim is made.
- If the school changes the schema significantly, the tool descriptions or joined tools may need adjustment.
