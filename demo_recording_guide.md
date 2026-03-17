# Demo Recording Guide

This is a short recording plan for showing the school what has been built without needing to host a public live demo.

## Goal

Show three things clearly:

1. We started from a very limited dataset and expanded it safely.
2. The local LLM is not answering blindly; it is working through constrained local tools.
3. The current setup now passes the school’s benchmark question set in the optimised configuration.

## Recommended format

- Length: 4 to 6 minutes
- Style: screen recording with your voiceover
- Tone: calm, practical, not hype-driven
- Resolution: 1440p or 1080p

## Before recording

1. Start LM Studio with the working local model loaded.
2. Start the browser chat locally:

```bash
python3 bridge.py --data-dir ./sample_data --serve-chat --port 8000
```

3. Open these tabs in advance:

- `index.html`
- `http://127.0.0.1:8000`
- `benchmark_report.html`

4. Increase browser zoom slightly if needed so the school can read it easily on a video call.
5. Close unrelated apps, notifications, and personal tabs.

## Suggested running order

### 1. Opening context

Screen:

- Open `index.html`
- Show the title slide and the “from one CSV to a local classroom data assistant” framing

Talk track:

> We started with a single file, S1EL_3.csv. From that, we created synthetic linked data and a set of local tools so we could test whether a local LLM would be suitable for the exact question types the school asked us about.

### 2. Explain the synthetic data step

Screen:

- Scroll to the slide showing how `S1EL_3.csv` was expanded into the linked synthetic dataset

Talk track:

> We did not want to make claims from one flat CSV alone, because several of the school’s questions need joins across student, subject, teacher, timetable, attendance, and EL-history data. So we built a safe synthetic model that lets us test those questions properly without using real student records.

### 3. Show the chat interface

Screen:

- Switch to the local browser chat

Talk track:

> This is a simple local interface over the same tool-calling bridge. The model runs locally in LM Studio, and the CSV queries happen locally as well.

### 4. Ask representative questions

Use a small set that covers the range of behaviour.

Recommended sequence:

1. `How many classes are there?`
2. `Who teaches Qian Hui Zheng?`
3. `Tell me about Qian Hui Zheng.`
4. `Where is Qian Hui Zheng now? Assume it's Monday Period 1.`
5. `Who takes the same subjects as Qian Hui Zheng?`

Talk track after one of the joined questions:

> Questions like this are exactly where the early version struggled, because the local model had to work out the joins itself. The newer version uses higher-level local tools for those harder cases.

### 5. Close on evidence

Screen:

- Open `benchmark_report.html`
- Point to the 14/14 result

Talk track:

> We are not asking you to trust the chat transcript alone. We benchmarked the system against the school’s question set, and the current optimised local setup passed all 14 benchmark questions in the latest verified run.

### 6. Finish with the right expectation

Screen:

- Return to the “pilot-ready with guardrails” slide

Talk track:

> Our recommendation is not to jump straight to full deployment. The sensible next step is a limited pilot on approved school data, using the same local-first approach and the same supported question boundary.

## Good phrasing to reuse

- “Local-first, with data staying on the machine.”
- “Synthetic data for safe validation, not as a substitute for a real pilot.”
- “The benchmark mirrors the school’s own question set.”
- “Pilot-ready with guardrails.”

## What not to say

- Do not imply this is production-ready for every possible question.
- Do not imply that any local model will work equally well.
- Do not imply the synthetic benchmark removes the need for a real-data validation step.

## If you want a shorter 2-minute version

1. One slide on the starting point and synthetic data.
2. Two chat questions.
3. One benchmark result slide.

## If the live model is slow during recording

Have a fallback:

- Ask the questions once before recording so the chat interface is warmed up.
- Keep the benchmark report open as proof even if a live answer takes a bit longer.
- If needed, record the chat section in shorter clips and stitch them together.
