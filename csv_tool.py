"""
CSV Tool for Local LLMs
========================
A set of simple, constrained tools that a local LLM (via LM Studio)
can call to query CSV files without needing to write code.

Designed to be used with LM Studio's OpenAI-compatible function calling API.
The LLM never sees raw student data — it sees column names, aggregated results,
and filtered views that the tool returns.

Usage:
    # As a standalone server:
    python csv_tool.py --serve --port 5555 --data-dir ./school_data

    # As a library (imported by the LM Studio bridge):
    from csv_tool import CSVTool
    tool = CSVTool(data_dir="./school_data")
    result = tool.execute("read_headers", {"file": "S1EL_2.csv"})
"""

import argparse
import csv
import json
import sys
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


class CSVTool:
    """Core CSV operations designed for local LLM function calling."""

    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self._csv_cache: dict[str, tuple[float, list[str], list[dict]]] = {}

    # ------------------------------------------------------------------ #
    #  Safety: resolve paths so the LLM can't escape the data directory
    # ------------------------------------------------------------------ #

    def _safe_path(self, filename: str) -> Path:
        """Resolve filename and ensure it stays inside data_dir."""
        clean = Path(filename).name  # strip any directory traversal
        path = (self.data_dir / clean).resolve()
        if not str(path).startswith(str(self.data_dir)):
            raise ValueError(f"Access denied: {filename}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {clean}")
        return path

    def _read_csv(self, filename: str) -> tuple[list[str], list[dict]]:
        """Read a CSV and return (headers, rows). Handles BOM and whitespace."""
        path = self._safe_path(filename)
        mtime = path.stat().st_mtime
        cached = self._csv_cache.get(path.name)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]

        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in reader.fieldnames or []]
            rows = []
            for row in reader:
                rows.append({k.strip(): v.strip() for k, v in row.items()})
        self._csv_cache[path.name] = (mtime, headers, rows)
        return headers, rows

    def _find_student(self, student_name: str) -> dict:
        _, students = self._read_csv("students.csv")
        for row in students:
            if row.get("NAME", "").lower() == student_name.lower():
                return row
        raise ValueError(f"Student not found: {student_name}")

    def _subject_rows_for_student(self, student_name: str) -> list[dict]:
        _, rows = self._read_csv("subject_enrolment.csv")
        return [row for row in rows if row.get("STUDENT", "").lower() == student_name.lower()]

    def _attendance_rows_for_student(self, student_name: str) -> list[dict]:
        _, rows = self._read_csv("attendance.csv")
        return sorted(
            [row for row in rows if row.get("STUDENT", "").lower() == student_name.lower()],
            key=lambda row: self._sort_key(row.get("TERM", "")),
        )

    def _el_history_rows_for_student(self, student_name: str) -> list[dict]:
        _, rows = self._read_csv("el_history.csv")
        return sorted(
            [row for row in rows if row.get("STUDENT", "").lower() == student_name.lower()],
            key=lambda row: self._sort_key(row.get("YEAR", "")),
        )

    def _teacher_lookup(self) -> dict[tuple[str, str], str]:
        _, rows = self._read_csv("subject_teachers.csv")
        return {
            (row.get("SUBJECT", ""), row.get("RIGOUR", "")): row.get("TEACHER", "")
            for row in rows
        }

    def _class_lookup(self) -> dict[str, dict]:
        _, rows = self._read_csv("class_info.csv")
        return {row.get("CLASS", ""): row for row in rows}

    def _student_groups(self) -> dict[str, list[dict]]:
        _, rows = self._read_csv("subject_enrolment.csv")
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row.get("STUDENT", ""), []).append(row)
        return groups

    # ------------------------------------------------------------------ #
    #  Tool functions — each returns a JSON-serialisable dict
    # ------------------------------------------------------------------ #

    def list_files(self, params: dict) -> dict:
        """List all CSV files in the data directory."""
        files = sorted(self.data_dir.glob("*.csv"))
        return {
            "files": [
                {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
                for f in files
            ]
        }

    def read_headers(self, params: dict) -> dict:
        """
        Read column names and sample values from a CSV.
        This is the 'schema discovery' step — the LLM should always
        call this before querying so it knows what columns exist.
        """
        filename = params.get("file", "")
        headers, rows = self._read_csv(filename)
        samples = {}
        for h in headers:
            values = [r.get(h, "") for r in rows[:20] if r.get(h, "")]
            unique = sorted(set(values))[:5]
            samples[h] = unique
        return {
            "file": filename,
            "row_count": len(rows),
            "columns": headers,
            "sample_values": samples,
        }

    def get_student_overview(self, params: dict) -> dict:
        """
        Return the most common joined student facts in one call.
        Good for profile, subjects, rigour, attendance, EL history, classmates,
        and "most common classes" style questions.
        """
        student_name = params.get("student", "").strip()
        student = self._find_student(student_name)
        class_lookup = self._class_lookup()
        teacher_lookup = self._teacher_lookup()

        subjects = []
        for row in sorted(
            self._subject_rows_for_student(student_name),
            key=lambda item: (item.get("SUBJECT", ""), item.get("RIGOUR", "")),
        ):
            subjects.append({
                "subject": row.get("SUBJECT", ""),
                "rigour": row.get("RIGOUR", ""),
                "teacher": teacher_lookup.get((row.get("SUBJECT", ""), row.get("RIGOUR", ""))),
            })

        _, student_rows = self._read_csv("students.csv")
        same_academic_class = sorted(
            row.get("NAME", "")
            for row in student_rows
            if row.get("ACADEMIC CLASS") == student.get("ACADEMIC CLASS") and row.get("NAME") != student_name
        )
        same_el_class = sorted(
            row.get("NAME", "")
            for row in student_rows
            if row.get("EL CLASS") == student.get("EL CLASS") and row.get("NAME") != student_name
        )

        shared_classmates = []
        for row in student_rows:
            other_name = row.get("NAME", "")
            if other_name == student_name:
                continue
            shared_classes = int(row.get("ACADEMIC CLASS") == student.get("ACADEMIC CLASS"))
            shared_classes += int(row.get("EL CLASS") == student.get("EL CLASS"))
            if shared_classes:
                shared_classmates.append({
                    "student": other_name,
                    "shared_classes": shared_classes,
                    "same_academic_class": row.get("ACADEMIC CLASS") == student.get("ACADEMIC CLASS"),
                    "same_el_class": row.get("EL CLASS") == student.get("EL CLASS"),
                })
        shared_classmates.sort(key=lambda row: (-row["shared_classes"], row["student"]))
        max_shared = shared_classmates[0]["shared_classes"] if shared_classmates else 0
        profile_summary = (
            f"Name: {student.get('NAME', '')}; "
            f"Class: {student.get('ACADEMIC CLASS', '')}; "
            f"EL: {student.get('EL CLASS', '')}; "
            f"Form Teacher: {class_lookup.get(student.get('ACADEMIC CLASS', ''), {}).get('FORM TEACHER', '')}"
        )
        subject_summary = "; ".join(
            f"{item['subject']} ({item['rigour']})"
            for item in subjects
        )
        attendance_summary = "; ".join(
            f"{row.get('TERM', '')}: {row.get('ATTENDANCE RATE', '')}%"
            for row in self._attendance_rows_for_student(student_name)
        )
        el_history_summary = "; ".join(
            f"{row.get('YEAR', '')}: {row.get('EL GRADE', '')}"
            for row in self._el_history_rows_for_student(student_name)
        )

        return {
            "student": student_name,
            "profile_summary": profile_summary,
            "subject_summary": subject_summary,
            "attendance_summary": attendance_summary,
            "el_history_summary": el_history_summary,
            "profile": {
                "name": student.get("NAME", ""),
                "academic_class": student.get("ACADEMIC CLASS", ""),
                "el_class": student.get("EL CLASS", ""),
                "form_teacher": class_lookup.get(student.get("ACADEMIC CLASS", ""), {}).get("FORM TEACHER", ""),
            },
            "subjects": subjects,
            "attendance": self._attendance_rows_for_student(student_name),
            "el_history": self._el_history_rows_for_student(student_name),
            "same_academic_class": same_academic_class,
            "same_el_class": same_el_class,
            "shared_classmates": shared_classmates,
            "most_shared_classmates": [
                row["student"] for row in shared_classmates if row["shared_classes"] == max_shared
            ],
            "same_academic_class_summary": ", ".join(same_academic_class),
            "same_el_class_summary": ", ".join(same_el_class),
            "most_shared_classmates_summary": ", ".join(
                row["student"] for row in shared_classmates if row["shared_classes"] == max_shared
            ),
        }

    def get_student_teachers(self, params: dict) -> dict:
        """
        Return all teachers for a student by joining subject_enrolment and
        subject_teachers automatically.
        """
        student_name = params.get("student", "").strip()
        self._find_student(student_name)
        teacher_lookup = self._teacher_lookup()

        teacher_assignments = []
        for row in sorted(
            self._subject_rows_for_student(student_name),
            key=lambda item: (item.get("SUBJECT", ""), item.get("RIGOUR", "")),
        ):
            teacher_assignments.append({
                "subject": row.get("SUBJECT", ""),
                "rigour": row.get("RIGOUR", ""),
                "teacher": teacher_lookup.get((row.get("SUBJECT", ""), row.get("RIGOUR", ""))),
            })

        return {
            "student": student_name,
            "teacher_summary": "; ".join(
                f"{row['subject']} ({row['rigour']}): {row['teacher']}"
                for row in teacher_assignments
                if row.get("teacher")
            ),
            "teacher_assignments": teacher_assignments,
            "unique_teachers": sorted({
                row["teacher"] for row in teacher_assignments if row.get("teacher")
            }),
        }

    def get_student_location(self, params: dict) -> dict:
        """Return where a student should be for a specific day and period."""
        student_name = params.get("student", "").strip()
        day = params.get("day", "").strip()
        period = str(params.get("period", "")).strip()
        student = self._find_student(student_name)
        _, rows = self._read_csv("timetable.csv")

        for row in rows:
            if (
                row.get("CLASS") == student.get("ACADEMIC CLASS")
                and row.get("DAY", "").lower() == day.lower()
                and row.get("PERIOD", "") == period
            ):
                return {
                    "student": student_name,
                    "academic_class": student.get("ACADEMIC CLASS", ""),
                    "day": row.get("DAY", ""),
                    "period": row.get("PERIOD", ""),
                    "subject": row.get("SUBJECT", ""),
                    "room": row.get("ROOM", ""),
                    "summary": (
                        f"{row.get('SUBJECT', '')} in {row.get('ROOM', '')} "
                        f"({row.get('DAY', '')} Period {row.get('PERIOD', '')})"
                    ),
                }

        return {
            "student": student_name,
            "academic_class": student.get("ACADEMIC CLASS", ""),
            "day": day,
            "period": period,
            "error": "No timetable entry found for that slot.",
        }

    def find_students_same_subjects(self, params: dict) -> dict:
        """
        Find students whose subject set matches the target student.
        Returns both subject-only matches and stricter subject+rigour matches.
        """
        student_name = params.get("student", "").strip()
        self._find_student(student_name)
        student_groups = self._student_groups()
        my_rows = student_groups.get(student_name, [])
        my_subjects = sorted({row.get("SUBJECT", "") for row in my_rows})
        my_subject_rigour = sorted({
            (row.get("SUBJECT", ""), row.get("RIGOUR", "")) for row in my_rows
        })

        matches_by_subject = []
        matches_by_subject_and_rigour = []
        for other_name, rows in student_groups.items():
            if other_name == student_name:
                continue
            other_subjects = {row.get("SUBJECT", "") for row in rows}
            other_subject_rigour = {
                (row.get("SUBJECT", ""), row.get("RIGOUR", "")) for row in rows
            }
            if other_subjects == set(my_subjects):
                matches_by_subject.append(other_name)
            if other_subject_rigour == set(my_subject_rigour):
                matches_by_subject_and_rigour.append(other_name)

        return {
            "student": student_name,
            "target_subjects": my_subjects,
            "target_subjects_with_rigour": [
                {"subject": subject, "rigour": rigour}
                for subject, rigour in my_subject_rigour
            ],
            "matches_by_subject": sorted(matches_by_subject),
            "matches_by_subject_and_rigour": sorted(matches_by_subject_and_rigour),
            "matches_by_subject_summary": ", ".join(sorted(matches_by_subject)),
        }

    def get_level_teachers(self, params: dict) -> dict:
        """Return form teachers for all classes in a level such as S1."""
        level = params.get("level", "").strip()
        _, rows = self._read_csv("class_info.csv")
        level_rows = [row for row in rows if row.get("LEVEL", "").lower() == level.lower()]
        if not level_rows:
            return {"level": level, "classes": [], "form_teachers": []}
        return {
            "level": level,
            "classes": [
                {"class": row.get("CLASS", ""), "form_teacher": row.get("FORM TEACHER", "")}
                for row in sorted(level_rows, key=lambda row: row.get("CLASS", ""))
            ],
            "form_teachers": sorted({row.get("FORM TEACHER", "") for row in level_rows}),
            "form_teachers_summary": ", ".join(
                sorted({row.get("FORM TEACHER", "") for row in level_rows})
            ),
        }

    def query(self, params: dict) -> dict:
        """
        Filter and select from a CSV.

        Parameters:
            file (str):       CSV filename
            filters (list):   List of {"column", "op", "value"} dicts
                              ops: =, !=, >, <, >=, <=, contains
            columns (list):   Columns to return (default: all)
            sort_by (str):    Column to sort by (optional)
            sort_order (str): "asc" or "desc" (default: asc)
            limit (int):      Max rows to return (default: 50)
        """
        filename = params.get("file", "")
        headers, rows = self._read_csv(filename)
        filters = params.get("filters", [])
        select_cols = params.get("columns", headers)
        sort_by = params.get("sort_by")
        sort_order = params.get("sort_order", "asc")
        limit = min(params.get("limit", 50), 200)  # hard cap at 200

        # Apply filters
        filtered = rows
        for f in filters:
            col = f.get("column", "")
            op = f.get("op", "=")
            val = f.get("value", "")
            filtered = [r for r in filtered if self._match(r.get(col, ""), op, val)]

        # Sort
        if sort_by and sort_by in headers:
            filtered.sort(
                key=lambda r: self._sort_key(r.get(sort_by, "")),
                reverse=(sort_order == "desc"),
            )

        # Select columns and limit
        result_rows = []
        for r in filtered[:limit]:
            result_rows.append({c: r.get(c, "") for c in select_cols if c in r})

        return {
            "file": filename,
            "total_matching": len(filtered),
            "returned": len(result_rows),
            "rows": result_rows,
        }

    def summarise(self, params: dict) -> dict:
        """
        Aggregate data from a CSV.

        Parameters:
            file (str):       CSV filename
            group_by (str):   Column to group by
            metric (str):     One of: count, sum, avg, min, max
            column (str):     Column to aggregate (not needed for 'count')
            filters (list):   Optional pre-filters (same format as query)
        """
        filename = params.get("file", "")
        headers, rows = self._read_csv(filename)
        group_by = params.get("group_by", "")
        metric = params.get("metric", "count")
        agg_col = params.get("column", "")
        filters = params.get("filters", [])

        # Apply pre-filters
        filtered = rows
        for f in filters:
            col = f.get("column", "")
            op = f.get("op", "=")
            val = f.get("value", "")
            filtered = [r for r in filtered if self._match(r.get(col, ""), op, val)]

        # Group
        groups: dict[str, list] = {}
        for r in filtered:
            key = r.get(group_by, "(blank)")
            groups.setdefault(key, []).append(r)

        # Aggregate
        results = []
        for key, group_rows in sorted(groups.items()):
            entry = {group_by: key}
            if metric == "count":
                entry["count"] = len(group_rows)
            else:
                values = []
                for r in group_rows:
                    try:
                        values.append(float(r.get(agg_col, "")))
                    except (ValueError, TypeError):
                        pass
                if values:
                    if metric == "sum":
                        entry[f"sum_{agg_col}"] = sum(values)
                    elif metric == "avg":
                        entry[f"avg_{agg_col}"] = round(sum(values) / len(values), 2)
                    elif metric == "min":
                        entry[f"min_{agg_col}"] = min(values)
                    elif metric == "max":
                        entry[f"max_{agg_col}"] = max(values)
                else:
                    entry[f"{metric}_{agg_col}"] = None
            results.append(entry)

        return {
            "file": filename,
            "group_by": group_by,
            "metric": metric,
            "total_rows_before_grouping": len(filtered),
            "groups": results,
        }

    def distinct(self, params: dict) -> dict:
        """
        Get unique values and their counts for a column.

        Parameters:
            file (str):    CSV filename
            column (str):  Column to get distinct values for
        """
        filename = params.get("file", "")
        headers, rows = self._read_csv(filename)
        col = params.get("column", "")
        counter = Counter(r.get(col, "").strip() for r in rows)
        return {
            "file": filename,
            "column": col,
            "unique_count": len(counter),
            "values": [
                {"value": v, "count": c}
                for v, c in counter.most_common()
            ],
        }

    def cross_tabulate(self, params: dict) -> dict:
        """
        Cross-tabulate two columns (like a pivot table).

        Parameters:
            file (str):     CSV filename
            row_col (str):  Column for rows
            col_col (str):  Column for columns
        """
        filename = params.get("file", "")
        headers, rows = self._read_csv(filename)
        row_col = params.get("row_col", "")
        col_col = params.get("col_col", "")

        # Build cross-tab
        table: dict[str, Counter] = {}
        col_values: set[str] = set()
        for r in rows:
            rk = r.get(row_col, "(blank)")
            ck = r.get(col_col, "(blank)")
            col_values.add(ck)
            table.setdefault(rk, Counter())[ck] += 1

        sorted_cols = sorted(col_values)
        result = []
        for rk in sorted(table.keys()):
            entry = {row_col: rk}
            for ck in sorted_cols:
                entry[ck] = table[rk].get(ck, 0)
            entry["_total"] = sum(table[rk].values())
            result.append(entry)

        return {
            "file": filename,
            "row_column": row_col,
            "col_column": col_col,
            "table": result,
        }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _match(cell_value: str, op: str, test_value) -> bool:
        """Compare a cell value against a test value using the given operator."""
        if op in {"in", "not_in"}:
            if isinstance(test_value, list):
                test_values = [str(value).strip().lower() for value in test_value]
            else:
                parts = [part.strip() for part in str(test_value).split(",") if part.strip()]
                test_values = [part.lower() for part in (parts or [str(test_value).strip()])]
            is_match = cell_value.lower() in test_values
            return is_match if op == "in" else not is_match

        # Try numeric comparison first
        try:
            cell_num = float(cell_value)
            test_num = float(test_value)
            return {
                "=": cell_num == test_num,
                "!=": cell_num != test_num,
                ">": cell_num > test_num,
                "<": cell_num < test_num,
                ">=": cell_num >= test_num,
                "<=": cell_num <= test_num,
                "contains": test_value.lower() in cell_value.lower(),
            }.get(op, False)
        except (ValueError, TypeError):
            pass

        # Fall back to string comparison (case-insensitive)
        cell_lower = cell_value.lower()
        test_lower = test_value.lower()
        return {
            "=": cell_lower == test_lower,
            "!=": cell_lower != test_lower,
            ">": cell_lower > test_lower,
            "<": cell_lower < test_lower,
            ">=": cell_lower >= test_lower,
            "<=": cell_lower <= test_lower,
            "contains": test_lower in cell_lower,
        }.get(op, False)

    @staticmethod
    def _sort_key(value: str):
        """Sort numerically if possible, otherwise alphabetically."""
        try:
            return (0, float(value))
        except (ValueError, TypeError):
            return (1, value.lower())

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    TOOLS = {
        "list_files": list_files,
        "read_headers": read_headers,
        "get_student_overview": get_student_overview,
        "get_student_teachers": get_student_teachers,
        "get_student_location": get_student_location,
        "find_students_same_subjects": find_students_same_subjects,
        "get_level_teachers": get_level_teachers,
        "query": query,
        "summarise": summarise,
        "distinct": distinct,
        "cross_tabulate": cross_tabulate,
    }

    def execute(self, tool_name: str, params: dict) -> dict:
        """Execute a tool by name and return the result."""
        fn = self.TOOLS.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}. Available: {list(self.TOOLS.keys())}"}
        try:
            return fn(self, params)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)}"}


# ====================================================================== #
#  Tool definitions for LM Studio's OpenAI-compatible function calling
# ====================================================================== #

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all CSV files available in the data directory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_student_overview",
            "description": "Look up one student and return joined profile data in one call: class, EL class, form teacher, subjects, rigour, attendance, EL history, same EL classmates, same academic classmates, and most shared classmates. Use this for questions like 'Tell me about Qian Hui Zheng', 'What subjects does X take?', 'What's X's attendance?', 'Who is in the same EL class as X?', or 'Who has the most common classes as X?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student": {
                        "type": "string",
                        "description": "Exact student name, e.g. 'Qian Hui Zheng'",
                    }
                },
                "required": ["student"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_student_teachers",
            "description": "Look up all teachers for a student by joining subject enrolments with subject teachers automatically. Use this for questions like 'Who teaches Qian Hui Zheng?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student": {
                        "type": "string",
                        "description": "Exact student name, e.g. 'Qian Hui Zheng'",
                    }
                },
                "required": ["student"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_student_location",
            "description": "Look up where a student is scheduled to be for a specific day and period by joining the student roster with the timetable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student": {"type": "string", "description": "Exact student name"},
                    "day": {"type": "string", "description": "Day of week, e.g. 'Monday'"},
                    "period": {"type": "string", "description": "Timetable period number as a string, e.g. '1'"},
                },
                "required": ["student", "day", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_students_same_subjects",
            "description": "Find which students take the same subjects as a target student. Returns both subject-only matches and exact subject+rigour matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student": {"type": "string", "description": "Exact student name"},
                },
                "required": ["student"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_level_teachers",
            "description": "Return the form teachers for all classes in a level such as S1. Use this for questions like 'Who teaches S1 level?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "description": "Level code such as 'S1'"},
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_headers",
            "description": "Read column names and sample values from a CSV file. ALWAYS call this before querying a file so you know what columns exist. Do NOT guess column names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "The CSV filename, e.g. 'students.csv'",
                    }
                },
                "required": ["file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Filter and select rows from a CSV file. Use this to find specific records matching conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "The CSV filename"},
                    "filters": {
                        "type": "array",
                        "description": "Conditions to filter by",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "description": "Column name (must match exactly)"},
                                "op": {
                                    "type": "string",
                                    "enum": ["=", "!=", ">", "<", ">=", "<=", "contains", "in", "not_in"],
                                    "description": "Comparison operator",
                                },
                                "value": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Value to compare against. Use an array for in/not_in.",
                                },
                            },
                            "required": ["column", "op", "value"],
                        },
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which columns to include in results (default: all)",
                    },
                    "sort_by": {"type": "string", "description": "Column to sort by"},
                    "sort_order": {"type": "string", "enum": ["asc", "desc"], "description": "Sort direction (default: asc)"},
                    "limit": {"type": "integer", "description": "Max rows to return (default: 50, max: 200)"},
                },
                "required": ["file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise",
            "description": "Group and aggregate data from a CSV. Use this for questions like 'how many students per class' or 'average score by teacher'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "The CSV filename"},
                    "group_by": {"type": "string", "description": "Column to group by"},
                    "metric": {
                        "type": "string",
                        "enum": ["count", "sum", "avg", "min", "max"],
                        "description": "Aggregation type",
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to aggregate (not needed for 'count')",
                    },
                    "filters": {
                        "type": "array",
                        "description": "Optional pre-filters before grouping",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "op": {
                                    "type": "string",
                                    "enum": ["=", "!=", ">", "<", ">=", "<=", "contains", "in", "not_in"],
                                },
                                "value": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                },
                            },
                            "required": ["column", "op", "value"],
                        },
                    },
                },
                "required": ["file", "group_by", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "distinct",
            "description": "Get all unique values and their counts for a specific column. Good for understanding what values exist in the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "The CSV filename"},
                    "column": {"type": "string", "description": "Column to get unique values for"},
                },
                "required": ["file", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_tabulate",
            "description": "Cross-tabulate two columns to show counts in a grid. Like a pivot table. Good for questions like 'how many G1/G2/G3 students does each teacher have'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "The CSV filename"},
                    "row_col": {"type": "string", "description": "Column for the rows of the table"},
                    "col_col": {"type": "string", "description": "Column for the columns of the table"},
                },
                "required": ["file", "row_col", "col_col"],
            },
        },
    },
]


# ====================================================================== #
#  Optional: Simple HTTP server for the tool
# ====================================================================== #

class ToolHTTPHandler(BaseHTTPRequestHandler):
    """Minimal HTTP endpoint so you can call tools via POST requests too."""

    tool: CSVTool  # set by serve()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        tool_name = body.get("tool", "")
        params = body.get("params", {})
        result = self.tool.execute(tool_name, params)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, indent=2).encode())

    def log_message(self, format, *args):
        print(f"[csv_tool] {args[0]}")


def serve(data_dir: str, port: int = 5555):
    tool = CSVTool(data_dir=data_dir)
    ToolHTTPHandler.tool = tool
    server = HTTPServer(("127.0.0.1", port), ToolHTTPHandler)
    print(f"CSV Tool server running on http://127.0.0.1:{port}")
    print(f"Data directory: {tool.data_dir}")
    print(f"Available files: {[f.name for f in tool.data_dir.glob('*.csv')]}")
    server.serve_forever()


# ====================================================================== #
#  CLI
# ====================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Tool for Local LLMs")
    parser.add_argument("--data-dir", default=".", help="Directory containing CSV files")
    parser.add_argument("--serve", action="store_true", help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--tool", help="Tool to run (for CLI testing)")
    parser.add_argument("--params", default="{}", help="JSON params for the tool")

    args = parser.parse_args()

    if args.serve:
        serve(args.data_dir, args.port)
    elif args.tool:
        tool = CSVTool(data_dir=args.data_dir)
        result = tool.execute(args.tool, json.loads(args.params))
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
