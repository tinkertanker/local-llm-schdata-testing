import unittest

from benchmark import _extract_facts, score_answer
from csv_tool import CSVTool


class CSVToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = CSVTool("./sample_data")

    def test_get_student_teachers_returns_joined_assignments(self):
        result = self.tool.execute("get_student_teachers", {"student": "Qian Hui Zheng"})
        assignments = {
            (row["subject"], row["rigour"], row["teacher"])
            for row in result["teacher_assignments"]
        }
        self.assertIn(
            ("English Language", "Higher", "GWENDOLYN CABRERA"),
            assignments,
        )
        self.assertIn(("Art", "Standard", "BRAIN BRANDT"), assignments)

    def test_find_students_same_subjects_returns_expected_matches(self):
        result = self.tool.execute("find_students_same_subjects", {"student": "Qian Hui Zheng"})
        self.assertEqual(
            result["matches_by_subject"],
            [
                "Aiden Tan",
                "Benjamin Seah",
                "Daniel Ho",
                "Ethan Koh",
                "Grace Lee",
                "Isla Wong",
                "Kayla Teo",
                "Liam Goh",
                "Noah Sim",
                "Sarah Yeo",
                "Wei Lin Tan",
                "Xavier Chong",
            ],
        )
        self.assertEqual(result["matches_by_subject_and_rigour"], [])

    def test_get_student_location_joins_timetable(self):
        result = self.tool.execute(
            "get_student_location",
            {"student": "Qian Hui Zheng", "day": "Monday", "period": "1"},
        )
        self.assertEqual(result["subject"], "English Language")
        self.assertEqual(result["room"], "Room 301")

    def test_get_student_overview_includes_classmates_and_profile(self):
        result = self.tool.execute("get_student_overview", {"student": "Qian Hui Zheng"})
        self.assertEqual(result["profile"]["academic_class"], "S1 EXCELLENCE 1")
        self.assertEqual(result["most_shared_classmates"], ["Sarah Yeo"])
        self.assertIn("Ryan Tay", result["same_academic_class"])
        self.assertIn("Sarah Yeo", result["same_el_class"])

    def test_query_supports_in_filters(self):
        result = self.tool.execute(
            "query",
            {
                "file": "subject_teachers.csv",
                "filters": [
                    {
                        "column": "SUBJECT",
                        "op": "in",
                        "value": ["English Language", "Science"],
                    }
                ],
                "columns": ["SUBJECT"],
            },
        )
        subjects = sorted({row["SUBJECT"] for row in result["rows"]})
        self.assertEqual(subjects, ["English Language", "Science"])

    def test_extract_facts_splits_compound_profile_truth(self):
        facts = _extract_facts("Name: Qian Hui Zheng, Class: S1 EXCELLENCE 1, EL: EL - G3")
        self.assertIn("name", facts)
        self.assertIn("qian hui zheng", facts)
        self.assertIn("class", facts)
        self.assertIn("s1 excellence 1", facts)
        self.assertIn("el", facts)
        self.assertIn("el - g3", facts)

    def test_extract_facts_splits_subject_and_rigour_pairs(self):
        facts = _extract_facts("English Language (Higher): GWENDOLYN CABRERA")
        self.assertIn("english language", facts)
        self.assertIn("higher", facts)
        self.assertIn("gwendolyn cabrera", facts)

    def test_q14_scoring_accepts_the_shared_classes_interpretation(self):
        result = score_answer(
            "Sarah Yeo shares the most classes with Qian Hui Zheng.",
            "Ryan Tay, Sarah Yeo, Timothy Loh",
            13,
        )
        self.assertEqual(result["label"], "PASS")


if __name__ == "__main__":
    unittest.main()
