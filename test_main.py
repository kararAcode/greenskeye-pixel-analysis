import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from main import build_time_series, capture_time_from_filename, greenPixelAnalysisBatch


class CaptureTimeTests(unittest.TestCase):
    def test_extracts_compact_capture_time(self):
        self.assertEqual(
            capture_time_from_filename("plant_20260820_143015.jpeg"),
            "2026-08-20T14:30:15",
        )

    def test_extracts_separated_capture_time(self):
        self.assertEqual(
            capture_time_from_filename("plant_2026-08-20_14-30-15.png"),
            "2026-08-20T14:30:15",
        )

    def test_unknown_capture_time_is_blank(self):
        self.assertEqual(capture_time_from_filename("plant.jpeg"), "")

    def test_time_series_is_sorted_and_ignores_unknown_times(self):
        rows = [
            ["later.png", "2026-08-21T14:30:15", 20, 20.0],
            ["unknown.png", "", 30, 30.0],
            ["earlier.png", "2026-08-20T14:30:15", 10, 10.0],
        ]
        series = build_time_series(rows)
        self.assertEqual([point[2] for point in series], ["earlier.png", "later.png"])


class BatchAnalysisTests(unittest.TestCase):
    def test_writes_identified_measurement_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "plant_20260820_143015.png"
            Image.new("RGB", (4, 2), color=(0, 255, 0)).save(image_path)

            archive_path = temp_path / "images.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(image_path, arcname=image_path.name)

            output_path = temp_path / "output"
            greenPixelAnalysisBatch(archive_path, output_path)

            with open(output_path / "results.csv", newline="", encoding="utf-8") as results:
                rows = list(csv.reader(results))

            self.assertEqual(
                rows[0],
                [
                    "Image Filename",
                    "Capture Time",
                    "Green Pixel Count",
                    "Green Pixel Percentage",
                ],
            )
            self.assertEqual(rows[1][0], image_path.name)
            self.assertEqual(rows[1][1], "2026-08-20T14:30:15")
            self.assertEqual(rows[1][2], "8")
            self.assertEqual(float(rows[1][3]), 100.0)
            self.assertTrue((output_path / "plant_20260820_143015_green_mask.png").exists())
            graph_path = output_path / "green_pixel_count_over_time.png"
            self.assertTrue(graph_path.exists())
            with Image.open(graph_path) as graph:
                self.assertEqual(graph.size, (1600, 900))


if __name__ == "__main__":
    unittest.main()
