import cv2
import argparse
import csv
from datetime import datetime
import numpy as np
import zipfile
import io
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
from pathlib import Path
import re


EXIF_CAPTURE_TIME_TAGS = (36867, 36868, 306)
FILENAME_CAPTURE_TIME_PATTERNS = (
    (re.compile(r"(?<!\d)(\d{8})[_T-]?(\d{6})(?!\d)"), "%Y%m%d%H%M%S"),
    (
        re.compile(
            r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})[T_ -](\d{2})[-_:](\d{2})[-_:](\d{2})(?!\d)"
        ),
        "%Y%m%d%H%M%S",
    ),
)


def normalize_capture_time(value):
    """Return an ISO-like capture timestamp or an empty string."""
    if not value:
        return ""

    value = str(value).strip().replace("\x00", "")
    for date_format in (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(value, date_format).isoformat()
        except ValueError:
            pass
    return ""


def capture_time_from_filename(filename):
    """Extract common GreenSkEye/camera timestamps from an archive filename."""
    basename = Path(filename).name
    for pattern, date_format in FILENAME_CAPTURE_TIME_PATTERNS:
        match = pattern.search(basename)
        if not match:
            continue

        compact_value = "".join(match.groups())
        try:
            return datetime.strptime(compact_value, date_format).isoformat()
        except ValueError:
            continue
    return ""


def get_capture_time(pil_image, filename):
    """Prefer embedded capture metadata and fall back to the filename."""
    exif = pil_image.getexif()
    for tag in EXIF_CAPTURE_TIME_TAGS:
        normalized_time = normalize_capture_time(exif.get(tag))
        if normalized_time:
            return normalized_time
    return capture_time_from_filename(filename)


def build_time_series(rows):
    """Return timestamped measurements in chronological order."""
    time_series = []
    for filename, capture_time, green_pixel_count, _ in rows:
        if not capture_time:
            continue
        try:
            timestamp = datetime.fromisoformat(capture_time)
        except ValueError:
            continue
        time_series.append((timestamp, green_pixel_count, filename))
    return sorted(time_series, key=lambda point: point[0])


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def generate_green_pixel_graph(rows, output_path):
    """Plot capture time against green-pixel count without extra dependencies."""
    time_series = build_time_series(rows)

    width, height = 1600, 900
    left, right, top, bottom = 160, 80, 110, 170
    plot_width = width - left - right
    plot_height = height - top - bottom

    graph = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(graph)
    title_font = load_font(42)
    label_font = load_font(28)
    tick_font = load_font(22)

    draw.text(
        (width / 2, 45),
        "Green Pixel Count Over Time",
        fill="#173f2b",
        font=title_font,
        anchor="mm",
    )

    if not time_series:
        draw.text(
            (width / 2, height / 2),
            "No images with a known capture time",
            fill="#4b5563",
            font=label_font,
            anchor="mm",
        )
        graph.save(output_path)
        return

    min_time = time_series[0][0]
    max_time = time_series[-1][0]
    time_span = (max_time - min_time).total_seconds()
    max_count = max(point[1] for point in time_series)
    y_max = max(max_count, 1)

    draw.line((left, top, left, top + plot_height), fill="#374151", width=3)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="#374151",
        width=3,
    )

    for tick_index in range(6):
        fraction = tick_index / 5
        y = top + plot_height - fraction * plot_height
        value = round(fraction * y_max)
        draw.line((left, y, left + plot_width, y), fill="#e5e7eb", width=2)
        draw.text(
            (left - 18, y),
            f"{value:,}",
            fill="#374151",
            font=tick_font,
            anchor="rm",
        )

    def x_position(timestamp):
        if time_span == 0:
            return left + plot_width / 2
        elapsed = (timestamp - min_time).total_seconds()
        return left + (elapsed / time_span) * plot_width

    def y_position(count):
        return top + plot_height - (count / y_max) * plot_height

    points = [
        (x_position(timestamp), y_position(count))
        for timestamp, count, _ in time_series
    ]
    if len(points) > 1:
        draw.line(points, fill="#079455", width=6, joint="curve")
    for x, y in points:
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="#079455", outline="white", width=3)

    label_indexes = sorted(
        set(round(index * (len(time_series) - 1) / min(5, len(time_series) - 1))
            for index in range(min(6, len(time_series))))
    ) if len(time_series) > 1 else [0]
    for index in label_indexes:
        timestamp = time_series[index][0]
        x = x_position(timestamp)
        draw.line(
            (x, top + plot_height, x, top + plot_height + 12),
            fill="#374151",
            width=2,
        )
        draw.text(
            (x, top + plot_height + 22),
            timestamp.strftime("%Y-%m-%d\n%H:%M:%S"),
            fill="#374151",
            font=tick_font,
            anchor="ma",
            align="center",
        )

    draw.text(
        (left + plot_width / 2, height - 42),
        "Capture Time",
        fill="#111827",
        font=label_font,
        anchor="mm",
    )
    draw.text(
        (left, top - 34),
        "Green Pixel Count",
        fill="#111827",
        font=label_font,
        anchor="lm",
    )

    graph.save(output_path)


def greenPixelAnalysisBatch(inputPath, outputDirPath):
    os.makedirs(outputDirPath, exist_ok=True)

    with zipfile.ZipFile(inputPath, "r") as zf:
        fileList = zf.namelist()
        rows = []

        for filename in fileList:
            if filename.endswith("/"):
                continue

            try:
                imageBytes = zf.read(filename)
                imageStream = io.BytesIO(imageBytes)

                pilImage = Image.open(imageStream)
                captureTime = get_capture_time(pilImage, filename)
                pilImage = ImageOps.exif_transpose(pilImage)
                pilImage = pilImage.convert("RGB")
                image = np.array(pilImage)

                hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                lowerGreen = np.array([40, 40, 40])
                upperGreen = np.array([80, 255, 255])

                mask = cv2.inRange(hsvImage, lowerGreen, upperGreen)
                greenPixelCount = cv2.countNonZero(mask)
                height, width = mask.shape[:2]
                greenPixelPercentage = (greenPixelCount / (height * width)) * 100

                rows.append(
                    [
                        filename,
                        captureTime,
                        greenPixelCount,
                        greenPixelPercentage,
                    ]
                )

                originalStem = Path(filename).stem
                maskFilename = f"{originalStem}_green_mask.png"
                maskPath = os.path.join(outputDirPath, maskFilename)

                cv2.imwrite(maskPath, mask)

            except Image.UnidentifiedImageError:
                print(f"Could not identify image format for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        resultsPath = os.path.join(outputDirPath, "results.csv")
        with open(resultsPath, "w", newline="", encoding="utf-8") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(
                [
                    "Image Filename",
                    "Capture Time",
                    "Green Pixel Count",
                    "Green Pixel Percentage",
                ]
            )
            writer.writerows(rows)

        graphPath = os.path.join(outputDirPath, "green_pixel_count_over_time.png")
        generate_green_pixel_graph(rows, graphPath)

    print("Green-pixel analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Measure green pixels and generate masks for a batch of images"
    )
    parser.add_argument("--input", required=True, help="Path to the input zip file")
    parser.add_argument("--output", required=True, help="Path to save output mask images")

    args = parser.parse_args()
    greenPixelAnalysisBatch(args.input, args.output)


if __name__ == "__main__":
    main()
