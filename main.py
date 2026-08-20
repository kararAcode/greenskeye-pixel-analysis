import cv2
import argparse
import csv
from datetime import datetime
import numpy as np
import zipfile
import io
from PIL import Image, ImageOps
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
