import cv2
import argparse
import numpy as np
import zipfile
import io
from PIL import Image
import csv
import os
from pathlib import Path


def greenPixelAnalysisBatch(inputPath, outputDirPath):
    os.makedirs(outputDirPath, exist_ok=True)

    resultsPath = os.path.join(outputDirPath, "results.csv")

    with zipfile.ZipFile(inputPath, "r") as zf:
        fileList = zf.namelist()

        rows = [
            ["Filename", "Green Pixel Count", "Green Pixel Percentage", "Mask Path"]
        ]

        for filename in fileList:
            if filename.endswith("/"):
                continue

            try:
                imageBytes = zf.read(filename)
                imageStream = io.BytesIO(imageBytes)

                pilImage = Image.open(imageStream).convert("RGB")
                image = np.array(pilImage)

                hsvImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                lowerGreen = np.array([40, 40, 40])
                upperGreen = np.array([80, 255, 255])

                height, width = hsvImage.shape[:2]
                mask = cv2.inRange(hsvImage, lowerGreen, upperGreen)

                greenPixelCount = cv2.countNonZero(mask)
                greenPixelPercentage = greenPixelCount / (height * width)

                originalStem = Path(filename).stem
                maskFilename = f"{originalStem}_green_mask.png"
                maskPath = os.path.join(outputDirPath, maskFilename)

                cv2.imwrite(maskPath, mask)

                rows.append([
                    filename,
                    greenPixelCount,
                    greenPixelPercentage,
                    maskFilename
                ])

            except Image.UnidentifiedImageError:
                print(f"Could not identify image format for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(resultsPath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a batch of images and count green pixels"
    )
    parser.add_argument("--input", required=True, help="Path to the input zip file")
    parser.add_argument("--output", required=True, help="Path to save the analysis output")

    args = parser.parse_args()
    greenPixelAnalysisBatch(args.input, args.output)


if __name__ == "__main__":
    main()
