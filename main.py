import cv2
import argparse
import numpy as np
import zipfile
import io
from PIL import Image
import os
from pathlib import Path


def greenPixelAnalysisBatch(inputPath, outputDirPath):
    os.makedirs(outputDirPath, exist_ok=True)

    with zipfile.ZipFile(inputPath, "r") as zf:
        fileList = zf.namelist()

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

                mask = cv2.inRange(hsvImage, lowerGreen, upperGreen)

                originalStem = Path(filename).stem
                maskFilename = f"{originalStem}_green_mask.png"
                maskPath = os.path.join(outputDirPath, maskFilename)

                cv2.imwrite(maskPath, mask)

            except Image.UnidentifiedImageError:
                print(f"Could not identify image format for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("Mask generation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate green-pixel masks for a batch of images"
    )
    parser.add_argument("--input", required=True, help="Path to the input zip file")
    parser.add_argument("--output", required=True, help="Path to save output mask images")

    args = parser.parse_args()
    greenPixelAnalysisBatch(args.input, args.output)


if __name__ == "__main__":
    main()
