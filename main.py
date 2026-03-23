import cv2
import argparse
import numpy as np 
import zipfile
import io
from PIL import Image

import csv

def greenPixelAnalyisBatch(inputPath, outputDirPath):
    with zipfile.ZipFile(inputPath, 'r') as zf:
        
        fileList = zf.namelist()
        
        rows = [
            ["Green Pixel Count", "Green Pixel Percentage"]
        ]
   
        for filename in fileList:
            try:
                imageBytes = zf.read(filename)
                imageStream = io.BytesIO(imageBytes)
                
                pilImage = Image.open(imageStream)
                
                image = np.array(pilImage)

    
                hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                lowerGreen = np.array([40, 40, 40])
                upperGreen = np.array([80, 255, 255])
                
                height, width = hsvImage.shape[:2]
                mask = cv2.inRange(hsvImage, lowerGreen, upperGreen)
                
                greenPixelCount = cv2.countNonZero(mask)
                greenPixelPercentage = greenPixelCount / (height*width)
                
                rows.append([greenPixelCount, greenPixelPercentage])
                
            except Image.UnidentifiedImageError:
                print(f"Could not identify image format for {filename}")
            except Exception as e:
                print(f"Error process {filename}: {e}")
                
                
        with open(outputDirPath+'/results.txt', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
                
            
    
        

        print("Analysis complete.")

def main():
    parser = argparse.ArgumentParser(description="Analyze a batch of images and count the average green pixels")
    parser.add_argument('--input', required=True, help='Path to the input zip file')
    parser.add_argument('--output', required=True, help='Path to save the analysis output')

    args = parser.parse_args()
    greenPixelAnalyisBatch(args.input, args.output)


if __name__ == "__main__":
    main()
