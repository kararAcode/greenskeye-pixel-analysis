# greenskeye-pixel-analysis

Batch green-pixel analysis model for the GreenSkEye job scheduler.

The model accepts a ZIP archive of images. For each recognizable image, it:

- generates a binary green-pixel mask;
- counts the green pixels;
- calculates the percentage of the full image classified as green; and
- records the image filename and capture time in `results.csv`.

Capture time is read from standard image EXIF metadata when available. If the
image has no usable EXIF timestamp, the model recognizes timestamps such as
`YYYYMMDD_HHMMSS` and `YYYY-MM-DD_HH-MM-SS` in the filename. Capture time is
left blank rather than guessed when neither source is available.

`results.csv` contains:

```text
Image Filename,Capture Time,Green Pixel Count,Green Pixel Percentage
```

Green Pixel Percentage is reported on a 0–100 scale.

Run the tests with:

```bash
python -m unittest
```
