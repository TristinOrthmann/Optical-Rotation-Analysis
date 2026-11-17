# Optical Rotation Analysis

This project is a single Python application for estimating a crystal's optical
rotation angle from a sequence of JPG images.

## Run It

```bash
python3 optical_rotation.py
```

## Use It

1. Choose the folder containing your JPG images. Images are processed in
   filename order, so name them in measurement order, such as `image_001.jpg`.
2. The first image appears on the left. Drag a rectangle around the crystal, or
   click **Use Whole Image**.
3. Enter the starting angle and choose either an angle increment or ending
   angle. Negative increments are supported.
4. Enter a graph title and optional axis titles.
5. Click **Analyze Images**.

The graph appears in the application and the chosen results folder receives:

- `results.csv`: filename, angle, and measured mean intensity
- `fit.png`: graph of measurements and the sinusoidal fit

The default fit uses a 180-degree period and two harmonics. A setting of `1`
uses a simple cosine; increase the harmonic count when the curve visibly needs
more flexibility, while keeping it low enough not to follow noise.
