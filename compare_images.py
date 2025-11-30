"""
Compare two images pixel-wise. Produces diff.png if different and prints max channel difference.
Usage:
  python compare_images.py seq.png par.png
"""

from PIL import Image, ImageChops
import sys
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("usage: compare_images.py seq.png par.png")
        return
    a = Image.open(sys.argv[1]).convert('RGB')
    b = Image.open(sys.argv[2]).convert('RGB')
    if a.size != b.size:
        print("DIFFERENT SIZES", a.size, b.size)
        return
    diff = ImageChops.difference(a,b)
    bbox = diff.getbbox()
    if bbox is None:
        print("IDENTICAL")
        return
    diff_arr = np.array(diff)
    print("Max per-channel diff:", diff_arr.max())
    diff.save("results/diff.png")
    print("Saved diff to results/diff.png, bbox:", bbox)

if __name__ == '__main__':
    main()
