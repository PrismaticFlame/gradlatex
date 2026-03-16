import argparse
from PIL import Image

def stack_vertical(img1_path, img2_path, output_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    width = max(img1.width, img2.width)
    combined = Image.new("RGB", (width, img1.height + img2.height), (255, 255, 255))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (0, img1.height))
    combined.save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack two images vertically into one.")
    parser.add_argument("img1", help="Path to the top image")
    parser.add_argument("img2", help="Path to the bottom image")
    parser.add_argument("output", help="Path for the output image")
    args = parser.parse_args()

    stack_vertical(args.img1, args.img2, args.output)
