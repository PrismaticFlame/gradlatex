import argparse
from pathlib import Path
from PIL import Image
import fitz  # pymupdf

def pdf_to_image(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(img)
    doc.close()
    if len(pages) == 1:
        return pages[0]
    # Stack multiple pages vertically
    width = max(p.width for p in pages)
    height = sum(p.height for p in pages)
    combined = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for p in pages:
        combined.paste(p, (0, y))
        y += p.height
    return combined

def open_image(path):
    if Path(path).suffix.lower() == ".pdf":
        return pdf_to_image(path)
    return Image.open(path)

def stack_vertical(img1_path, img2_path, output_path):
    img1 = open_image(img1_path)
    img2 = open_image(img2_path)
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
