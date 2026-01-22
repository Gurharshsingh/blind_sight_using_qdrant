import os
import fitz  
# -------- INPUT --------
PDF_FOLDER = "data/class10/science"

# -------- OUTPUT --------
OUTPUT_BASE = "extracted/class10_science"

# -------- FILTERS --------
MIN_WIDTH = 120
MIN_HEIGHT = 120


def extract_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    total_saved = 0

    os.makedirs(output_dir, exist_ok=True)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1  

        img_counter = 0

        for img in page.get_images(full=True):
            xref = img[0]

            try:
                pix = fitz.Pixmap(doc, xref)

                
                if pix.n > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                
                if pix.width < MIN_WIDTH or pix.height < MIN_HEIGHT:
                    continue

                image_path = os.path.join(
                    output_dir,
                    f"page_{page_number}_img_{img_counter}.png"
                )

                pix.save(image_path)

                img_counter += 1
                total_saved += 1

            except Exception:
                continue

        print(f"✔ Page {page_number}: {img_counter} images saved")

    print(f"✔ DONE ({os.path.basename(pdf_path)}): {total_saved} images extracted\n")


def extract_all_chapters():
    for file in os.listdir(PDF_FOLDER):
        if not file.lower().endswith(".pdf"):
            continue

        chapter_name = os.path.splitext(file)[0]  
        pdf_path = os.path.join(PDF_FOLDER, file)

        chapter_output_dir = os.path.join(
            OUTPUT_BASE,
            chapter_name,
            "images"
        )

        print(f"\n📘 Processing {chapter_name}")
        extract_images_from_pdf(pdf_path, chapter_output_dir)


if __name__ == "__main__":
    extract_all_chapters()
