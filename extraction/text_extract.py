import os
import fitz  # PyMuPDF

# -------- INPUT --------
PDF_FOLDER = "data/class10/science" # change class to class6 or class7 or class8 as needed

# -------- OUTPUT --------
OUTPUT_BASE = "extracted/class10_science"   # change class to class6 or class7 or class8 as needed


def extract_raw_text_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1  

        
        text = page.get_text(
            "text",
            flags=fitz.TEXT_PRESERVE_WHITESPACE
        )

        output_path = os.path.join(
            output_dir, f"page_{page_number}.txt"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✔ Page {page_number}: raw text extracted")

    print(f"✔ DONE ({os.path.basename(pdf_path)})\n")


def extract_all_chapters():
    for file in sorted(os.listdir(PDF_FOLDER)):
        if not file.lower().endswith(".pdf"):
            continue

        chapter_name = os.path.splitext(file)[0]  
        pdf_path = os.path.join(PDF_FOLDER, file)

        chapter_output_dir = os.path.join(
            OUTPUT_BASE,
            chapter_name,
            "page_text"
        )

        print(f"\n📘 Processing {chapter_name}")
        extract_raw_text_from_pdf(pdf_path, chapter_output_dir)


if __name__ == "__main__":
    extract_all_chapters()
