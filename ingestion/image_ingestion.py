import os
import re
import uuid
import torch
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# =====================================================
# CONFIG
# =====================================================

QDRANT_URL = "http://localhost:6335"
COLLECTION = "page_images"

BASE_DIR = "extracted/class10_science"

CLASS = 10
SUBJECT = "Science"

BATCH_SIZE = 16          # keep small for stability
VECTOR_SIZE = 512

# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =====================================================
# CLIP MODEL
# =====================================================

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()

# =====================================================
# UTILITIES
# =====================================================

def extract_page_number(filename: str):
    """
    Extracts page number from filenames like:
    page_3.png
    page_12_img_1.jpg
    """
    match = re.search(r"page_(\d+)", filename)
    return int(match.group(1)) if match else None


def upsert_batch(batch):
    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": batch},
        timeout=60
    )

    if r.status_code != 200:
        print("❌ Qdrant Error")
        print("Status:", r.status_code)
        print("Response:", r.text)
        print("Sample point:", batch[0])
        r.raise_for_status()

    print(f"✔ Upserted {len(batch)} image vectors")


# =====================================================
# MAIN INGESTION
# =====================================================

def ingest_images():
    batch = []

    for chapter_folder in sorted(os.listdir(BASE_DIR)):
        if not chapter_folder.startswith("chapter"):
            continue

        chapter_number = int(chapter_folder.replace("chapter", ""))
        image_dir = os.path.join(BASE_DIR, chapter_folder, "images")

        if not os.path.exists(image_dir):
            continue

        print(f"📘 Processing Chapter {chapter_number}")

        for file in sorted(os.listdir(image_dir)):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            page_number = extract_page_number(file)
            if page_number is None:
                continue

            img_path = os.path.join(image_dir, file)

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"⚠ Skipping image {file}: {e}")
                continue

            inputs = processor(
                images=image,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                vector = model.get_image_features(**inputs)[0].cpu().tolist()

            # ---------- HARD VALIDATION ----------
            if len(vector) != VECTOR_SIZE:
                raise ValueError(
                    f"Invalid vector size {len(vector)} for {file}"
                )

            batch.append({
                "id": str(uuid.uuid4()),   # ✅ REQUIRED FOR REST API
                "vector": vector,
                "payload": {
                    "class": CLASS,
                    "subject": SUBJECT,
                    "chapter": chapter_number,
                    "page_number": page_number,
                    "content_type": "page_image",
                    "file_name": file
                }
            })

            if len(batch) >= BATCH_SIZE:
                upsert_batch(batch)
                batch.clear()

    # final flush
    if batch:
        upsert_batch(batch)

    print("\n🎉 IMAGE INGESTION COMPLETED FOR ALL CHAPTERS")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    ingest_images()
