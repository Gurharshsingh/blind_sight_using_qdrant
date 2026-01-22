import os
import uuid
import requests
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
QDRANT = "http://localhost:6335"
COLLECTION = "text_pages"
BASE_DIR = "extracted/class10_science"

CLASS = 10
SUBJECT = "Science"

BATCH_SIZE = 50
VECTOR_SIZE = 384 

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- UPSERT ----------------
def upsert_batch(batch):
    r = requests.put(
        f"{QDRANT}/collections/{COLLECTION}/points",
        json={"points": batch},
        timeout=60
    )

    if r.status_code != 200:
        print("❌ Qdrant Error")
        print("Status:", r.status_code)
        print("Response:", r.text)
        print("Sample point:", batch[0])
        r.raise_for_status()

    print(f"✔ Upserted {len(batch)} text vectors")

# ---------------- INGEST ----------------
batch = []

for chapter_folder in sorted(os.listdir(BASE_DIR)):
    if not chapter_folder.startswith("chapter"):
        continue

    chapter_number = int(chapter_folder.replace("chapter", ""))
    text_dir = os.path.join(BASE_DIR, chapter_folder, "page_text")

    if not os.path.exists(text_dir):
        continue

    print(f"📘 Processing Chapter {chapter_number}")

    for file in sorted(os.listdir(text_dir)):
        if not file.endswith(".txt"):
            continue

        page_number = int(file.replace("page_", "").replace(".txt", ""))

        with open(os.path.join(text_dir, file), encoding="utf-8") as f:
            text = f.read().strip()

        if len(text) < 50:
            continue

        vector = model.encode(text).tolist()

        # HARD CHECK
        if len(vector) != VECTOR_SIZE:
            raise ValueError(f"Invalid vector size {len(vector)}")

        batch.append({
        "id": str(uuid.uuid4()),   # REST-compliant ID
        "vector": vector,
        "payload": {
            "class": CLASS,
            "subject": SUBJECT,
            "chapter": chapter_number,
            "page_number": page_number,
            "content_type": "page_text",
            "text": text   # 🔥 THIS LINE IS CRITICAL
    }
})

        if len(batch) >= BATCH_SIZE:
            upsert_batch(batch)
            batch.clear()

# final flush
if batch:
    upsert_batch(batch)

print("\n🎉 TEXT INGESTION COMPLETED FOR ALL CHAPTERS")
