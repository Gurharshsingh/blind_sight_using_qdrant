import re
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from speech_to_text import speech_to_text
from text_to_speech import speak


# =====================================================
# CONFIG
# =====================================================

QDRANT_URL = "http://localhost:6335"

TEXT_COLLECTION = "text_pages"
IMAGE_COLLECTION = "page_images"

TOP_K_TEXT = 1
TOP_K_IMAGES = 1

OPENROUTER_API_KEY = ""
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-oss-120b:free"

device = "cpu"

# =====================================================
# MODELS
# =====================================================

text_model = SentenceTransformer("all-MiniLM-L6-v2")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# =====================================================
# UTILS
# =====================================================

def is_image_question(question: str) -> bool:
    keywords = [
        "image", "diagram", "picture",
        "figure", "illustration", "shown"
    ]
    return any(k in question.lower() for k in keywords)


def extract_chapter(question: str):
    m = re.search(r"chapter\s*(\d+)", question.lower())
    return int(m.group(1)) if m else None

# =====================================================
# QDRANT SEARCH
# =====================================================

def search_text(question, class_no, subject, chapter=None):
    vector = text_model.encode(question).tolist()

    must = [
        {"key": "class", "match": {"value": class_no}},
        {"key": "subject", "match": {"value": subject}},
        {"key": "content_type", "match": {"value": "page_text"}}
    ]

    if chapter:
        must.append({"key": "chapter", "match": {"value": chapter}})

    payload = {
        "vector": vector,
        "limit": TOP_K_TEXT,
        "with_payload": True,
        "filter": {"must": must}
    }

    r = requests.post(
        f"{QDRANT_URL}/collections/{TEXT_COLLECTION}/points/search",
        json=payload
    )
    r.raise_for_status()
    return r.json()["result"]


def search_images(question, class_no, subject, chapter=None):
    inputs = clip_processor(
        text=[question],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        vector = clip_model.get_text_features(**inputs)[0].cpu().tolist()

    must = [
        {"key": "class", "match": {"value": class_no}},
        {"key": "subject", "match": {"value": subject}},
        {"key": "content_type", "match": {"value": "page_image"}}
    ]

    if chapter:
        must.append({"key": "chapter", "match": {"value": chapter}})

    payload = {
        "vector": vector,
        "limit": TOP_K_IMAGES,
        "with_payload": True,
        "filter": {"must": must}
    }

    r = requests.post(
        f"{QDRANT_URL}/collections/{IMAGE_COLLECTION}/points/search",
        json=payload
    )
    r.raise_for_status()
    return r.json()["result"]

# =====================================================
# PROMPT BUILDERS
# =====================================================

def build_text_prompt(question, text_results):
    text_block = "\n\n".join(
        f"[Chapter {r['payload']['chapter']} – Page {r['payload']['page_number']}]\n"
        f"{r['payload']['text']}"
        for r in text_results
    )

    return f"""
You are a science teacher.

Answer the question using ONLY the textbook content below.
TEXTBOOK CONTENT:
{text_block}

QUESTION:
{question}

ANSWER:
""".strip()


def build_image_prompt(question, image_results, text_results):
    pages = sorted({
        img["payload"]["page_number"]
        for img in image_results
    })

    text_block = "\n\n".join(
        f"[Page {r['payload']['page_number']}]\n{r['payload']['text']}"
        for r in text_results
        if r["payload"]["page_number"] in pages
    )

    image_block = "\n".join(
        f"- Diagram on Page {img['payload']['page_number']}"
        for img in image_results
    )

    return f"""
You are a science teacher.

Explain what is shown in the diagram(s) using ONLY the textbook content below.

DIAGRAM PAGES:
{image_block}

TEXTBOOK EXPLANATION:
{text_block}

QUESTION:
{question}

ANSWER:
""".strip()

# =====================================================
# OPENROUTER
# =====================================================

def generate_answer(prompt):
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1000
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# =====================================================
# CLI
# =====================================================

def main():
    print("\n📘 Multi-Class Science Q&A (type 'exit')\n")

    class_no = int(input("Enter class: ").strip())
    subject = input("Enter subject: ").strip()

    while True:
        mode = input("\nType 't' for text or 's' for speech: ").strip().lower()

        if mode == "s":
            question = speech_to_text(duration=5)
            print(f"\n🗣️ You said: {question}")
        else:
            question = input("\nAsk: ").strip()




        if question.lower() == "exit":
            break

        chapter = extract_chapter(question)

        if is_image_question(question):
            image_results = search_images(
                question, class_no, subject, chapter
            )

            if not image_results:
                print("\n❌ No related images found.\n")
                continue

            text_results = search_text(
                question, class_no, subject, chapter
            )

            prompt = build_image_prompt(
                question, image_results, text_results
            )
        else:
            text_results = search_text(
                question, class_no, subject, chapter
            )

            if not text_results:
                print("\n❌ No relevant text found.\n")
                continue

            prompt = build_text_prompt(question, text_results)

        answer = generate_answer(prompt)

        print("\n📄 Answer:\n")
        print(answer)
        speak(answer)
        print("\n" + "-" * 70)

# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    main()

