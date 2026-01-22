import requests as req


BASE = "http://localhost:6335"  # change if needed

def create_collection(name, size):
    url = f"{BASE}/collections/{name}"
    payload = {
        "vectors": {
            "size": size,
            "distance": "Cosine"
        }
    }

    r = req.put(url, json=payload)

    if r.status_code == 409:
        print(f"ℹ Collection already exists: {name}")
        return

    r.raise_for_status()
    print(f"✔ Created collection: {name}")


# ---- Create collections ----
create_collection("text_pages", 384)
# create_collection("page_images", 512)
