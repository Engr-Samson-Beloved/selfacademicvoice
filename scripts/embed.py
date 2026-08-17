import json
import re
from pathlib import Path

import docx
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------
# Configuration
# ---------------------------------------

DATASET_DIR = Path("data/dataset")
DB_DIR = "data/vector_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_CHARS = 100
MAX_CHARS = 400


def is_prose(text):
    words = text.split()
    return len(words) >= 15 and "." in text and not text[0].isdigit()


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_paragraph(text):
    if len(text) < MIN_CHARS:
        return []
    if len(text) <= MAX_CHARS:
        return [text]
    chunks = []
    current = ""
    for sentence in split_into_sentences(text):
        if len(current) + len(sentence) + 1 > MAX_CHARS and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DB_DIR)

try:
    client.delete_collection("ghostwriter")
    print("Old collection deleted.")
except:
    pass

collection = client.create_collection("ghostwriter")

# ---------------------------------------
# Read reports
# ---------------------------------------

documents = []
metadatas = []
ids = []

counter = 0

for file in DATASET_DIR.glob("*.json"):

    report_name = file.stem

    print(f"Reading {report_name}")

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:

        paragraph = item.get("paragraph", "").strip()

        if not paragraph:
            continue

        section = item.get("section", "Unknown")

        for chunk in chunk_paragraph(paragraph):
            documents.append(chunk)
            metadatas.append({"report": report_name, "section": section})
            ids.append(f"{report_name}_{counter}")
            counter += 1

# ---------------------------------------
# Read docx files
# ---------------------------------------

for file in DATASET_DIR.glob("*.docx"):

    report_name = file.stem

    print(f"Reading {report_name}")

    doc = docx.Document(file)

    for para in doc.paragraphs:
        text = para.text.strip()
        for chunk in chunk_paragraph(text):
            if not is_prose(chunk):
                continue
            documents.append(chunk)
            metadatas.append({"report": report_name, "section": "Text"})
            ids.append(f"{report_name}_{counter}")
            counter += 1

print(f"\nCollected {len(documents)} paragraphs (after chunking).")

# ---------------------------------------
# Create embeddings
# ---------------------------------------

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
).tolist()

# ---------------------------------------
# Store in ChromaDB
# ---------------------------------------

print("Saving to ChromaDB...")

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("\nDone!")

print(f"Stored {collection.count()} documents.")