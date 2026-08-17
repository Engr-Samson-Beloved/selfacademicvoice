from pathlib import Path
from docx import Document

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("RAW_DIR:", RAW_DIR)
print("Files found:", list(RAW_DIR.glob("*.docx")))

for file in RAW_DIR.glob("*.docx"):
    doc = Document(file)

    paragraphs = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)

    output_file = OUTPUT_DIR / f"{file.stem}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Extracted {file.name} -> {output_file.name}")