from pathlib import Path
import json
import re

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/dataset")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def split_sections(text):
    """
    Splits report into sections based on headings.
    """

    heading_pattern = re.compile(
        r'^\d+\.\d+\s+.*$|^[A-Z][A-Z\s]{3,}$',
        re.MULTILINE
    )

    matches = list(heading_pattern.finditer(text))

    sections = []

    for i, match in enumerate(matches):

        heading = match.group().strip()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        body = text[start:end].strip()

        if body:
            sections.append(
                {
                    "section": heading,
                    "content": body
                }
            )

    return sections


def split_paragraphs(sections):

    records = []

    for section in sections:

        paragraphs = section["content"].split("\n\n")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if len(paragraph) < 40:
                continue

            records.append(
                {
                    "section": section["section"],
                    "paragraph": paragraph
                }
            )

    return records


def main():

    for file in INPUT_DIR.glob("*.txt"):

        print(f"Processing {file.name}")

        text = read_file(file)

        sections = split_sections(text)

        dataset = split_paragraphs(sections)

        output = OUTPUT_DIR / f"{file.stem}.json"

        save_json(output, dataset)

        print(f"Saved {len(dataset)} paragraphs.")


if __name__ == "__main__":
    main()