from pathlib import Path
import re

# Directories
INPUT_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_file(file_path: Path) -> str:
    """Read text from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_file(file_path: Path, text: str):
    """Save text to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_whitespace(text: str) -> str:
    """Clean unnecessary whitespace."""
    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3 or more newlines with just 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_duplicate_lines(text: str) -> str:
    """
    Removes consecutive duplicate lines.

    Example:
    INTRODUCTION
    INTRODUCTION

    becomes

    INTRODUCTION
    """

    lines = text.splitlines()

    cleaned = []
    previous = None

    for line in lines:
        line = line.strip()

        if line == previous:
            continue

        cleaned.append(line)
        previous = line

    return "\n".join(cleaned)


def remove_references(text: str) -> str:
    """
    Removes everything after the References section.
    """

    patterns = [
        r"\nREFERENCES\b",
        r"\nReferences\b",
        r"\nREFERENCE\b",
        r"\nReference\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            text = text[:match.start()]
            break

    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = remove_duplicate_lines(text)
    text = remove_references(text)

    return text


def main():

    files = list(INPUT_DIR.glob("*.txt"))

    if not files:
        print("No text files found.")
        return

    for file in files:

        print(f"Cleaning {file.name}...")

        text = read_file(file)

        cleaned = clean_text(text)

        output_file = OUTPUT_DIR / file.name

        save_file(output_file, cleaned)

        print(f"Saved -> {output_file}")


if __name__ == "__main__":
    main()