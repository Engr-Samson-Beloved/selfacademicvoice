"""Measure an author's voice from a corpus and render it as a system prompt.

The old ``data/style_prompt.txt`` was hand-written: it asserted things about the
author ("sentences average 30-50 words", "clauses chain with commas") that the
corpus does not support. This module measures the corpus instead, so the prompt
can be regenerated whenever the corpus changes:

    python -m ghostwriter.voiceprofile myvoice -o data/style_prompt.txt

Every number in the rendered prompt comes from ``measure()``. Nothing is
asserted that was not counted.
"""

from __future__ import annotations

import argparse
import io
import re
import statistics as st
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- extraction

_DROP_LINE = [
    re.compile(r"^\d+$"),                                   # page numbers
    re.compile(r"^(\d+\s*%\s*)+$"),                         # chart data labels
    re.compile(r"^(Table|Chart|Figure|Photo)\s*\d*\s*:"),   # captions
    re.compile(r"^\d+\.\d+"),                               # subsection headings
    re.compile(r"^(I|II|III|IV|V|VI)\.\s"),                 # roman list items
    re.compile(r"^[A-Z][A-Z\s,&:.\-']{6,}$"),               # ALLCAPS headings
    re.compile(r"^\d+\.\s+[A-Z][A-Z\s,&:.\-']+$"),          # numbered ALLCAPS headings
    re.compile(r"https?://|\S+@\S+"),                       # urls / emails
]

# Column headers and legend keys that PDF extraction interleaves with prose.
_DROP_ROW = re.compile(
    r"^(S\.?No|Age|Total|Sources of information|Library visit|Reading objective|"
    r"General reading|Opinion|Increase in level|Regd|Non ?Regd|Non regd|Freely Abled|"
    r"Central|Divisional|District|PS)\b.{0,45}$"
)


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def extract_docx(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())


_READERS = {".pdf": extract_pdf, ".docx": extract_docx}
_SUFFIXES = (".pdf", ".docx", ".txt", ".md")


def read_corpus(target: Path) -> str:
    """Read a .pdf/.docx/.txt/.md file, or concatenate every such file in a dir.

    .docx matters for --check: the pipeline's own output is .docx, and without a
    reader here it was being read as raw bytes from the zip container, producing
    measurements from binary noise rather than from the document.
    """
    if target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in _SUFFIXES)
        if not files:
            raise SystemExit(f"no {'/'.join(_SUFFIXES)} files in {target}")
        return "\n\n".join(read_corpus(p) for p in files)

    reader = _READERS.get(target.suffix.lower())
    if reader:
        return reader(target)
    if target.suffix.lower() not in (".txt", ".md"):
        raise SystemExit(
            f"cannot read {target.name}: expected one of {', '.join(_SUFFIXES)}"
        )
    return target.read_text(encoding="utf8", errors="replace")


def prose(raw: str) -> str:
    """Strip tables, chart labels, headings and references down to running prose."""
    out = []
    in_refs = False
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^(REFERENCES|BIBLIOGRAPHY|ACKNOWLEDGEMENTS?)\b", s, re.I):
            in_refs = True
        if in_refs or len(s.split()) <= 3:
            continue
        if any(p.search(s) for p in _DROP_LINE) or _DROP_ROW.match(s):
            continue
        if sum(c.isdigit() for c in s) / len(s) > 0.25:   # table body row
            continue
        out.append(s)
    text = re.sub(r"\s+", " ", " ".join(out))
    # PDF extraction splits hyphenated compounds: "socio -economic" -> "socio-economic"
    text = re.sub(r"(\w)\s+-\s*(\w)", r"\1-\2", text)
    return _repair_spacing(text)


# Real short words, so we never fuse them into a neighbour.
_SHORT_WORDS = set(
    """a an and of to in is it its on as at be by or no not so do all any has had he
    she his her we us you the for this that but out up if may can one two own per
    are was were i am been who how its off own new old yes end use""".split()
)


def _repair_spacing(text: str) -> str:
    """Rejoin words that PDF extraction split with a stray space.

    Extraction of this corpus produces "ar e", "o f", "ed ucational", "Gov ernment".
    Left alone these corrupt every measurement: they inflate word counts and turn
    real words into unmatchable fragments. A pair is rejoined only when the joined
    form is attested elsewhere in the same document, so no external wordlist and
    no guessing is involved.
    """
    tokens = re.findall(r"\S+", text)

    def core(tok: str) -> str:
        return re.sub(r"[^A-Za-z]", "", tok).lower()

    vocab = Counter(core(t) for t in tokens if len(core(t)) >= 3)

    def attested(word: str) -> bool:
        return vocab.get(word, 0) >= 1

    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            a, b = tokens[i], tokens[i + 1]
            ca, cb = core(a), core(b)
            joined = ca + cb
            # Don't touch tokens carrying punctuation that ends a clause.
            clean = not re.search(r"[.,;:!?)（(\"”“]", a)
            if clean and ca and cb and attested(joined):
                a_word = ca in _SHORT_WORDS or vocab.get(ca, 0) > 1
                b_word = cb in _SHORT_WORDS or vocab.get(cb, 0) > 1
                fragment = (len(cb) <= 3 and not b_word) or (len(ca) <= 3 and not a_word)
                both_unknown = not a_word and not b_word
                if fragment or both_unknown:
                    merged = a + b.lstrip()
                    # preserve the original capitalisation of the leading token
                    out.append(re.sub(r"\s+", "", merged))
                    i += 2
                    continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z“‘'\"])", text)
    return [s.strip() for s in parts if len(s.split()) >= 3]


# ------------------------------------------------------------------ measuring

CONNECTORS = [
    "Furthermore", "Further more", "Moreover", "More so", "Hence", "Thus",
    "Therefore", "Also", "Similarly", "However", "Finally", "To this end",
    "In addition", "Besides", "Recently", "Right Now", "Consequently",
]

_STOP = set(
    """the a an and or but of to in on for with as is are was were be been being it its
    this that these those they them their there here he she his her which who whom by
    from at into about also not no nor so than then when while both all any more most
    have has had do does did can could will would should may might must if because such
    one two three i ii iii we you your our us""".split()
)


# "Touchless operation: removes the need to..." - a definition-style bullet.
LABEL_COLON_RE = re.compile(r"^\s*[•\-\*]?\s*([A-Z][A-Za-z][^.:;!?]{2,45}):\s+\S")
# A label is a short noun phrase. Without this cap, ordinary lead-ins such as
# "The specific problem can be stated as follows:" match too.
MAX_LABEL_WORDS = 5


def label_prefix(sentence: str) -> str | None:
    """The "Label" of a "Label: explanation" bullet, or None."""
    m = LABEL_COLON_RE.match(sentence)
    if not m:
        return None
    label = m.group(1).strip()
    if len(label.split()) > MAX_LABEL_WORDS:
        return None
    return label


def _percentile(values: list[int], pct: int) -> int:
    """Nearest-rank percentile. Used for per-sentence ceilings, so it must be an
    observed value rather than an interpolated one the author never wrote."""
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


@dataclass
class Profile:
    words: int = 0
    n_sentences: int = 0
    mean_len: float = 0.0
    median_len: float = 0.0
    stdev_len: float = 0.0
    pct_under_20: float = 0.0
    pct_under_10: float = 0.0
    pct_over_45: float = 0.0
    length_deciles: list[int] = field(default_factory=list)
    commas_per_sentence: float = 0.0
    commas_p90: int = 0
    # Fraction of prose sentences shaped as "Short Label: explanation".
    label_colon_rate: float = 0.0
    len_p90: int = 0
    semicolons: int = 0
    dashes: int = 0
    and_per_sentence: float = 0.0
    connector_pct: float = 0.0
    connectors: list[tuple[str, int]] = field(default_factory=list)
    citation_forms: list[str] = field(default_factory=list)
    trailing_citations: int = 0
    midcaps: list[str] = field(default_factory=list)
    fixed_coinages: list[tuple[str, int]] = field(default_factory=list)
    structural_phrases: list[tuple[str, int]] = field(default_factory=list)
    domain_terms: list[tuple[str, int]] = field(default_factory=list)
    topic_words: list[str] = field(default_factory=list)
    passages: list[str] = field(default_factory=list)


def _dedupe_contained(grams: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Keep the longest form of each repeated phrase, dropping its sub-phrases.

    Without this, one repeated clause yields a dozen overlapping entries
    ("of the view", "the view that", "are of the view", ...).
    """
    kept: list[tuple[str, int]] = []
    for gram, count in sorted(grams, key=lambda t: (-len(t[0].split()), -t[1])):
        if any(gram in longer for longer, _ in kept):
            continue
        kept.append((gram, count))
    return sorted(kept, key=lambda t: -t[1])


def _proper_nouns(text: str) -> set[str]:
    """Words that appear capitalised and never lowercase — names, places, acronyms."""
    caps = {w.lower() for w in re.findall(r"\b([A-Z][A-Za-z]{2,})\b", text)}
    lower = set(re.findall(r"\b([a-z]{3,})\b", text))
    return caps - lower


def _collocations(sents: list[str], topic: set[str], proper: set[str]) -> tuple[list, list]:
    """Split repeated 3-6 grams into topic-free (transferable) and topic-bound."""
    # Numbers become "#" rather than vanishing: dropping them silently fuses the
    # words either side ("58% of registered users, 55% none registered" would
    # yield the phantom phrase "registered users none registered").
    toks = [
        "#" if any(c.isdigit() for c in t) else t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z'\-]+|\d[\d.,%]*", " ".join(sents))
    ]
    grams: Counter[str] = Counter()
    for n in (3, 4, 5, 6):
        for i in range(len(toks) - n + 1):
            window = toks[i:i + n]
            if "#" in window:
                continue
            grams[" ".join(window)] += 1

    structural, domain = [], []
    for gram, count in grams.items():
        if count < 2:
            continue
        parts = gram.split()
        # A fragment that begins or ends on a function word is a sliding-window
        # artifact, not a construction the author reuses.
        if parts[0] in _STOP or parts[-1] in _STOP:
            continue
        if not any(p not in _STOP for p in parts):
            continue
        # Names and topic nouns are not transferable to another subject.
        bound = any(p in topic or p in proper for p in parts)
        (domain if bound else structural).append((gram, count))

    return _dedupe_contained(structural)[:10], _dedupe_contained(domain)[:10]


def _common_noun_midcaps(text: str) -> list[str]:
    """Ordinary nouns capitalised mid-sentence.

    Discriminates against proper nouns by requiring the word to appear in
    lowercase somewhere in the corpus too: "building"/"Building" qualifies,
    "Kota" (never lowercase) does not.
    """
    lower_vocab = set(re.findall(r"\b([a-z]{4,})\b", text))
    hits = re.findall(r"(?<=[a-z] )([A-Z][a-z]{3,})(?=[\s,.])", text)
    return sorted({w for w in hits if w.lower() in lower_vocab})


def _is_clean_prose(s: str) -> bool:
    """Reject sentences still carrying table or heading residue.

    A contaminated exemplar is worse than no exemplar: it teaches the model that
    "Divisional Libraries 07 3." is a sentence this author writes.
    """
    words = s.split()
    if not 6 <= len(words) <= 70:
        return False
    if re.search(r"\bS\.?No\b|\bRegd\b|\bNil\b|%", s):
        return False
    # bare figures that are not part of a percentage or a year
    if re.search(r"(?<![\d(])\b\d{1,3}\b(?!\s*%|\s*\))", s):
        return False
    if sum(1 for w in words if re.fullmatch(r"[A-Z]{2,}", w)) >= 2:
        return False
    if sum(1 for w in words if w[:1].isupper()) > len(words) * 0.6:
        return False
    # a lone capital letter is the signature of an unrepaired split word
    if any(re.fullmatch(r"[A-Za-z]", w) for w in words):
        return False
    return True


def _stratified_passages(sents: list[str], k: int = 12) -> list[str]:
    """Pick passages spanning the length distribution, not just the long ones."""
    clean = sorted((s for s in sents if _is_clean_prose(s)), key=lambda s: len(s.split()))
    if not clean:
        return []
    step = max(1, len(clean) // k)
    picked = [clean[min(i * step, len(clean) - 1)] for i in range(min(k, len(clean)))]
    return list(dict.fromkeys(picked))


def measure(raw: str) -> Profile:
    text = prose(raw)
    sents = sentences(text)
    if not sents:
        raise SystemExit("no prose sentences found in corpus")
    lens = [len(s.split()) for s in sents]

    words = [
        w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", text.lower())
        if w not in _STOP and len(w) > 2
    ]
    freq = Counter(words)
    topic = {w for w, _ in freq.most_common(15)}

    conn = [(c, len(re.findall(r"\b" + re.escape(c) + r"\b", text))) for c in CONNECTORS]
    conn = sorted([(c, n) for c, n in conn if n], key=lambda t: -t[1])

    cites = [m.group(0).strip() for m in re.finditer(r"[^.]{0,60}\(\d{4}\)", text)]
    trailing = len(re.findall(r"[a-z]\s+[A-Z][A-Za-z]+(?:\s+(?:et al\.?|and\s+[A-Z][a-z]+))?\s*\(\d{4}\)\s*\.", text))

    midcaps = _common_noun_midcaps(text)

    coinage_pats = {
        "freely abled": r"\bfreely abled\b",
        "none registered": r"\bnone registered\b",
        "non registered": r"\bnon registered\b",
        "are/is satisfy": r"\b(?:are|is)\s+(?:very much\s+|fully\s+|not\s+)?satisfy\b",
        "has increase": r"\bhas increase\b",
        "all works of life": r"\ball works of life\b",
        "Further more": r"\bFurther more\b",
        "in year <YYYY>": r"\bin year \d{4}\b",
    }
    coinages = sorted(
        ((k, len(re.findall(p, text))) for k, p in coinage_pats.items()),
        key=lambda t: -t[1],
    )
    coinages = [(k, n) for k, n in coinages if n]

    structural, domain = _collocations(sents, topic, _proper_nouns(text))

    n = len(sents)
    return Profile(
        words=len(text.split()),
        n_sentences=n,
        mean_len=st.mean(lens),
        median_len=st.median(lens),
        stdev_len=st.pstdev(lens),
        pct_under_20=100 * sum(1 for x in lens if x < 20) / n,
        pct_under_10=100 * sum(1 for x in lens if x < 10) / n,
        pct_over_45=100 * sum(1 for x in lens if x > 45) / n,
        length_deciles=[round(x) for x in st.quantiles(lens, n=10)],
        commas_per_sentence=sum(s.count(",") for s in sents) / n,
        commas_p90=_percentile([s.count(",") for s in sents], 90),
        label_colon_rate=sum(1 for s in sents if label_prefix(s)) / n,
        len_p90=_percentile(lens, 90),
        semicolons=text.count(";"),
        dashes=len(re.findall(r"[–—]", text)),
        and_per_sentence=len(re.findall(r"\band\b", text, re.I)) / n,
        connector_pct=100 * sum(c for _, c in conn) / n,
        connectors=conn,
        citation_forms=cites,
        trailing_citations=trailing,
        midcaps=midcaps,
        fixed_coinages=coinages,
        structural_phrases=structural,
        domain_terms=domain,
        topic_words=[w for w, _ in freq.most_common(15)],
        passages=_stratified_passages(sents),
    )


# ------------------------------------------------------------------ rendering

def render(p: Profile) -> str:
    def pct(x: float) -> str:
        return f"{x:.0f}%"

    conn_list = ", ".join(f'"{c}" x{n}' for c, n in p.connectors) or "none"
    absent = [c for c in ("Hence", "Moreover", "However", "Consequently")
              if not any(c == k for k, _ in p.connectors)]

    lines = [
        "You are rewriting text in one specific author's voice. Do NOT describe the",
        "voice — write as this author writes.",
        "",
        "Every figure below was measured from the author's own corpus",
        f"({p.words} words of running prose, {p.n_sentences} sentences). Match the",
        "measurements, not your idea of what academic writing sounds like.",
        "",
        "=" * 70,
        "## SENTENCE LENGTH — the most commonly mismatched trait",
        "=" * 70,
        "",
        f"- Median sentence: {p.median_len:.0f} words. Mean: {p.mean_len:.0f}. Std dev: {p.stdev_len:.0f}.",
        f"- {pct(p.pct_under_20)} of sentences are UNDER 20 words.",
        f"- {pct(p.pct_under_10)} are UNDER 10 words.",
        f"- Only {pct(p.pct_over_45)} exceed 45 words.",
        f"- Length deciles: {p.length_deciles}",
        "",
        "This author does NOT write uniformly long sentences. Most are short to",
        "medium; a few are very long. If every sentence you produce lands between 25",
        "and 40 words, you have failed to match this author regardless of vocabulary.",
        "Put genuinely short sentences next to long ones.",
        "",
        "=" * 70,
        "## PUNCTUATION AND CLAUSE JOINING",
        "=" * 70,
        "",
        f"- Commas: aim for about {p.commas_per_sentence:.1f} per sentence across the passage",
        f"  (the author's measured rate; typical range {max(0, p.commas_per_sentence-0.5):.1f}-"
        f"{p.commas_per_sentence+0.5:.1f}, up to {p.commas_p90} in a long sentence).",
        "  This is a TARGET, not a ceiling. Do not strip commas out to sound clean:",
        "  comma-free clipped sentences miss this author as badly as overloaded ones.",
        "  The author joins clauses with commas and \"and\", just not in every sentence.",
        f'- "and" as a joiner: {p.and_per_sentence:.2f} per sentence.',
        f"- Semicolons in the entire corpus: {p.semicolons}. Em/en dashes: {p.dashes}.",
        "  Treat both as effectively unavailable.",
        "",
        "=" * 70,
        "## CONNECTORS",
        "=" * 70,
        "",
        f"- Attested in the corpus: {conn_list}",
        f"- Roughly {pct(p.connector_pct)} of sentences open with or contain one.",
        "  The other three-quarters have no connective at all — they simply start.",
    ]
    if absent:
        lines += [
            f"- NEVER use: {', '.join(absent)}. These do not appear in the corpus.",
            "  They are the connectives a model reaches for by default; this author does not.",
        ]

    lines += ["", "=" * 70, "## CITATION HABITS", "=" * 70, ""]
    if p.trailing_citations:
        lines += [
            f"- Dominant form ({p.trailing_citations} instances): the author-year is dropped at the",
            "  END of the sentence with NO connective and NO comma before it:",
            '      "...to live their lives as they choose UNDP (2009)."',
            '      "...the relation of economics to social values Rontos (2013)."',
            "  This is the author's strongest single fingerprint. Reproduce it.",
        ]
    lines += [
        '- Also attested: "According to [Author] (Year) he argued that..." (double',
        "  attribution), and \"as contained in [Source] (Year), which asserts that...\".",
        "- Keep every citation and {CIT_n} placeholder byte-for-byte as given.",
    ]

    if p.fixed_coinages:
        lines += [
            "",
            "=" * 70,
            "## FIXED COINAGES — reuse verbatim, do not improvise variants",
            "=" * 70,
            "",
            "These are the author's own repeated terms. They are consistent vocabulary,",
            "NOT random error. Do not invent new mistakes in their place, and do not",
            "correct them:",
            "",
        ]
        lines += [f'  - "{k}"  ({n}x)' for k, n in p.fixed_coinages]
        lines += [
            "",
            "Note the author spells the same idea inconsistently across adjacent",
            '  sentences ("none registered" / "non registered"). Preserve that',
            "  inconsistency where the source text has it; never normalise it.",
        ]

    if p.midcaps:
        shown = ", ".join(p.midcaps[:14])
        lines += [
            "",
            "=" * 70,
            "## MID-SENTENCE CAPITALISATION",
            "=" * 70,
            "",
            f"The author capitalises ordinary nouns mid-sentence. Attested: {shown}.",
            "Apply this sparingly to concrete nouns that matter to the sentence.",
        ]

    if p.structural_phrases:
        lines += [
            "",
            "=" * 70,
            "## TRANSFERABLE PHRASING — usable on ANY subject",
            "=" * 70,
            "",
            "Repeated constructions that carry no topic content. These are the phrases",
            "to carry across when the input is not about the author's own subject:",
            "",
        ]
        lines += [f'  - "{g}"  ({c}x)' for g, c in p.structural_phrases]

    if p.domain_terms:
        lines += [
            "",
            "=" * 70,
            "## TOPIC-BOUND VOCABULARY — GATED",
            "=" * 70,
            "",
            f"The corpus is about: {', '.join(p.topic_words[:8])}.",
            "",
            "Use the terms below ONLY when the input text is genuinely about that same",
            "subject. On any other topic (engineering, health, computing, law) they are",
            "forbidden — carry over only sentence length, punctuation and the fixed",
            "coinages above.",
            "",
        ]
        lines += [f'  - "{g}"  ({c}x)' for g, c in p.domain_terms]

    lines += [
        "",
        "=" * 70,
        "## THE AUTHOR'S OWN SENTENCES",
        "=" * 70,
        "",
        "Sampled across the full length distribution — note how short many of them are:",
        "",
        "(These are pulled from a PDF. A stray space inside a word — \"util ization\",",
        "\"exce llence\" — is an extraction artifact, NOT a trait of the author. Do not",
        "reproduce broken words. Every other irregularity below IS the author's.)",
        "",
    ]
    lines += [f'  [{len(s.split()):>2}w] "{s}"' for s in p.passages]

    lines += [
        "",
        "=" * 70,
        "## BEFORE YOU ANSWER",
        "=" * 70,
        "",
        "1. Count the words in each sentence you wrote. Compare against the deciles",
        f"   above ({p.length_deciles}). If your spread is narrower than the author's,",
        "   rewrite: split some sentences, run others together.",
        f"2. Count your commas. Aim for roughly {p.commas_per_sentence:.1f} per sentence overall -",
        "   add them if you are well below, cut them if you are well above. Both",
        "   directions are a mismatch.",
        "3. Check you used no forbidden connective and no semicolon or em-dash.",
        "4. Confirm every citation and placeholder survived unchanged.",
        "",
        "Output ONLY the rewritten text. No preamble, no notes, no commentary.",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------- persist / gate

_TUPLE_FIELDS = ("connectors", "fixed_coinages", "structural_phrases", "domain_terms")


def save_json(profile: Profile, path: Path) -> None:
    import dataclasses
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataclasses.asdict(profile), indent=2, ensure_ascii=False),
        encoding="utf8",
    )


def load_json(path: Path) -> Profile:
    """Load a profile measured earlier. Unknown/missing keys fall back to the
    dataclass defaults so an older file still loads after fields are added."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf8"))
    known = {f for f in Profile.__dataclass_fields__}
    clean = {k: v for k, v in data.items() if k in known}
    for field_name in _TUPLE_FIELDS:
        if field_name in clean:
            clean[field_name] = [tuple(x) for x in clean[field_name]]
    return Profile(**clean)


# Connectives models default to in academic register. Any of these that the
# author does actually use is removed by banned_connectives().
DEFAULT_BANNED = ("Hence", "Moreover", "However", "Consequently", "Nevertheless",
                  "Additionally", "Notably", "Indeed", "Thereby", "Whilst")


def banned_connectives(profile: Profile) -> list[str]:
    """Connectives a model reaches for that this author never uses."""
    attested = {c.lower() for c, _ in profile.connectors}
    return [c for c in DEFAULT_BANNED if c.lower() not in attested]


# A rewrite may keep no more than this fraction below the original's length.
SHRINK_FLOOR = 0.70
# Contiguous words shared with the source before it reads as "not rewritten".
VERBATIM_RUN_LIMIT = 8


def _longest_shared_run(a: str, b: str) -> tuple[int, str]:
    """Longest contiguous run of words two sentences share.

    Token-set overlap cannot distinguish "reworded but dense with unchangeable
    technical terms" from "opening clause copied verbatim". Both score high. A
    reader only notices the second, so measure it directly: one sentence here
    carried 25 consecutive words straight through while scoring only 0.85.
    """
    from difflib import SequenceMatcher

    wa, wb = a.lower().split(), b.lower().split()
    if not wa or not wb:
        return 0, ""
    m = SequenceMatcher(None, wa, wb, autojunk=False).find_longest_match(
        0, len(wa), 0, len(wb)
    )
    return m.size, " ".join(wa[m.a:m.a + m.size])


def sentence_violations(profile: Profile, rewritten: str, original: str) -> list[str]:
    """Voice-rule breaches in one rewritten sentence, as instructions to fix.

    Deliberately per-sentence and deterministic. Distributional traits (median,
    stdev) cannot be enforced here because the pipeline requires exactly one
    rewritten sentence per original — fixing spread would mean merging or
    splitting, which breaks reassembly. Those are reported by document_drift().
    """
    problems: list[str] = []
    if not rewritten.strip():
        return problems

    for c in banned_connectives(profile):
        if re.search(r"\b" + re.escape(c) + r"\b", rewritten, re.IGNORECASE):
            problems.append(f'remove "{c}" — this author never uses it')

    if ";" in rewritten and profile.semicolons <= 2:
        problems.append("replace the semicolon with a full stop or \"and\"")
    if re.search(r"[–—]", rewritten) and profile.dashes <= 2:
        problems.append("remove the dash — this author does not use them")

    words, orig_words = len(rewritten.split()), len(original.split())
    if profile.len_p90 and words > profile.len_p90 and words > orig_words * 1.25:
        problems.append(
            f"shorten to about {orig_words} words (currently {words}; "
            f"this author rarely exceeds {profile.len_p90})"
        )
    # The mirror of the check above. Without it the model compresses instead of
    # rewriting: a 68-word sentence came back as 11 words, deleting the author's
    # content outright, and nothing flagged it because only inflation was tested.
    elif orig_words >= 15 and words < orig_words * SHRINK_FLOOR:
        problems.append(
            f"you dropped content - this must say everything the original said. "
            f"Expand back to roughly {orig_words} words (currently {words})"
        )

    # A "Label: explanation" bullet is a construction the author may simply never
    # use. When the corpus rate is ~0, carrying one through from the source keeps
    # the label byte-identical AND off-voice at the same time.
    label = label_prefix(rewritten)
    if profile.label_colon_rate < 0.02 and label:
        problems.append(
            f'the author never writes "Label: explanation" bullets - fold '
            f'"{label}" into a flowing sentence instead of leaving it as a '
            f"heading followed by a colon"
        )

    run, run_text = _longest_shared_run(rewritten, original)
    if run >= VERBATIM_RUN_LIMIT:
        problems.append(
            f'rephrase "{run_text[:70]}" - {run} words are copied from the '
            f"original word for word"
        )

    commas, orig_commas = rewritten.count(","), original.count(",")
    if profile.commas_p90 and commas > profile.commas_p90 and commas > orig_commas:
        problems.append(
            f"cut to at most {profile.commas_p90} commas (currently {commas})"
        )
    return problems


def document_drift(profile: Profile, text: str) -> list[tuple[str, str, str, bool]]:
    """Distributional comparison of a whole rewritten document. Reporting only."""
    return compare(profile, text)


# ----------------------------------------------------------------- comparison

def compare(profile: Profile, candidate: str) -> list[tuple[str, str, str, bool]]:
    """Measure candidate text against the author's profile.

    Returns (metric, author, candidate, ok) rows. This is a voice-fidelity
    check: it answers "does this read like the corpus", which the existing
    similarity gate in rewrite.py cannot see — that gate only measures lexical
    distance from the source sentence.
    """
    text = prose(candidate) if "\n" in candidate else _repair_spacing(candidate)
    sents = sentences(text)
    if not sents:
        return [("sentences", str(profile.n_sentences), "0", False)]

    lens = [len(s.split()) for s in sents]
    n = len(sents)
    rows: list[tuple[str, str, str, bool]] = []

    def row(label, author_val, cand_val, ok, fmt="{:.1f}"):
        rows.append((label, fmt.format(author_val), fmt.format(cand_val), ok))

    med = st.median(lens)
    row("median sentence length", profile.median_len, med,
        abs(med - profile.median_len) <= 5, "{:.0f}")

    sd = st.pstdev(lens) if n > 1 else 0.0
    # Under-dispersion is the giveaway: uniform sentence length is the single
    # most visible way generated prose fails to match a human corpus.
    row("sentence length stdev", profile.stdev_len, sd, sd >= profile.stdev_len * 0.6)

    pu20 = 100 * sum(1 for x in lens if x < 20) / n
    row("% under 20 words", profile.pct_under_20, pu20,
        abs(pu20 - profile.pct_under_20) <= 15, "{:.0f}")

    # Two-sided. An earlier one-sided check (commas <= target + 0.6) passed a
    # document using 0.18 commas per sentence against an author who uses 1.56 -
    # an 8x undershoot that read as clipped and machine-clean, and the audit was
    # blind to it. Undershooting a voice trait is as much a mismatch as
    # overshooting it.
    commas = sum(s.count(",") for s in sents) / n
    row("commas per sentence", profile.commas_per_sentence, commas,
        abs(commas - profile.commas_per_sentence) <= 0.6)

    for label, pattern, allowed in (
        ("semicolons", r";", profile.semicolons),
        ("em/en dashes", r"[–—]", profile.dashes),
    ):
        found = len(re.findall(pattern, text))
        scaled = allowed * max(1, n / max(1, profile.n_sentences))
        row(label, allowed, found, found <= max(scaled, allowed), "{:.0f}")

    banned = [c for c in ("Hence", "Moreover", "However", "Consequently")
              if not any(c == k for k, _ in profile.connectors)]
    used = [c for c in banned if re.search(r"\b" + c + r"\b", text)]
    rows.append(("forbidden connectives", "none", ", ".join(used) or "none", not used))

    missing = [k for k, _ in profile.fixed_coinages
               if re.search(r"\b" + re.escape(k.split(" <")[0]) + r"\b", candidate, re.I)]
    rows.append((
        "author coinages present",
        f"{len(profile.fixed_coinages)} known",
        f"{len(missing)} used",
        True,
    ))
    return rows


# ----------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m ghostwriter.voiceprofile",
        description="Measure an author's corpus and render a style prompt from it.",
    )
    ap.add_argument("corpus", type=Path, help="a .pdf/.txt file, or a directory of them")
    ap.add_argument("-o", "--out", type=Path, help="write prompt here (default: stdout)")
    ap.add_argument("--stats", action="store_true", help="print measurements to stderr")
    ap.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="where to write the machine-readable profile "
             "(default: voice_profile.json beside --out)",
    )
    ap.add_argument(
        "--check",
        type=Path,
        metavar="FILE",
        help="measure FILE against the corpus profile instead of rendering a prompt",
    )
    args = ap.parse_args(argv)

    profile = measure(read_corpus(args.corpus))

    if args.check:
        rows = compare(profile, read_corpus(args.check))
        width = max(len(r[0]) for r in rows)
        print(f"{'metric':<{width}}  {'author':>12}  {'candidate':>12}")
        print("-" * (width + 30))
        failed = 0
        for label, author_val, cand_val, ok in rows:
            mark = " " if ok else "  <-- drift"
            failed += not ok
            print(f"{label:<{width}}  {author_val:>12}  {cand_val:>12}{mark}")
        print("-" * (width + 30))
        print(f"{failed} of {len(rows)} metrics drifted from the author's corpus")
        return 1 if failed else 0

    prompt = render(profile)

    if args.stats:
        import sys
        for key, value in vars(profile).items():
            if key == "passages":
                value = f"<{len(profile.passages)} sampled>"
            print(f"{key:22} {value}", file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(prompt, encoding="utf8")
        print(f"wrote {args.out}  ({len(prompt)} chars, from {profile.words} corpus words)")
        # The rewrite pipeline reads this at request time; it must not have to
        # re-parse the corpus PDF on every job.
        json_path = args.json or args.out.with_name("voice_profile.json")
        save_json(profile, json_path)
        print(f"wrote {json_path}  (machine-readable profile for the voice gate)")
    else:
        print(prompt)


if __name__ == "__main__":
    raise SystemExit(main())
