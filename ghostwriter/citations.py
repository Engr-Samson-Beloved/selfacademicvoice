import re
from copy import deepcopy

from docx.oxml.ns import qn

_PLACEHOLDER_RE = re.compile(r"\{\{CIT_\d+\}\}")


def _is_citation_sdt(elem):
    if elem.tag != qn("w:sdt"):
        return False
    sdt_pr = elem.find(qn("w:sdtPr"))
    return sdt_pr is not None and sdt_pr.find(qn("w:citation")) is not None


def _run_text(elem):
    parts = []
    for node in elem.iter():
        if node.tag == qn("w:t"):
            parts.append(node.text or "")
        elif node.tag == qn("w:tab"):
            parts.append("\t")
        elif node.tag == qn("w:br"):
            parts.append("\n")
    return "".join(parts)


def _run_bold(r_elem):
    rpr = r_elem.find(qn("w:rPr"))
    if rpr is None:
        return False
    b = rpr.find(qn("w:b"))
    if b is None:
        return False
    return b.get(qn("w:val")) not in ("0", "false")


def extract(p, counter):
    """Return (visible_text, records, spans).

    visible_text: the paragraph text with each citation content control
        replaced by a ``{{CIT_n}}`` placeholder; normal runs and hyperlink
        runs are included, so this matches what the rewrite pipeline sees.
    records: list of ``(placeholder, deepcopy(sdt_element), display_text)``.
    spans: list of ``(start, end, bold)`` aligned to the newline-normalized
        ``visible_text``, used for per-sentence bold detection.
    """
    parts = []
    records = []
    spans = []
    pos = 0
    for child in p._p.iterchildren():
        tag = child.tag
        if tag == qn("w:sdt") and _is_citation_sdt(child):
            counter[0] += 1
            ph = "{{CIT_%d}}" % counter[0]
            parts.append(ph)
            spans.append((pos, pos + len(ph), False))
            pos += len(ph)
            display = "".join(t.text or "" for t in child.iter(qn("w:t")))
            records.append((ph, deepcopy(child), display))
        elif tag == qn("w:r"):
            t = _run_text(child)
            parts.append(t)
            tn = re.sub(r"\n", " ", t)
            spans.append((pos, pos + len(tn), _run_bold(child)))
            pos += len(tn)
        elif tag == qn("w:hyperlink"):
            t = "".join(_run_text(r) for r in child.findall(qn("w:r")))
            parts.append(t)
            tn = re.sub(r"\n", " ", t)
            spans.append(
                (pos, pos + len(tn), any(_run_bold(r) for r in child.findall(qn("w:r"))))
            )
            pos += len(tn)
    return "".join(parts), records, spans


def split_placeholders(text):
    """Split ``text`` into ``(chunk, placeholder_or_None)`` pairs."""
    out = []
    pos = 0
    for m in _PLACEHOLDER_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], None))
        out.append(("", m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], None))
    return out