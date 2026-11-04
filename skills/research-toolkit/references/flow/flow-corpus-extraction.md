---
name: flow-corpus-extraction
description: Re-extract or extend a PDF document corpus with PyMuPDF - inclusive page-range Markdown rendering, monospace code detection by per-line font mix (0.6 threshold), byte-exact provenance headers, toc/chapters/whole split modes, the printed-folio to PDF-page offset-by-mode trick, manifest regeneration and slug-based unit-to-skill routing. Use when asked to extract text or code from a PDF, slice a PDF by page range or chapter, rebuild an extraction manifest, map a book's printed folios to PDF page numbers, split a bookmarkless PDF by chapter, or route corpus units to downstream skills.
---

# PDF Corpus Extraction (Python)

## Purpose
Convert one PDF into page-referenced, code-aware Markdown units plus a JSON routing manifest: a pure
function of `(pdf, split policy, page range)`. No analysis, no OCR.

## Inputs
- `pdf`: any PDF whose text layer PyMuPDF reads; `OUT` (output dir) and `SOURCE_KEY` (routing prefix) are arguments, never fixed folders.
- `MODE`: `toc` (needs bookmarks) | `chapters` (offset trick, bookmarks as fallback) | `whole` (one unit).
- `LEVELS` bookmark levels, `SKIP_RE` front/back-matter titles, `MIN_CHARS = 300` boilerplate drop.
- `THRESHOLD = 0.6` per-line monospace share, `FRONT_PAGES = 2` Contents pages, `BODY_START = 3`.

## Algorithms (all pure numpy/scipy)
Diagnose (bookmarks, mono share, footer folios), pick the mode, render, stamp provenance, regenerate the manifest; each block builds its own throwaway PDF, so every printed number is self-verifying.

### Font mix, fence state machine, provenance header
```python
import re, pymupdf
MONO = re.compile(r"(courier|mono|consolas|inconsolata|menlo|source code)", re.I); THRESHOLD = 0.6
def page_lines(page):                        # -> [(text, is_code, mono_share)] in reading order
    rows = []
    for blk in page.get_text("dict")["blocks"]:
        for line in (blk.get("lines") or []):
            total = sum(len(s["text"].strip()) for s in line["spans"])
            mono = sum(len(s["text"].strip()) for s in line["spans"] if MONO.search(s["font"]))
            if total:
                rows.append(("".join(s["text"] for s in line["spans"]).rstrip(), mono / total > THRESHOLD, mono / total))
    return rows
def fence(buf, code): return ("```python\n" + "\n".join(buf) + "\n```") if code else "\n".join(buf)
def render_pages(doc, start, end):           # inclusive, 1-indexed PDF page range
    out = []
    for pno in range(max(1, start), min(end, doc.page_count) + 1):    # out-of-range pages never run
        if not (lines := page_lines(doc[pno - 1])):                   # an empty page adds nothing
            continue
        out.append(f"<!-- source pages {pno} -->")
        buf, code = [lines[0][0]], lines[0][1]
        for text, is_code, _share in lines[1:]:
            if is_code != code:              # flag vs flag; the legacy bug compared to buf (a list)
                out.append(fence(buf, code)); buf, code = [], is_code
            buf.append(text)
        out.append(fence(buf, code))
    return "\n\n".join(out).strip()          # after FIXUPS + NFKC + blank-run collapse
def header(heading, label, start, end): return f"# {heading}\n\n> Source: {label}\n> Pages: {start}-{end} (PDF page numbering, 1-indexed)\n\n"
doc = pymupdf.open(); [doc.new_page() for _ in range(2)]
doc[0].insert_text((72, 90), "Lists are mutable and variable length (McKinney p69).", fontname="tiro", fontsize=11)
doc[0].insert_text((72, 120), "b_list[1] = 'peekaboo'", fontname="cour", fontsize=11)
gap = pymupdf.Font("helv").text_length("Count = ", 11)
doc[1].insert_text((72, 90), "Count = ", fontname="helv", fontsize=11)   # serif + Courier, one line
doc[1].insert_text((72 + gap, 90), "len(data)", fontname="cour", fontsize=11)
body = header("Data Structures", "demo.pdf", 1, 2) + render_pages(doc, 1, 2)
mixed, leak = page_lines(doc[1])[0], "\n```python\nLists" in body
cite = re.search(r"^> Pages:\s*(\d+)-(\d+)", body, re.M).group(0)
print(f"RESULT threshold={THRESHOLD} mixed_share={mixed[2]:.4f} is_code={mixed[1]} prose_fenced={leak} cite={cite}")
print(f"RESULT code_blocks={body.count('```python' + chr(10))} chars={len(body)}")
```
Code is declared by per-line font mix, never by indentation: the Courier span on the mixed line scores
9/16 = 0.5625 and stays prose, the serif-versus-monospace split of the reference unit (McKinney p65-100).
Consecutive code lines share one fence and the state flips on the flag alone - testing the flag against the
buffer *list* was the legacy defect that fenced every prose line of that chapter.

### Split modes, range arithmetic, manifest and routing
```python
import re, pymupdf
SKIP_RE = r"^(Cover|Copyright|Table of Contents|Preface|References|Index)$"
def toc_sections(doc, levels, skip_re=SKIP_RE):
    entries = [{"title": t.strip(), "page": p} for lvl, t, p in doc.get_toc()
               if lvl in levels and t.strip() and not re.match(skip_re, t.strip())]
    return [{"title": e["title"], "start": e["page"], "end": max(e["page"], min(
        entries[i + 1]["page"] - 1 if i + 1 < len(entries) else doc.page_count, doc.page_count))}
        for i, e in enumerate(entries)]
def slugify(title, n=58):                    # file stem AND the unit -> skill routing token
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:n].rstrip("-") or "section"
doc = pymupdf.open(); [doc.new_page() for _ in range(101)]
doc[64].insert_textbox(pymupdf.Rect(72, 100, 540, 300), "Built-in data structures, functions and files: tuples, lists, dicts, sets. " * 6, fontname="tiro", fontsize=11)
doc[64].insert_text((287, 730), "47", fontname="tiro", fontsize=10); doc[100].insert_text((72, 72), "4. NumPy: Array Object", fontname="helv", fontsize=14)
doc.set_toc([[1, "Cover", 1], [1, "3. Built-In Data Structures, Functions, and Files", 65],
             [1, "4. NumPy: Array Object", 101], [1, "Index", 101]])
secs = toc_sections(doc, {1})
raw = lambda s: sum(len(doc[p - 1].get_text()) for p in range(s["start"], s["end"] + 1))
kept = [s for s in secs if raw(s) >= 300]                                   # MIN_CHARS drops toc stubs only
units = [{"unit": f"mckinney-pda/{slugify(s['title'])}", "skill_key": slugify(s["title"]),
          "heading": s["title"], "file": f"units/{slugify(s['title'])}.md", "pages": [s["start"], s["end"]],
          "chars": raw(s), "code_blocks": 0} for s in kept]
manifest = {"other_book": [{"unit": "other_book/x"}]}; manifest["mckinney-pda"] = units  # only its own key
folio = int(re.findall(r"(?m)^\s*(\d{1,4})\s*$", doc[64].get_text(), re.M)[0])
print(f"RESULT bookmarks={len(doc.get_toc())} after_skip={len(secs)} kept={len(kept)} whole_units=1 pages=1-{doc.page_count}")
print(f"RESULT ch3_pages={kept[0]['start']}-{kept[0]['end']} folio={folio} offset={65 - folio}")
print(f"RESULT routing={units[0]['skill_key']} manifest_sources={sorted(manifest)}")
```
A unit runs to `next.start - 1`, clamped by `max(start, min(end, page_count))`, so the level-1 bookmark at p65
against the next at p101 renders exactly **65-100** and the folio offset is **65 - 47 = 18** (McKinney p65-100,
McKinney p65). `whole` is one unit over 1..page_count; `MIN_CHARS` filters only `toc` stubs.

### Printed folio -> PDF page: the offset trick (`chapters` mode)
```python
import re, pymupdf
from collections import Counter
CH = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][^\n]{3,70}?)\s*(?:[.\s]{3,})(\d{1,3})\s*$", re.M)
REC = re.compile(r"^\s*(\d{1,2})\.(\d{1,3})\s+([^\n.]{5,90}?)\s*(?:[.\s]{3,})(\d{1,3})\s*$", re.M)
BODY = re.compile(r"(?m)^\s*(\d{1,2})\.(\d{1,3})\s+([A-Z][^\n]{3,90})\s*$")
doc = pymupdf.open(); [doc.new_page() for _ in range(6)]
for i, ln in enumerate(["Contents", "1. Alpha Recipes ...... 1", "1.1 Alpha widget ..... 1",
                        "1.2 Alpha gadget ..... 2", "2. Beta Recipes ...... 3", "2.1 Beta widget ...... 3"]):
    doc[1].insert_text((72, 80 + 22 * i), ln, fontname="tiro", fontsize=11)
for i, (title, folio) in enumerate([("1.1 Alpha widget", 1), ("1.2 Alpha gadget", 2), ("2.1 Beta widget", 3)]):
    doc[3 + i].insert_text((72, 80), title, fontname="helv", fontsize=13); doc[3 + i].insert_text((300, 730), str(folio), fontname="tiro", fontsize=10)
doc.set_toc([])                                       # no bookmarks at all: force the offset trick
front = "".join(doc[i].get_text() for i in range(2))                      # FRONT_PAGES
recipes = {(int(c), int(r)): int(p) for c, r, _t, p in REC.findall(front)}
chapters = {int(c): (t.strip(), int(p)) for c, t, p in CH.findall(front)}
votes = Counter()
for pno in range(3, doc.page_count + 1):                                  # BODY_START, never Contents
    for c, r, _t in BODY.findall(doc[pno - 1].get_text()[:1500]):
        if (int(c), int(r)) in recipes: votes[pno - recipes[(int(c), int(r))]] += 1
offset, matches = (votes.most_common(1)[0][0], sum(votes.values())) if votes else (None, 0)
starts = sorted(((int(c), p + offset) for c, (_t, p) in chapters.items()), key=lambda kv: kv[1])
bounds = [p for _c, p in starts] + [doc.page_count + 1]
rng = [(c, s, max(s, min(bounds[i + 1] - 1, doc.page_count))) for i, (c, s) in enumerate(starts)]
print(f"RESULT recipes={len(recipes)} chapters={len(chapters)} offset={offset} matches={matches} votes={dict(votes)}")
print("RESULT ranges=" + ";".join(f"ch{c:02d} {s}-{e}" for c, s, e in rng) + f" bookmark_fallback={offset is None} contents_lookalikes={len(BODY.findall(front))}")
```
Unfoliated front matter makes `PDF page = printed folio + offset` one constant, recovered as the **mode** of
the votes so one OCR slip or running head cannot move it: three agreeing votes give `offset=3`, so `ch01` is
pages 4-5 and `ch02` page 6; zero votes falls back to `toc_sections` or prints a one-line reason.

## Workflow
1. Diagnose first: bookmarks decide `toc`, monospace share decides whether fences get found, footer integers give a quick median offset (median to diagnose, mode to extract). `toc` for a bookmarked book, `chapters` for a bookmarkless cookbook, `whole` for a short paper.
2. Open one unit, confirm only Courier-family lines sit inside fences, then route on `source_key/slug`.

## Output contract
- `OUT/units/<slug>.md`: header (`# heading`, blank, `> Source: <label>`, `> Pages: start-end (PDF page
  numbering, 1-indexed)`, blank) then the body with `<!-- source pages N -->` per rendered page.
- `OUT/manifest.json`: `{source_key: [{unit, skill_key, heading, file, pages, chars, code_blocks}]}`;
  `chars` excludes the header, `code_blocks` counts fences, regeneration replaces only its own key.

## Validity conditions
- Keep the header wording byte-for-byte: `^> Pages:\s*(\d+)-(\d+)` is the citation of record, and pages are
  1-indexed PDF indices, never printed folios - converting is exactly the offset trick (McKinney p65-100).
- Contents leaders need `[.\s]{3,}` dot fill and headings are scanned only from `BODY_START`: Contents
  lines match the same body regex (the 3 look-alikes printed above) and would vote for offset 0.
- No `LEVELS` match or every section under `MIN_CHARS` gives zero units; a bookmark overshooting
  `page_count` clamps instead of going negative; image-only scans give no `dict` lines, hence empty units.
- A book that restarts its folio series mis-slices its first and last chapter: check boundaries visually.

## Out of scope
Analysing the numbers inside a unit, loading tabular data files and OCR belong to sibling flow skills.

## Conventions
- `pymupdf`, `re`, `json`, `collections`, `unicodedata`, `pathlib`: font share and the mode are plain
  arithmetic, so the numpy/scipy stack named in the heading is unused; nothing installs packages.
- Normalise before NFKC with FIXUPS (hyphen variants to `-`, curly quotes to straight, ligatures to
  `fi/fl`, soft hyphen dropped) and skip falsy keys: an empty key scatters separators over the page.
- Report coverage as counts only, never beyond the pages rendered: "Extracted 1 unit (mode=toc), 244 chars,
  1 code block across pages 1-2; each carries `> Pages:` provenance" for the demo above (McKinney p65-100).
