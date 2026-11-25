import re
#import sys
from pathlib import Path

# Regular expressions for parsing out relevant text
START_RE = re.compile(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE | re.IGNORECASE)
END_RE = re.compile(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE | re.IGNORECASE)
BRACKET_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)
CONTENTS_RE = re.compile(r"^\s*(?:TABLE OF )?CONTENTS\s*\.?\s*$", re.MULTILINE | re.IGNORECASE)


def read_text(path):
    """Read a file, trying UTF-8 first and falling back to latin-1."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def strip_gutenberg(text):
    """Return the text between the START and END markers, or None if either is missing."""
    start = START_RE.search(text)
    end = END_RE.search(text)
    if not start or not end or end.start() < start.end():
        return None
    return text[start.end():end.start()].strip()


def strip_brackets(text):
    """Remove [bracketed] notes (including nested and multi-line ones) and tidy blank lines."""
    prev = None
    while prev != text:
        prev = text
        text = BRACKET_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_to_first_prose(text, min_words=12, long_words=25, search_limit=0.25,
                         keep_headings=True, max_heading_words=8, max_heading_lines=3):
    """
    Cut everything before the first prose-like line.

    A line counts as prose if it has at least `long_words` words, or if it and the
    next line both have at least `min_words` words (wrapped paragraphs).

    If keep_headings is True, walk back from that line over the preceding lines,
    keeping each one while it is non-blank, has at most `max_heading_words` words,
    and we have not yet gone back `max_heading_lines` lines.
    """
    lines = text.splitlines(keepends=True)
    limit = len(text) * search_limit
    offset = 0

    for i, line in enumerate(lines):
        if offset > limit:
            break
        n = len(line.split())
        next_n = len(lines[i + 1].split()) if i + 1 < len(lines) else 0

        if n >= long_words or (n >= min_words and next_n >= min_words):
            start = i
            if keep_headings:
                steps = 0
                while (start > 0
                       and steps < max_heading_lines
                       and lines[start - 1].strip()
                       and len(lines[start - 1].split()) <= max_heading_words):
                    start -= 1
                    steps += 1
            return "".join(lines[start:])
        offset += len(line)

    return text

def strip_front_matter(text, contents_limit=0.25, **kwargs):
    """
    If a CONTENTS heading appears near the start, look for the first prose only
    after it (removing any title page or preface before it). Otherwise, look for
    the first prose from the top of the file. Extra keyword arguments are passed
    to strip_to_first_prose.
    """
    match = CONTENTS_RE.search(text)
    if match and match.start() <= len(text) * contents_limit:
        return strip_to_first_prose(text[match.end():], **kwargs)
    return strip_to_first_prose(text, **kwargs)

def combine_subdirectory(subdir, output_dir):
    parts = []
    for path in sorted(subdir.glob("*.txt")):
        text = read_text(path)
        body = strip_gutenberg(text)
        if body is None:
            print(f"  Warning: markers not found in {path.name}; keeping the whole file")
            body = text.strip()

        body = strip_brackets(body)         # first, so notes at the start aren't mistaken for prose
        before = len(body)
        body = strip_front_matter(body)
        if len(body) < before * 0.9:
            print(f"  Note: front matter removal cut {1 - len(body)/before:.0%} of {path.name}")

        parts.append(body)

    if not parts:
        print(f"Skipping {subdir.name}: no .txt files found")
        return

    output_path = output_dir / f"{subdir.name}.txt"
    output_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"{subdir.name}: combined {len(parts)} files -> {output_path}")


def combine_all(root, output_dir=None):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if subdir == output_dir:
            continue
        combine_subdirectory(subdir, output_dir)


if __name__ == "__main__":
    # usage: python gutenberg.py /path/to/root [/path/to/output_dir]
    input_dir = '../story/source'
    output_dir = '../story/historic'
    #combine_all(*sys.argv[1:3])
    combine_all(input_dir, output_dir)