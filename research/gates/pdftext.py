"""Minimal PDF text extractor: inflate FlateDecode streams, pull Tj/TJ operands."""
import re, sys, zlib

def extract(path):
    raw = open(path, "rb").read()
    chunks = []
    for m in re.finditer(rb"stream\r?\n", raw):
        s = m.end()
        e = raw.find(b"endstream", s)
        if e < 0:
            continue
        blob = raw[s:e]
        try:
            txt = zlib.decompress(blob)
        except Exception:
            try:
                txt = zlib.decompressobj().decompress(blob)
            except Exception:
                continue
        chunks.append(txt)
    out = []
    for c in chunks:
        try:
            t = c.decode("latin-1")
        except Exception:
            continue
        if "Tj" not in t and "TJ" not in t:
            continue
        buf = []
        for mm in re.finditer(r"\((?:\\.|[^\\()])*\)|TJ|Tj|T\*|Td|TD", t):
            tok = mm.group(0)
            if tok.startswith("("):
                s = tok[1:-1]
                s = re.sub(r"\\([()\\])", r"\1", s)
                s = re.sub(r"\\[0-7]{1,3}", " ", s)
                buf.append(s)
            elif tok in ("T*", "Td", "TD"):
                buf.append("\n")
        out.append("".join(buf))
    return "\n".join(out)

if __name__ == "__main__":
    t = extract(sys.argv[1])
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(sys.argv) > 2:
        pat = sys.argv[2]
        for i, line in enumerate(t.split("\n")):
            if re.search(pat, line, re.I):
                print(f"{i}: {line.strip()}")
    else:
        print(t[:12000])
