import random
from pathlib import Path
from app.chunking import chunk
from app.ingest import load_pages, is_garbled

dropped_samples = []
for pdf in sorted(Path("data").glob("*.pdf")):
    kept = dropped = 0
    for page, text in load_pages(pdf):
        for c in chunk(text, "recursive"):
            if is_garbled(c):
                dropped += 1
                dropped_samples.append((pdf.name, page, c))
            else:
                kept += 1
    print(f"{pdf.name:45s} kept {kept:4d}  dropped {dropped:4d}")

print("\n--- 6 random dropped chunks from NON-min-qualifications docs ---")
other = [s for s in dropped_samples if "min-qualifications" not in s[0]]
for doc, page, c in random.sample(other, min(6, len(other))):
    print(f"\n[{doc} p{page}]\n{c[:300]}")
