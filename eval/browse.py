"""Find the page number for a phrase. Usage: python -m eval.browse "assistant professor" """
import sys
from pathlib import Path

import pymupdf

term = sys.argv[1].lower()
for pdf in sorted(Path("data").glob("*.pdf")):
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            pos = text.lower().find(term)
            if pos >= 0:
                snippet = text[max(0, pos - 60): pos + 100].replace("\n", " ")
                print(f"{pdf.name} p{i}: ...{snippet}...")
