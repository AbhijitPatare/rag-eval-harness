"""Two chunking strategies with the same interface: chunk(text, strategy) -> list[str]."""

SEPARATORS = ["\n\n", "\n", ". ", " "]


def fixed(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(text), step):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return fixed(text, size, 0)
    sep, rest = seps[0], seps[1:]
    out, buf = [], ""
    for part in text.split(sep):
        if len(part) > size:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split(part, size, rest))
            continue
        cand = f"{buf}{sep}{part}" if buf else part
        if len(cand) <= size:
            buf = cand
        else:
            out.append(buf)
            buf = part
    if buf:
        out.append(buf)
    return out


def recursive(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    pieces = [p.strip() for p in _split(text, size, SEPARATORS) if p.strip()]
    result = []
    for i, p in enumerate(pieces):
        if i > 0 and overlap:
            tail = pieces[i - 1][-overlap:]
            if " " in tail:
                tail = tail[tail.find(" ") + 1:]  # don't start mid-word
            p = f"{tail} {p}"
        result.append(p)
    return result


STRATEGIES = {"fixed": fixed, "recursive": recursive}


def chunk(text: str, strategy: str) -> list[str]:
    return STRATEGIES[strategy](text)

STRATEGIES["recursive_small"] = lambda text: recursive(text, 400, 50)
