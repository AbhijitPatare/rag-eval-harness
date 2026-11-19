import re

g_path, e_path = "app/generate.py", "eval/answer_eval.py"
g = open(g_path, encoding="utf-8").read()
m = re.search(r'SYSTEM = """(.*?)"""', g, flags=re.S)
loose = m.group(1)
assert "synonyms" in loose, "generate.py is already strict"
STRICT = '''SYSTEM = """You answer questions using ONLY the numbered context passages provided.
- After each claim, cite the passage number in square brackets, e.g. [2]. Use [1][3] for several. Numbers only.
- If the passages do not contain the answer, reply with exactly: NOT_FOUND
- Never use outside knowledge. Be concise."""'''
g = g[:m.start()] + STRICT + g[m.end():]
open(g_path, "w", encoding="utf-8").write(g)

e = open(e_path, encoding="utf-8").read()
if "LOOSE = g.SYSTEM" in e:
    e = e.replace("LOOSE = g.SYSTEM", 'LOOSE = """' + loose + '"""')
    open(e_path, "w", encoding="utf-8").write(e)
print("generate.py is now strict; LOOSE pinned in answer_eval.py")
