import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "build" / "space"
if out.exists():
    shutil.rmtree(out)
(out / "ui").mkdir(parents=True)
shutil.copytree(root / "app", out / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copy(root / "ui" / "app.py", out / "ui" / "app.py")
for f in ("Dockerfile", "requirements.txt", "README.md"):
    shutil.copy(root / "deploy" / f, out / f)
bad = [p for p in out.rglob("*") if p.name.startswith(".env")]
assert not bad, bad
print("built", out)
for p in sorted(out.rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(out))
