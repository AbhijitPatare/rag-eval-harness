import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
names = [m.name.replace("models/", "") for m in c.models.list() if ("flash" in m.name or "lite" in m.name) and "image" not in m.name]
for n in names:
    try:
        c.models.generate_content(model=n, contents="Reply with OK")
        print("WORKS", n)
    except errors.APIError as e:
        print("FAIL ", n, e.code, str(e.message)[:70])
    except Exception as e:
        print("FAIL ", n, type(e).__name__)
