import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "pm25_24h_pakistan.joblib"
REQ = ROOT / "requirements.txt"

def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))
    return ok

all_ok = True

# Python
all_ok &= check("Python executable", sys.version_info >= (3, 10),
                sys.version.split()[0])

# Required packages and pinned versions
expected = {
    "joblib": "1.5.3",
    "numpy": "2.3.3",
    "pandas": "2.3.2",
    "sklearn": "1.8.0",
    "scipy": "1.17.1",
}

try:
    import joblib, numpy, pandas, sklearn, scipy
    actual = {
        "joblib": joblib.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    for pkg, wanted in expected.items():
        all_ok &= check(f"{pkg} version", actual[pkg] == wanted,
                        f"installed={actual[pkg]}, expected={wanted}")
except Exception as e:
    all_ok &= check("Required packages import", False, str(e))

# Requirements file
all_ok &= check("requirements.txt exists", REQ.exists())

# Model
all_ok &= check("Model file exists", MODEL.exists(), str(MODEL))
if MODEL.exists():
    size_mb = MODEL.stat().st_size / (1024 * 1024)
    all_ok &= check("Model under GitHub 100 MB limit", size_mb < 100,
                    f"{size_mb:.2f} MB")
    try:
        import joblib
        model = joblib.load(MODEL)
        all_ok &= check("Model loads successfully", model is not None,
                        type(model).__name__)
    except Exception as e:
        all_ok &= check("Model loads successfully", False, repr(e))

# Environment secrets
env = ROOT / ".env"
all_ok &= check(".env excluded from repository", not env.exists(),
                "Remove/ignore .env before pushing" if env.exists() else "")

gitignore = ROOT / ".gitignore"
if gitignore.exists():
    txt = gitignore.read_text(errors="ignore")
    all_ok &= check(".env listed in .gitignore", ".env" in txt)
else:
    all_ok &= check(".gitignore exists", False, "Create one before GitHub push")

print()
if all_ok:
    print("PREFLIGHT RESULT: PASS")
    sys.exit(0)
else:
    print("PREFLIGHT RESULT: FAIL")
    sys.exit(1)
