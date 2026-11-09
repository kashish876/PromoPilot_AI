import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLE = ROOT / "sample_data"


def zip_of(folder: Path, only=None, prefix="show/") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(folder.glob("*.txt")):
            if only is None or any(k in f.name for k in only):
                z.write(f, prefix + f.name)
    return buf.getvalue()
