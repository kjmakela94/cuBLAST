"""Pre-download the sparse matrices used by the notebook into ./matrices/<name>/<name>.mtx(.gz).

    python scripts/download_matrices.py            # default set (NIST Matrix Market + SuiteSparse)
    python scripts/download_matrices.py memplus    # only selected names

The notebook finds files in ./matrices or ../matrices, so run this from the repository root.
SuiteSparse (sparse.tamu.edu) was returning HTTP 403 for every download in Oct 2026; the NIST files work.
"""
import shutil, sys, tarfile, urllib.request
from pathlib import Path

OUT = Path("matrices")
SOURCES = {   # name: direct .mtx.gz URL, or SuiteSparse "Group/Name"
    "s3dkq4m2": "https://math.nist.gov/pub/MatrixMarket2/misc/cylshell/s3dkq4m2.mtx.gz",
    "memplus":  "https://math.nist.gov/pub/MatrixMarket2/misc/hamm/memplus.mtx.gz",
    "ecology2": "McRae/ecology2",
}
SS_MIRRORS = ["https://sparse.tamu.edu/MM/{s}.tar.gz", "http://sparse-files.engr.tamu.edu/MM/{s}.tar.gz"]

def get(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)

def fetch(name, src):
    folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob(f"{name}.mtx*")):
        print(f"  {name}: already present"); return True
    if src.startswith("http"):
        dst = folder / Path(src).name
        try:
            print(f"  {name}: {src}"); get(src, dst)
        except Exception as e:
            print(f"    failed: {e}"); dst.unlink(missing_ok=True); return False
    else:
        tgz = OUT / f"{name}.tar.gz"
        for tmpl in SS_MIRRORS:
            url = tmpl.format(s=src)
            try:
                print(f"  {name}: {url}"); get(url, tgz); break
            except Exception as e:
                print(f"    failed: {e}"); tgz.unlink(missing_ok=True)
        else:
            return False
        with tarfile.open(tgz) as t:
            m = sorted((m for m in t.getmembers() if m.isfile() and m.name.endswith(".mtx")),
                       key=lambda m: Path(m.name).name != f"{name}.mtx")[0]
            with t.extractfile(m) as s, open(folder / f"{name}.mtx", "wb") as d:
                shutil.copyfileobj(s, d)
        tgz.unlink()
    print(f"  {name}: OK ->", *folder.iterdir()); return True

if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    ok = {n: fetch(n, SOURCES[n]) for n in names}
    missing = [n for n, v in ok.items() if not v]
    if missing:
        print("\nnot downloaded:", ", ".join(missing), "→ the notebook will use labelled SYNTHETIC stand-ins for these")
