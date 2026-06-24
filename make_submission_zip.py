"""Package the OP'26 submission into a single zip.

Includes: source code, notebook, processed datasets, dataset examples (small
UrbanEV samples instead of the 62 MB raw clone), all outputs (figures/CSVs/PDF),
README and requirements. Excludes caches, .git and the full UrbanEV repo.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
ACN_XLSX = PARENT / "acndata_sessions.json.xlsx"
URBANEV = ROOT / "data" / "raw" / "ST-EVCDP" / "datasets"
EXAMPLES = ROOT / "data" / "examples"
ZIP_PATH = PARENT / "OP26_Submission.zip"

# ---- 1. build small dataset examples -------------------------------------- #
EXAMPLES.mkdir(parents=True, exist_ok=True)
if ACN_XLSX.exists():
    shutil.copy(ACN_XLSX, EXAMPLES / "acndata_sessions.json.xlsx")   # primary dataset
if URBANEV.exists():
    for name in ["occupancy", "volume", "price", "time"]:           # big -> sample 300 rows
        pd.read_csv(URBANEV / f"{name}.csv").head(300).to_csv(
            EXAMPLES / f"urbanev_{name}_sample.csv", index=False)
    for name in ["stations", "information"]:                         # small -> include full
        shutil.copy(URBANEV / f"{name}.csv", EXAMPLES / f"urbanev_{name}.csv")
(EXAMPLES / "README.txt").write_text(
    "Dataset examples for OP'26.\n\n"
    "- acndata_sessions.json.xlsx : full primary ACN-Data dataset (16,305 rows).\n"
    "- urbanev_*_sample.csv       : first 300 rows of the UrbanEV/ST-EVCDP 5-min series.\n"
    "- urbanev_stations.csv / urbanev_information.csv : full (small) UrbanEV metadata.\n"
    "- Cleaned analytical tables are in data/processed/.\n"
    "Full UrbanEV (62 MB): git clone https://github.com/IntelligentSystemsLab/ST-EVCDP\n",
    encoding="utf-8")

# ---- 2. collect files ----------------------------------------------------- #
INCLUDE_DIRS = ["src", "notebooks", "outputs", "data/processed", "data/examples"]
INCLUDE_FILES = ["README.md", "requirements.txt", "build_notebook.py",
                 "make_submission_zip.py"]
EXCLUDE = {"__pycache__", ".ipynb_checkpoints", ".git"}


def keep(p: Path) -> bool:
    return not any(part in EXCLUDE for part in p.parts)


files = []
for d in INCLUDE_DIRS:
    for p in (ROOT / d).rglob("*"):
        if p.is_file() and keep(p):
            files.append(p)
for f in INCLUDE_FILES:
    if (ROOT / f).exists():
        files.append(ROOT / f)

# ---- 3. write zip --------------------------------------------------------- #
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, Path("OP26_Submission") / p.relative_to(ROOT))

size_mb = ZIP_PATH.stat().st_size / 1e6
print(f"wrote {ZIP_PATH}  ({size_mb:.1f} MB, {len(files)} files)")
