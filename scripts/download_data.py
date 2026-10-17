"""Download the real PaySim dataset (6,362,620 rows, 493 MB) into data/paysim.csv.

Source: community mirror of the Kaggle dataset `ealaxi/paysim1`
(Lopez-Rojas et al., PaySim). Official page: https://www.kaggle.com/datasets/ealaxi/paysim1
If the mirror is down, download from Kaggle and save the CSV as data/paysim.csv.
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/theman10/paysim/resolve/main/paysim.csv"
DEST = Path(__file__).resolve().parent.parent / "data" / "paysim.csv"
EXPECTED_ROWS = 6_362_620


def main():
    if DEST.exists():
        print(f"{DEST} already exists")
    else:
        DEST.parent.mkdir(exist_ok=True)
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, DEST)
    with open(DEST, "rb") as f:
        rows = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b"")) - 1
    print(f"rows: {rows:,}")
    if rows != EXPECTED_ROWS:
        sys.exit(f"expected {EXPECTED_ROWS:,} rows - file looks truncated/corrupt")
    print("OK")


if __name__ == "__main__":
    main()
