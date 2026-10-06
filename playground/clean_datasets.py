"""Normalize source CSVs and remove alternate or combined song entries."""

import os
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_FOLDER = Path(os.getenv("CSV_FOLDER", ROOT / "datasets")).resolve()
OUTPUT_FOLDER = CSV_FOLDER / "Cleaned_csvs"
EXCLUDED_TITLE_TERMS = (
    "remix",
    "deluxe",
    "live",
    "acoustic",
    "demo",
    "version",
    "acapella",
    "instrumental",
    "edit",
    "radio edit",
    "mix",
    "mashup",
    "medley",
    "cover",
)
TITLE_PATTERN = re.compile(
    "|".join(re.escape(term) for term in EXCLUDED_TITLE_TERMS),
    re.IGNORECASE,
)


def clean_dataset(csv_path):
    """Return normalized canonical-song rows from one source CSV."""
    frame = pd.read_csv(csv_path)
    frame = frame.rename(columns={"tArtist": "Artist"})
    required = {"Artist", "Title", "Lyric"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"Missing required columns: {missing}")

    titles = frame["Title"].fillna("").astype(str)
    alternate_version = titles.str.contains(TITLE_PATTERN, na=False)
    multi_song_entry = titles.str.count("/") >= 2
    attributed_to_another_artist = titles.str.contains(r"\[[^\]]+\]", regex=True)
    complete = frame[["Artist", "Title", "Lyric"]].notna().all(axis=1)
    return frame[
        complete & ~alternate_version & ~multi_song_entry & ~attributed_to_another_artist
    ].copy()


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(CSV_FOLDER.glob("*.csv")):
        cleaned = clean_dataset(csv_path)
        output_path = OUTPUT_FOLDER / csv_path.name
        cleaned.to_csv(output_path, index=False)
        print(f"Processed {csv_path.name}: kept {len(cleaned)} rows")


if __name__ == "__main__":
    main()
