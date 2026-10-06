"""Collect song metadata and lyrics from Genius for local dataset creation."""

import csv
import os
import re

from lyricsgenius import Genius


def fetch_and_save_songs(artist_name, max_songs, output_csv):
    """Fetch an artist's songs and write their metadata and lyrics to CSV."""
    api_key = os.getenv("GENIUS_ACCESS_TOKEN", "").strip()
    if not api_key:
        raise RuntimeError("Set GENIUS_ACCESS_TOKEN before collecting songs")

    genius = Genius(api_key, timeout=15, remove_section_headers=False)
    artist = genius.search_artist(
        artist_name,
        max_songs=max_songs,
        sort="popularity",
        include_features=False,
    )
    if artist is None:
        raise RuntimeError(f"No data found for artist {artist_name!r}")

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Artist", "Title", "Album", "Lyric", "Year"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for song in artist.songs:
            song_data = song.to_dict()
            release = song_data.get("release_date_components") or {}
            album = song_data.get("album") or {}
            lyrics = song_data.get("lyrics") or ""
            start = re.search(r"\[(Intro|Verse\s*1)\]", lyrics, re.IGNORECASE)
            if start:
                lyrics = lyrics[start.start():].strip()

            writer.writerow({
                "Artist": song_data.get("artist", artist_name),
                "Title": song_data.get("title", ""),
                "Album": album.get("name", "Unknown Album"),
                "Lyric": lyrics,
                "Year": release.get("year", ""),
            })

    print(f"Saved {len(artist.songs)} songs to {output_csv!r}")


if __name__ == "__main__":
    fetch_and_save_songs("Sabrina Carpenter", 100, "sabrina_carpenter_lyrics.csv")
