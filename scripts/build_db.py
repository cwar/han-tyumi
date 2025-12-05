#!/usr/bin/env python3
"""Build SQLite database from kglw.net API + lyrics.json.

This script:
1. Fetches fresh show/setlist/song data from the kglw.net API
2. Merges lyrics from data/lyrics.json (static file)
3. Creates a SQLite database for han-tyumi

Run this at container startup to get fresh data.
"""

import json
import sqlite3
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

API_BASE = "https://kglw.net/api/v2"
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR / "kglw.db"
LYRICS_PATH = PROJECT_DIR / "data" / "lyrics.json"


def fetch_json(endpoint: str, params: dict | None = None, retries: int = 3) -> list[dict]:
    """Fetch JSON data from the API with retry logic."""
    url = f"{API_BASE}/{endpoint}.json"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"

    print(f"  Fetching: {url}")
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=120) as response:
                data = json.loads(response.read().decode())
                if data.get("error"):
                    print(f"  API Error: {data.get('error_message')}")
                    return []
                return data.get("data", [])
        except (HTTPError, URLError, TimeoutError) as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    print(f"  Failed to fetch {url} after {retries} attempts")
    return []


def load_lyrics() -> dict[str, str]:
    """Load lyrics from static JSON file."""
    if not LYRICS_PATH.exists():
        print(f"  Lyrics file not found: {LYRICS_PATH}")
        return {}

    with open(LYRICS_PATH, 'r', encoding='utf-8') as f:
        lyrics = json.load(f)

    print(f"  Loaded {len(lyrics)} songs with lyrics")
    return lyrics


def create_tables(conn: sqlite3.Connection):
    """Create database tables."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT,
            isoriginal INTEGER DEFAULT 1,
            original_artist TEXT,
            lyrics TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shows (
            id INTEGER PRIMARY KEY,
            showdate TEXT,
            showtitle TEXT,
            venue_id INTEGER,
            tour_id INTEGER,
            shownotes TEXT,
            showyear INTEGER,
            showorder INTEGER,
            opener TEXT,
            soundcheck TEXT,
            isverified INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS venues (
            id INTEGER PRIMARY KEY,
            venuename TEXT,
            city TEXT,
            state TEXT,
            country TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tours (
            id INTEGER PRIMARY KEY,
            tourname TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS setlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            show_id INTEGER,
            song_id INTEGER,
            position INTEGER,
            settype TEXT,
            setnumber TEXT,
            transition TEXT,
            footnote TEXT,
            isjamchart INTEGER DEFAULT 0,
            isreprise INTEGER DEFAULT 0,
            isjam INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS albums (
            id INTEGER PRIMARY KEY,
            albumtitle TEXT,
            displayname TEXT,
            slug TEXT,
            releasedate TEXT,
            cover TEXT,
            album_notes TEXT,
            artist_id INTEGER DEFAULT 1
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id INTEGER,
            discography_id INTEGER,
            position INTEGER,
            tracktime TEXT,
            disc_number INTEGER DEFAULT 1
        )
    """)

    conn.commit()


def populate_songs(conn: sqlite3.Connection, lyrics: dict[str, str]):
    """Fetch songs from API and merge with lyrics."""
    print("\nFetching songs...")
    songs = fetch_json("songs")

    cursor = conn.cursor()
    for song in songs:
        song_id = song["id"]
        song_lyrics = lyrics.get(str(song_id), "")

        cursor.execute("""
            INSERT OR REPLACE INTO songs (id, name, slug, isoriginal, original_artist, lyrics)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (song_id, song["name"], song["slug"],
              song.get("isoriginal", 1), song.get("original_artist", ""), song_lyrics))

    conn.commit()
    with_lyrics = sum(1 for s in songs if lyrics.get(str(s["id"])))
    print(f"  Inserted {len(songs)} songs ({with_lyrics} with lyrics)")


def populate_venues(conn: sqlite3.Connection):
    """Fetch venues from API."""
    print("\nFetching venues...")
    venues = fetch_json("venues")

    cursor = conn.cursor()
    for venue in venues:
        cursor.execute("""
            INSERT OR REPLACE INTO venues (id, venuename, city, state, country)
            VALUES (?, ?, ?, ?, ?)
        """, (venue["venue_id"], venue["venuename"], venue.get("city", ""),
              venue.get("state", ""), venue.get("country", "")))

    conn.commit()
    print(f"  Inserted {len(venues)} venues")


def populate_shows_and_setlists(conn: sqlite3.Connection):
    """Fetch shows and setlists from API."""
    cursor = conn.cursor()

    # Shows
    print("\nFetching shows...")
    shows = fetch_json("shows")
    tours = {}

    for show in shows:
        tour_id = show.get("tour_id")
        if tour_id and tour_id not in tours:
            tours[tour_id] = show.get("tourname", "")

        cursor.execute("""
            INSERT OR REPLACE INTO shows (id, showdate, showtitle, venue_id, tour_id,
                                          shownotes, showyear, showorder, opener, soundcheck, isverified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (show["show_id"], show["showdate"], show.get("showtitle", ""),
              show.get("venue_id"), tour_id, "", show.get("show_year"),
              show.get("showorder", 1), "", "", 0))

    conn.commit()
    print(f"  Inserted {len(shows)} shows")

    # Tours
    for tour_id, tourname in tours.items():
        cursor.execute("INSERT OR REPLACE INTO tours (id, tourname) VALUES (?, ?)",
                       (tour_id, tourname))
    conn.commit()
    print(f"  Inserted {len(tours)} tours")

    # Setlists (with full data including show notes)
    print("\nFetching setlists...")
    setlists = fetch_json("setlists", {"limit": "100000"})

    show_updates = {}
    for entry in setlists:
        show_id = entry.get("show_id")
        if show_id and show_id not in show_updates:
            show_updates[show_id] = {
                "shownotes": entry.get("shownotes", ""),
                "opener": entry.get("opener", ""),
                "soundcheck": entry.get("soundcheck", "")
            }

        cursor.execute("""
            INSERT INTO setlists (show_id, song_id, position, settype, setnumber,
                                  transition, footnote, isjamchart, isreprise, isjam)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (show_id, entry.get("song_id"), entry.get("position", 0),
              entry.get("settype", "Set"), entry.get("setnumber", "1"),
              entry.get("transition", ", "), entry.get("footnote", ""),
              entry.get("isjamchart", 0), entry.get("isreprise", 0), entry.get("isjam", 0)))

    # Update shows with complete info
    for show_id, info in show_updates.items():
        cursor.execute("""
            UPDATE shows SET shownotes = ?, opener = ?, soundcheck = ? WHERE id = ?
        """, (info["shownotes"], info["opener"], info["soundcheck"], show_id))

    conn.commit()
    print(f"  Inserted {len(setlists)} setlist entries")
    print(f"  Updated {len(show_updates)} shows with notes/opener info")


def populate_albums(conn: sqlite3.Connection):
    """Fetch albums from API."""
    print("\nFetching albums...")
    albums = fetch_json("albums")

    cursor = conn.cursor()
    seen_albums = {}

    for entry in albums:
        album_title = entry.get("album_title", "")
        artist_id = entry.get("artist_id", 1)
        album_key = (album_title, artist_id)

        if album_key not in seen_albums:
            album_id = abs(hash(album_key)) % (10**9)
            seen_albums[album_key] = album_id

            album_url = entry.get("album_url", "")
            album_slug = album_url.replace("/albums/", "") if album_url else ""

            cursor.execute("""
                INSERT OR REPLACE INTO albums (id, albumtitle, displayname, slug, releasedate,
                                               cover, album_notes, artist_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (album_id, album_title, entry.get("album_displayname", ""),
                  album_slug, entry.get("releasedate", ""), "",
                  entry.get("album_notes", ""), artist_id))

        # Link track to song
        song_name = entry.get("song_name", "")
        if song_name:
            cursor.execute("""
                INSERT INTO tracks (song_id, discography_id, position, tracktime, disc_number)
                SELECT id, ?, ?, ?, ? FROM songs WHERE name = ?
            """, (seen_albums[album_key], entry.get("position", 0),
                  entry.get("tracktime", ""), entry.get("disc_number", 1), song_name))

    conn.commit()
    print(f"  Inserted {len(seen_albums)} albums")


def main():
    print("=== Building kglw.db from API + lyrics.json ===\n")

    # Remove existing database
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing database")

    # Load lyrics
    print("Loading lyrics...")
    lyrics = load_lyrics()

    # Create database
    conn = sqlite3.connect(DB_PATH)

    try:
        print("\nCreating tables...")
        create_tables(conn)

        populate_songs(conn, lyrics)
        time.sleep(0.5)

        populate_venues(conn)
        time.sleep(0.5)

        populate_shows_and_setlists(conn)
        time.sleep(0.5)

        populate_albums(conn)

        # Summary
        cursor = conn.cursor()
        print("\n=== Database Summary ===")
        for table in ["songs", "shows", "venues", "tours", "setlists", "albums", "tracks"]:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} rows")

        lyrics_count = cursor.execute(
            "SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''"
        ).fetchone()[0]
        print(f"\n  Songs with lyrics: {lyrics_count}")

        print(f"\nDatabase saved to: {DB_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
