"""iTunes song fetcher.

This module provides helpers to fetch song metadata from the iTunes Search API
and save it to CSV.

Important: previously, this file executed a very large bulk download at import
time. That made the project slow/unusable when any other script imported it.
Now, fetching only happens when running this file as a script.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import requests

def fetch_itunes_bulk(terms, limit=200, country="us"):
    all_results = []

    for term in terms:
        params = {
            "term": term,
            "entity": "song",
            "limit": limit,
            "country": country
        }

        url = "https://itunes.apple.com/search"
        response = requests.get(url, params=params).json()

        all_results.extend(response.get("results", []))

    df = pd.DataFrame(all_results).drop_duplicates(subset="trackId")
    return df
def save_to_csv(df, path="data/itunes_songs.csv"):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch iTunes songs and save to CSV")
    p.add_argument(
        "--letters",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of letters to generate search terms (1 or 2). 2 can be large.",
    )
    p.add_argument("--limit", type=int, default=200, help="iTunes API limit per term")
    p.add_argument("--country", default="us", help="iTunes country code (e.g. us, br)")
    p.add_argument(
        "--out",
        default="itunes_songs.csv",
        help="Output CSV path (default: itunes_songs.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    terms = ["".join(p) for p in itertools.product(alphabet, repeat=args.letters)]

    print(f"Fetching iTunes songs for {len(terms)} terms (limit={args.limit}, country={args.country})...")
    df = fetch_itunes_bulk(terms, limit=args.limit, country=args.country)
    out = save_to_csv(df, args.out)
    print(f"Saved {len(df)} unique tracks to {out}")


if __name__ == "__main__":
    main()