"""Download eight public music recordings and their original attribution files.

URLs and SHA-256 hashes are from librosa's example-data registry. No librosa
installation is needed. Already verified files are reused; network is setup only.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, expected_hash):
    if destination.exists():
        content = destination.read_bytes()
    else:
        print(f"Downloading {destination.name}", flush=True)
        with urlopen(quote(url, safe=":/"), timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch: {destination}. Check the source before replacing it.")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def main():
    tracks = json.loads((ROOT / "data/catalog.json").read_text(encoding="utf-8"))
    for track in tracks:
        download(track["audio_url"], ROOT / "data" / track["path"], track["sha256"])
        download(track["source"], ROOT / "data/attributions" / f"{track['id']}.txt",
                 track["attribution_sha256"])
    print(f"Verified {len(tracks)} music recordings and attribution files.")


if __name__ == "__main__":
    main()
