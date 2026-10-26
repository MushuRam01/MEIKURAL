"""Copy Version 1's selected Saraga recording folders, including all annotations.

Preview:
    python src/ver1/preprocessing/copy_raag_folders.py --dry-run
Copy:
    python src/ver1/preprocessing/copy_raag_folders.py

Paths default to the repository, independently of the working directory.
Existing identical copies are skipped; different contents are never overwritten.
"""

import argparse
import filecmp
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Match metadata identifiers, not folder substrings or approximate raag names.
RAAG_IDS = {
    "Shree": "46997b02-f09c-4969-8138-4e1861f61967",
    "Jog": "063ea5a0-23b1-4bb5-8537-3d924fe8ebb3",
    "Miyan Malhar": "1b05a564-059f-445b-b325-cf26318367e3",
    "Lalit": "a99e07d5-20a0-467b-8dcd-aa5a095177fd",
    "Todi": "6f13484e-6fdd-402d-baf3-3835835454d0",
    "Marwa": "3eb7ba30-4b94-432e-9618-875ee57e01ab",
    "Bhairavi": "e6710c21-4b6c-413d-abb5-2146bad52b9d",
}


def read_raag_list(path):
    """Read the first column of the Version1 Markdown table in raag_list.txt."""
    selected = []
    in_version = False
    for line in path.read_text(encoding="utf-8").splitlines():
        version = re.match(r"^\s*Version\s*(\d+)\b", line, re.IGNORECASE)
        if version:
            if in_version:
                break
            in_version = version.group(1) == "1"
            continue
        if not in_version or not line.strip().startswith("|"):
            continue
        name = line.split("|")[1].strip()
        if not name or name.casefold() == "raag" or set(name) <= {"-", ":"}:
            continue
        if name not in RAAG_IDS:
            raise ValueError(f"No explicit Saraga ID mapping for {name!r}; update RAAG_IDS first.")
        if name in selected:
            raise ValueError(f"Duplicate raag in Version1: {name}")
        selected.append(name)
    if not selected:
        raise ValueError(f"No raags found in the Version1 table: {path}")
    return selected


def identical_folders(source, destination):
    """Compare relative entries and full file bytes; do not trust mtimes."""
    if not destination.is_dir() or destination.is_symlink():
        return False
    left = {p.relative_to(source): p for p in source.rglob("*")}
    right = {p.relative_to(destination): p for p in destination.rglob("*")}
    if left.keys() != right.keys():
        return False
    for key, path in left.items():
        other = right[key]
        if path.is_symlink() or other.is_symlink() or path.is_dir() != other.is_dir():
            return False
        if path.is_file() and not filecmp.cmp(path, other, shallow=False):
            return False
    return True


def plan_copies(source, destination, names):
    if not source.is_dir():
        raise FileNotFoundError(f"Saraga collection not found: {source}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-nested directories.")
    name_by_id = {RAAG_IDS[name]: name for name in names}
    planned = []
    targets = set()
    source_folders = set()
    for metadata_path in sorted(source.rglob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raags = metadata.get("raags") or []
        if len(raags) != 1 or raags[0].get("uuid") not in name_by_id:
            continue
        name = name_by_id[raags[0]["uuid"]]
        folder = metadata_path.parent
        if folder in source_folders:
            raise ValueError(f"Multiple selected metadata files in one recording folder: {folder}")
        if not list(folder.glob("*.mp3")):
            raise ValueError(f"Selected recording has no MP3 audio: {folder}")
        # Keep the release/artist in the name: many source folders share 'Raag Jog'.
        target = destination / name / f"{folder.parent.name}__{folder.name}"
        if target in targets:
            raise ValueError(f"Destination name collision: {target}")
        source_folders.add(folder)
        targets.add(target)
        if target.exists() and not identical_folders(folder, target):
            raise FileExistsError(f"Destination differs from source; refusing to overwrite: {target}")
        planned.append((name, folder, target))
    counts = Counter(name for name, _, _ in planned)
    missing = [name for name in names if not counts[name]]
    if missing:
        raise ValueError(f"No matching recordings for: {', '.join(missing)}")
    return planned


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raag-list", type=Path, default=ROOT / "docs/raag_list.txt")
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/saraga_hindustani/saraga1.5_hindustani")
    parser.add_argument("--destination", type=Path, default=ROOT / "data/ver1/set")
    parser.add_argument("--dry-run", action="store_true", help="Show selected folders without copying anything")
    args = parser.parse_args()
    try:
        names = read_raag_list(args.raag_list.resolve())
        planned = plan_copies(args.source.resolve(), args.destination.resolve(), names)
        copied = skipped = 0
        for name, source, target in planned:
            if target.exists():
                print(f"SKIP (identical): {target}")
                skipped += 1
            elif args.dry_run:
                print(f"WOULD COPY: {source}\n         -> {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Publish a complete folder only; failed copies leave no partial target.
                with tempfile.TemporaryDirectory(prefix=".copy-", dir=target.parent) as temporary:
                    staged = Path(temporary) / "recording"
                    shutil.copytree(source, staged)
                    staged.rename(target)
                print(f"COPIED: {target}")
                copied += 1
        counts = Counter(name for name, _, _ in planned)
        print("\nSelected recording folders:")
        for name in names:
            print(f"  {name}: {counts[name]}")
        print(f"Total: {len(planned)} | copied: {copied} | already present: {skipped}")
        if args.dry_run:
            print("Dry run only; no directories or audio files were created.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
