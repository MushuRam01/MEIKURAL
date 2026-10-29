"""Copy every selected single-label Saraga recording into Version 2's dataset.

Preview: python3 -m src.ver2.preprocessing.copy_raag_folders --dry-run
Copy:    python3 -m src.ver2.preprocessing.copy_raag_folders

Selected raags and destination come from src/ver2/config.json.
Complete recording folders include audio and annotations; originals are preserved.
"""

import argparse
from pathlib import Path

from ...ver1.preprocessing.copy_raag_folders import RAAG_IDS, copy_folders, plan_copies
from ..config import DEFAULT_CONFIG, ROOT, load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/saraga_hindustani/saraga1.5_hindustani")
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating directories or copying files")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        names = config["classes"]
        unknown = [name for name in names if name not in RAAG_IDS]
        if unknown:
            raise ValueError(f"No explicit Saraga ID mapping for: {', '.join(unknown)}; update RAAG_IDS first")
        destination = resolve_path(config["paths"]["dataset"]).resolve()
        planned = plan_copies(args.source.expanduser().resolve(), destination, names)
        copy_folders(planned, names, args.dry_run)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
