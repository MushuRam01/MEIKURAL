"""Generate the config/model tables: python -m src.ver2.documentation.generate."""

import argparse
import json
from pathlib import Path

from ..config import DEFAULT_CONFIG, ROOT, feature_shape, load_config
from ..training.model import model_dimensions


def tex(value):
    escapes = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%",
               "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(escapes.get(char, char) for char in str(value))


def generate(config, destination):
    rows, total = model_dimensions(config)
    lines = ["% Generated from config.json; do not edit numerical defaults here.",
             r"\subsection{Configured settings}",
             r"\begin{longtable}{p{0.49\linewidth}p{0.44\linewidth}}",
             r"\textbf{Setting} & \textbf{Value} \\\hline\endhead"]
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                lines.append(f"{tex(section + '.' + key)} & {tex(json.dumps(value))} " + r"\\")
        else:
            lines.append(f"{tex(section)} & {tex(', '.join(values))} " + r"\\")
    lines += [r"\end{longtable}", r"\subsection{Derived CNN dimensions}",
              f"Feature shape: {tex(feature_shape(config))}. Trainable parameters: {total:,}.",
              r"\begin{longtable}{llll}",
              r"\textbf{Layer} & \textbf{Output (C,H,W)} & \textbf{Parameters} & \textbf{RF (H,W)} \\\hline\endhead"]
    for row in rows:
        shape = " x ".join(map(str, row["shape"]))
        receptive = " x ".join(map(str, row.get("receptive_field", []))) or "aggregated"
        lines.append(f"{tex(row['layer'])} & {shape} & {row['parameters']:,} & {receptive} " + r"\\")
    lines.append(r"\end{longtable}")
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/ver2_tables.tex")
    args = parser.parse_args()
    generate(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
