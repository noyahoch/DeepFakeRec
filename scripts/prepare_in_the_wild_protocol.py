from __future__ import annotations

import argparse
import csv
from pathlib import Path


FILE_COLUMN_CANDIDATES = (
    "file",
    "filename",
    "file_name",
    "path",
    "audio",
    "audio_path",
    "sample",
    "sample_name",
)

LABEL_COLUMN_CANDIDATES = (
    "label",
    "class",
    "target",
    "is_fake",
    "type",
)


def _find_file_column(fieldnames: list[str]) -> str:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in FILE_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for name in fieldnames:
        if "file" in name.lower() or "path" in name.lower():
            return name
    raise ValueError(f"Could not detect file column in {fieldnames}")


def _find_label_column(fieldnames: list[str]) -> str:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for name in fieldnames:
        if "label" in name.lower() or "fake" in name.lower() or "class" in name.lower():
            return name
    raise ValueError(f"Could not detect label column in {fieldnames}")


def _normalise_label(raw_label: str) -> str:
    value = raw_label.strip().lower()
    if value in {"bonafide", "bona-fide", "bona_fide", "real", "human", "genuine", "0"}:
        return "bonafide"
    if value in {"spoof", "fake", "deepfake", "synthetic", "1"}:
        return "spoof"
    raise ValueError(f"Unsupported label value: {raw_label!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert In-The-Wild metadata CSV into the ASVspoof-style protocol format used by this repo."
    )
    parser.add_argument("--metadata", required=True, help="Path to the dataset CSV metadata file.")
    parser.add_argument("--output", required=True, help="Where to write the generated protocol text file.")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_path = Path(args.output)

    with open(metadata_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {metadata_path}")
        file_column = _find_file_column(reader.fieldnames)
        label_column = _find_label_column(reader.fieldnames)

        rows_written = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            for row in reader:
                filename = (row.get(file_column) or "").strip()
                label = _normalise_label(row.get(label_column) or "")
                if not filename:
                    continue
                speaker_id = "ITW"
                out.write(f"{speaker_id} {filename} - - {label}\n")
                rows_written += 1

    print(
        f"Wrote {rows_written} protocol rows to {output_path} using "
        f"file column {file_column!r} and label column {label_column!r}."
    )


if __name__ == "__main__":
    main()
