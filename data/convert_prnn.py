#!/usr/bin/env python3
"""
Convert PRNN-format whitespace-delimited data files into the CSV format
expected by pnn-cli (comma-separated, last column named `label`, 0-based
integer class ids).

Outputs are written alongside this script in data/ as
  synth_train.csv / synth_test.csv
  pima_train.csv  / pima_test.csv
  fglass_train.csv / fglass_test.csv
  crabs_train.csv  / crabs_test.csv
  viruses_train.csv / viruses_test.csv
  cushings_train.csv / cushings_test.csv

Usage:
    python3 convert_prnn.py
"""

import csv
import io
import os
import random

PRNN_DIR = os.path.join(os.path.dirname(__file__), "PRNN")
OUT_DIR = os.path.dirname(__file__)
SPLIT_SEED = 42
TRAIN_FRAC = 0.8


# ── helpers ───────────────────────────────────────────────────────────────────

def read_whitespace(path, has_header=True):
    """Read a whitespace-delimited file. Returns (header_list, rows_list)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split())
    if has_header:
        return rows[0], rows[1:]
    return None, rows


def encode_labels(column, mapping=None):
    """Map a list of string labels to 0-based integers.
    If mapping is not provided, build one from sorted unique values.
    Returns (encoded_list, mapping_dict)."""
    if mapping is None:
        mapping = {v: i for i, v in enumerate(sorted(set(column)))}
    return [mapping[v] for v in column], mapping


def write_csv(path, feature_names, rows, labels):
    """Write feature rows + labels to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ["label"])
        for row, lbl in zip(rows, labels):
            writer.writerow(row + [lbl])
    print(f"  wrote {path}  ({len(rows)} rows, {len(feature_names)} features)")


def random_split(rows, labels, train_frac, seed):
    """Stratified-by-class shuffle+split."""
    rng = random.Random(seed)
    combined = list(zip(rows, labels))
    rng.shuffle(combined)
    n_train = max(1, int(len(combined) * train_frac))
    train = combined[:n_train]
    test = combined[n_train:]
    tr_rows, tr_labels = zip(*train) if train else ([], [])
    te_rows, te_labels = zip(*test) if test else ([], [])
    return list(tr_rows), list(tr_labels), list(te_rows), list(te_labels)


# ── datasets ──────────────────────────────────────────────────────────────────

def convert_synth():
    """synth.tr / synth.te — columns: xs ys yc.  yc is already 0/1."""
    print("synth:")
    for split, fname, out in [
        ("train", "synth.tr", "synth_train.csv"),
        ("test",  "synth.te", "synth_test.csv"),
    ]:
        header, rows = read_whitespace(os.path.join(PRNN_DIR, fname))
        label_col = header.index("yc")
        feat_names = [c for c in header if c != "yc"]
        feats = [[r[i] for i in range(len(header)) if i != label_col] for r in rows]
        labels = [int(r[label_col]) for r in rows]
        write_csv(os.path.join(OUT_DIR, out), feat_names, feats, labels)


def convert_pima():
    """pima.tr / pima.te — label column 'type', values Yes/No."""
    print("pima:")
    mapping = {"No": 0, "Yes": 1}
    for split, fname, out in [
        ("train", "pima.tr", "pima_train.csv"),
        ("test",  "pima.te", "pima_test.csv"),
    ]:
        header, rows = read_whitespace(os.path.join(PRNN_DIR, fname))
        label_col = header.index("type")
        feat_names = [c for c in header if c != "type"]
        feats = [[r[i] for i in range(len(header)) if i != label_col] for r in rows]
        labels = [mapping[r[label_col]] for r in rows]
        write_csv(os.path.join(OUT_DIR, out), feat_names, feats, labels)


def convert_fglass():
    """fglass.dat — label column 'type', integer values 1-7 (4 is absent).
    Remapped to contiguous 0-based ids."""
    print("fglass:")
    header, rows = read_whitespace(os.path.join(PRNN_DIR, "fglass.dat"))
    label_col = header.index("type")
    feat_names = [c for c in header if c != "type"]
    feats = [[r[i] for i in range(len(header)) if i != label_col] for r in rows]
    raw_labels = [r[label_col] for r in rows]
    labels, mapping = encode_labels(raw_labels)
    print(f"  class mapping: {mapping}")
    tr_rows, tr_labels, te_rows, te_labels = random_split(feats, labels, TRAIN_FRAC, SPLIT_SEED)
    write_csv(os.path.join(OUT_DIR, "fglass_train.csv"), feat_names, tr_rows, tr_labels)
    write_csv(os.path.join(OUT_DIR, "fglass_test.csv"), feat_names, te_rows, te_labels)


def convert_crabs():
    """crabs.dat — classify crab species (sp: B=0 / O=1).
    Drops the row-index column; encodes sex (M=0, F=1) as a feature."""
    print("crabs:")
    header, rows = read_whitespace(os.path.join(PRNN_DIR, "crabs.dat"))
    # columns: sp sex index FL RW CL CW BD
    sp_col   = header.index("sp")
    sex_col  = header.index("sex")
    idx_col  = header.index("index")
    drop = {idx_col}
    feat_names = [c for i, c in enumerate(header) if i not in drop and i != sp_col]
    sex_map = {"M": "0", "F": "1"}
    feats = []
    for r in rows:
        row = []
        for i, v in enumerate(r):
            if i in drop or i == sp_col:
                continue
            row.append(sex_map.get(v, v))
        feats.append(row)
    raw_labels = [r[sp_col] for r in rows]
    labels, mapping = encode_labels(raw_labels)
    print(f"  class mapping: {mapping}")
    tr_rows, tr_labels, te_rows, te_labels = random_split(feats, labels, TRAIN_FRAC, SPLIT_SEED)
    write_csv(os.path.join(OUT_DIR, "crabs_train.csv"), feat_names, tr_rows, tr_labels)
    write_csv(os.path.join(OUT_DIR, "crabs_test.csv"), feat_names, te_rows, te_labels)


def convert_viruses():
    """viruses.dat — no header, 18 columns, last column is class (1-5).
    Remapped to 0-based."""
    print("viruses:")
    _, rows = read_whitespace(os.path.join(PRNN_DIR, "viruses.dat"), has_header=False)
    n_cols = len(rows[0])
    feat_names = [f"f{i}" for i in range(n_cols - 1)]
    feats = [r[:-1] for r in rows]
    raw_labels = [r[-1] for r in rows]
    labels, mapping = encode_labels(raw_labels)
    print(f"  class mapping: {mapping}")
    tr_rows, tr_labels, te_rows, te_labels = random_split(feats, labels, TRAIN_FRAC, SPLIT_SEED)
    write_csv(os.path.join(OUT_DIR, "viruses_train.csv"), feat_names, tr_rows, tr_labels)
    write_csv(os.path.join(OUT_DIR, "viruses_test.csv"), feat_names, te_rows, te_labels)


def convert_cushings():
    """Cushings.dat — drop patient Label, encode Type (a/b/c/o).
    Very small (27 rows), keeps all rows including the 6 uncertain diagnoses."""
    print("cushings:")
    header, rows = read_whitespace(os.path.join(PRNN_DIR, "Cushings.dat"))
    # columns: Label Tetrahydrocortisone Pregnanetriol Type
    label_col = header.index("Type")
    drop = {header.index("Label")}
    feat_names = [c for i, c in enumerate(header) if i not in drop and i != label_col]
    feats = [[r[i] for i in range(len(header)) if i not in drop and i != label_col] for r in rows]
    raw_labels = [r[label_col] for r in rows]
    labels, mapping = encode_labels(raw_labels)
    print(f"  class mapping: {mapping}")
    tr_rows, tr_labels, te_rows, te_labels = random_split(feats, labels, TRAIN_FRAC, SPLIT_SEED)
    write_csv(os.path.join(OUT_DIR, "cushings_train.csv"), feat_names, tr_rows, tr_labels)
    write_csv(os.path.join(OUT_DIR, "cushings_test.csv"), feat_names, te_rows, te_labels)


if __name__ == "__main__":
    convert_synth()
    convert_pima()
    convert_fglass()
    convert_crabs()
    convert_viruses()
    convert_cushings()
    print("\nDone. All CSVs written to", OUT_DIR)
