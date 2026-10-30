# CALVADOS2 surrogate training data

From the repository root, using an environment with BFF, PyTorch and pandas:

```bash
python examples/calvados/fetch_data.py
python -m examples.calvados.convert_data
```

The downloader uses only the Python standard library. Conversion additionally
requires pandas to read the upstream snapshots (`pip install pandas` if needed).
Paths resolve relative to the scripts, independently of the working directory.

`raw/M1_24/` holds unchanged, checksum-verified upstream files. Conversion writes
one BFF `QoIDataset` per protein to `generated/qoi/`, plus `generated/summary.json`.
Each dataset has 11 rows, 20 parameter columns sorted by amino-acid one-letter
code, and one Rg output. The experimental targets come from the upstream
snapshots. Your `c2_rg.csv` remains separate.

Downloaded data and generated outputs are ignored by Git; scripts and provenance
are versionable. See [SOURCE.md](SOURCE.md) for the fixed commit, cycle mapping,
units, and limitations.

This is an initial data import, not a validated inference example. Eleven
parameter sets cannot broadly cover 20 parameter dimensions; the centered
training matrix has rank at most ten. The 51 proteins provide separate outputs,
not additional parameter-space locations. No surrogate fitting, posterior
sampling, or new MD is performed. Additional compatible parameter sets or a
scientifically justified reduction of parameter dimension are needed before
reliable inference. Different-cutoff runs must not be pooled as the same model.
