"""Run the OpenMM Hst5 pilot at the M1 starting lambdas (lambdas_0) and
compare our Rg distribution against the upstream cycle-1 snapshot."""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import examples.calvados_gmx.run_pilot_openmm as rp  # noqa: E402
from examples.calvados.fetch_data import RAW, verified_bytes  # noqa: E402

rp.LAMBDA_COLUMN = "lambdas_0"
rp.OUT = rp.HERE / "pilot_openmm_lambda0"
rp.main()

snapshot = pd.read_pickle(io.BytesIO(verified_bytes(RAW / "1_init_proteinsRgs.pkl")))
upstream = snapshot.loc["Hst5"]
theirs = np.asarray(upstream.rgarray, dtype=float)
ours = np.loadtxt(rp.OUT / "rg.txt")

print()
print("== engine validation at lambdas_0 (M1 start), Hst5, 293 K, I=0.15")
print(f"   upstream (HOOMD, 10-replica concat, 18 us): "
      f"n={len(theirs)}, mean {theirs.mean():.4f} nm, "
      f"std {theirs.std():.4f}, min {theirs.min():.3f}, max {theirs.max():.3f}")
print(f"   ours (OpenMM, single seed, 1.1 ns):         "
      f"n={len(ours)}, mean {ours.mean():.4f} nm, "
      f"std {ours.std():.4f}, min {ours.min():.3f}, max {ours.max():.3f}")
print(f"   upstream recorded mean Rg field: {upstream.Rg:.4f} nm")
tail = ours[len(ours) // 2 :]
print(f"   ours, second half only:              mean {tail.mean():.4f} nm")
print(f"   mean difference (ours - theirs):     {ours.mean() - theirs.mean():+.4f} nm")
overlap = (np.linspace(0, 1, 1000)[:, None] <= 0)  # placeholder
se_theirs = theirs.std() / np.sqrt(len(theirs))
print(f"   expected std of our mean if same dist: {theirs.std() / np.sqrt(len(ours)):.4f} nm")
