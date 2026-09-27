"""Diagnose K2: per CWRU file, median order features (envelope energy at defect frequencies) and RMS."""
import numpy as np
from data.splits import build_dataset
from edge.fingerprint import FEATURE_NAMES, PHYSICS_SLICE

d = build_dataset()
names = FEATURE_NAMES[PHYSICS_SLICE]
print("fid  class       size load  logrms  " + "  ".join(f"{n:>14s}" for n in names) + "  argmax(defect)")
for f in np.unique(d["fid"]):
    m = d["fid"] == f
    o = np.median(d["X"][m][:, PHYSICS_SLICE], axis=0)
    rms = np.median(d["X"][m][:, 0])
    arg = names[1 + int(np.argmax(o[1:]))]
    print(f"{f:4d} {d['cls'][m][0]:11s} {d['size'][m][0]:4d} {d['load'][m][0]:4d}  {rms:6.2f}  "
          + "  ".join(f"{v:14.2f}" for v in o) + f"  {arg}")
