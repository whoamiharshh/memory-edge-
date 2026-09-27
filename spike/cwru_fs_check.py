"""Empirically infer the sampling rate of CWRU normal file 97 (sources disagree / are silent).
Shaft runs at RPM/60 Hz. Assuming fs=12k, find the dominant low-frequency peak; if it sits at RPM/60 -> 12k,
if at RPM/60/4 -> the file is really 48k. Same check on fault file 105 (documented 12k) as a control."""
import numpy as np, scipy.io as sio, scipy.signal as ss
for f in ["97", "105"]:
    m = sio.loadmat(f"data/raw/cwru/{f}.mat")
    x = m[[k for k in m if k.endswith("DE_time")][0]].ravel()
    rpm = float(m[[k for k in m if k.endswith("RPM")][0]].ravel()[0])
    fr, p = ss.welch(x - x.mean(), fs=12000, nperseg=2**15)
    band = (fr > 3) & (fr < 80)
    top = fr[band][np.argsort(p[band])[-3:]][::-1]
    print(f"file {f}: rpm={rpm:.0f} shaft={rpm/60:.2f}Hz  top low-freq peaks assuming 12k: {np.round(top,2)}")
