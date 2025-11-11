#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
from scipy.stats import chi2
from scipy.signal import find_peaks


# In[29]:


def detrend_anomaly(y): 
    """Remove linear trend and mean, return zero-mean anomalies."""
    x = np.arange(len(y))
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_dt = y - (m * x + b)
    return y_dt - np.mean(y_dt)

def ar1_red_noise_spectrum(var, r1, freqs):
    """
    This function calculates the expected spectral power of a red-noise (AR(1))
    """
    w = 2 * np.pi * freqs
    return var * (1 - r1**2) / (1 + r1**2 - 2 * r1 * np.cos(w))

def estimate_ar1(y):
    """
    This function calculates how much a time series correlates with its previous 
    value (lag-1 autocorrelation) and its variance
    """
    y = y - np.mean(y)
    var = np.var(y, ddof=1)
    y1, y2 = y[:-1], y[1:]
    r1 = np.sum(y1 * y2) / np.sum(y1**2)
    return r1, var

def mtm_psd(y, dt=1.0, NW=3.0, K=None, adaptive=True, nfft=None):
    """
    Multitaper PSD with DPSS tapers; adaptive Thomson weighting by default.
    Returns (freqs, S_mtm) with freqs in cycles per dt.
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(N)))
    if K is None:
        K = int(2 * NW - 1)  # =5 for NW=3

    tapers, eigs = dpss(N, NW, K, return_ratios=True)
    x = detrend_anomaly(y)

    Sk = []
    for k in range(K):
        Xk = np.fft.rfft(x * tapers[k], n=nfft)
        Sk.append((np.abs(Xk) ** 2) / (N / dt))
    Sk = np.array(Sk)

    if adaptive:
        wk = np.tile(eigs[:, None], (1, Sk.shape[1]))
        wk = wk / np.sum(wk, axis=0, keepdims=True)
        for _ in range(5):
            S = np.sum(wk * Sk, axis=0)
            wk = (eigs[:, None] * S[None, :]) / np.maximum(S[None, :], 1e-12)
            wk = wk / np.sum(wk, axis=0, keepdims=True)
    else:
        wk = np.tile(eigs[:, None], (1, Sk.shape[1]))
        wk = wk / np.sum(wk, axis=0, keepdims=True)

    S_mtm = np.sum(wk * Sk, axis=0)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    return freqs, S_mtm

def red_noise_significance(y, freqs, K, dt=1.0):
    r1, var = estimate_ar1(y)
    S_red = ar1_red_noise_spectrum(var, r1, freqs * dt)
    dof = 2 * K
    sig95 = S_red * chi2.ppf(0.95, dof) / dof
    return sig95, r1


#if __name__ == "__main__":

#excel_path = "reconstructed streamflow.xlsx" 
excel_path = "Full Reconstruction.xlsx" 
df_raw = pd.read_excel(excel_path, sheet_name=0, header=0, skiprows =0)

years = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce")   
flow1 = pd.to_numeric(df_raw.iloc[:, 1], errors="coerce")
flow2 = pd.to_numeric(df_raw.iloc[:, 2], errors="coerce")
flow3 = pd.to_numeric(df_raw.iloc[:, 3], errors="coerce")
flow4 = pd.to_numeric(df_raw.iloc[:, 4], errors="coerce")

df = pd.DataFrame({"Year": years, "Flow1": flow1, "Flow2": flow2, "Flow3": flow3, "Flow4": flow4}).dropna()
df = df[(df["Year"] >= 1490) & (df["Year"] <= 1995)].drop_duplicates("Year").sort_values("Year")

full_years = np.arange(1490, 1995 + 1)
#flow_full = np.interp(full_years, df["Year"].values, df["Flow"].values)
    
dt, NW = 1.0, 3.0
K = 2 * int(NW) - 1  # =5

flows = [flow1, flow2, flow3, flow4]
labels = [
    "Stepwise Regression (Non-PCA)",
    "Stepwise Regression (PCA)",
    "K-Nearest Neighbor (PCA)",
    "Neural Network (PCA)"
]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

plt.figure(figsize=(8, 4.6))

for flow, label, color in zip(flows, labels, colors):
    freqs, S = mtm_psd(flow, dt=dt, NW=NW, K=K, adaptive=True)
    mask = (freqs > 0) & (freqs <= 0.5)
    freqs, S = freqs[mask], S[mask]

    plt.plot(freqs, S, lw=1.4, color=color, label=label)

    peaks, _ = find_peaks(S)
    top = peaks[np.argsort(S[peaks])[-6:]]
    for p in top[np.argsort(freqs[top])]:
        f = freqs[p]
        print(f"{label}: Peak at f={f:.4f} cpy  (~{1/f:.1f} yr)  Power={S[p]:.3f}")
        #plt.plot(f, S[p], "o", color=color, ms=4)  # mark the peak

plt.xlabel("Frequency (cycles / year)")
plt.ylabel("Power")
plt.title("MTM Spectra of Streamflow Reconstructions (NW=3, K=5)")
plt.xlim(0, 0.5)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




