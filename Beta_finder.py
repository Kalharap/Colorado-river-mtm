#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
from scipy.stats import linregress

def detrend_anomaly(y):
    x = np.arange(len(y))
    m, b = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    y = y - (m * x + b)
    return y - np.mean(y)

def mtm_psd(y, dt=1.0, NW=3.0, K=None, adaptive=True, nfft=None):
    y = np.asarray(y, float)
    N = len(y)
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(N)))
    if K is None:
        K = int(2 * NW - 1)  # 5 for NW=3
    tapers, eigs = dpss(N, NW, K, return_ratios=True)
    x = detrend_anomaly(y)

    Sk = []
    for k in range(K):
        Xk = np.fft.rfft(x * tapers[k], n=nfft)
        Sk.append((np.abs(Xk) ** 2) / (N / dt))  
    Sk = np.array(Sk)

    wk = eigs[:, None] / np.sum(eigs)
    S = np.sum(wk * Sk, axis=0)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    return freqs, S

# ---------- Log-binning utility ----------
def logbin(x, y, nbins=14, xmin=None, xmax=None):
    """Logarithmically bin (x,y) by x; return bin centers, mean(log10 y), stderr."""
    x = np.asarray(x); y = np.asarray(y)
    mask = (x > 0) & np.isfinite(y) & (y > 0)
    x = x[mask]; y = y[mask]

    if xmin is None: xmin = x.min()
    if xmax is None: xmax = x.max()

    # log-spaced bin edges
    edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    xc, ym, ys = [], [], []
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i+1])
        if m.sum() < 2:  # need >=2 to show errorbars
            continue
        # geometric mean for x center (typical on log axis)
        xc.append(10 ** np.mean(np.log10(x[m])))
        logs = np.log10(y[m])
        ym.append(np.mean(logs))
        ys.append(np.std(logs, ddof=1) / np.sqrt(m.sum()))  # standard error
    return np.array(xc), np.array(ym), np.array(ys)

# ---------- Main ----------
if __name__ == "__main__":
   
    #excel_path = "reconstructed streamflow.xlsx" 
    excel_path = "Full Reconstruction.xlsx"
    df_raw = pd.read_excel(excel_path, sheet_name=0, header=0)
    years = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce")
    flow  = pd.to_numeric(df_raw.iloc[:, 5], errors="coerce")
    df = pd.DataFrame({"Year": years, "Flow": flow}).dropna()
    df = df[(df["Year"] >= 1493) & (df["Year"] <= 1962)].drop_duplicates("Year").sort_values("Year")

   
    full_years = np.arange(1493, 1962 + 1)
    flow_full = np.interp(full_years, df["Year"].values, df["Flow"].values)

   
    freqs, S = mtm_psd(flow_full, dt=1.0, NW=3.0, K=5, adaptive=False)
   
    m = (freqs > 0) & (freqs <= 0.5)
    f, P = freqs[m], S[m]

    fmin = max(f.min(), 1.0/200.0)  
    fmax = 0.45
    xb, yb, yerr = logbin(f[(f>=fmin)&(f<=fmax)], P[(f>=fmin)&(f<=fmax)], nbins=14)

    res = linregress(np.log10(xb), yb)
    m_slope = res.slope
    a_inter  = res.intercept
    beta = -m_slope  # since P ~ f^{-beta}

   
    plt.figure(figsize=(6, 4.2))
    plt.errorbar(np.log10(xb), yb, yerr=yerr, fmt='k-', lw=1.5, capsize=3)
    xx = np.linspace(np.log10(xb.min()), np.log10(xb.max()), 200)
    yy = a_inter + m_slope * xx
    plt.plot(xx, yy, '--', lw=1.5, color='gray')

    plt.xlabel("log frequency")
    plt.ylabel("log power spectral density")
    plt.title(f"log–log PSD")
    plt.tight_layout()
    plt.show()

    print(f"Slope m = {m_slope:.3f}  =>  beta ≈ {-m_slope:.3f}")

