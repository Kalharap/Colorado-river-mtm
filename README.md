# Colorado River Streamflow Reconstruction

This repository contains Python scripts and data used to reconstruct natural streamflow of the **Colorado River** using multiple statistical and machine learning methods.  
The analyses were conducted as part of my PhD research at the University of Nevada, Las Vegas.

---

## 🌊 Overview

Three different approaches were applied to reconstruct annual streamflow using tree-ring chronologies and principal components:

1. **Stepwise Linear Regression** – baseline statistical reconstruction.
2. **K-Nearest Neighbors (KNN)** – non-parametric method for capturing local relationships.
3. **Neural Network (NN)** – nonlinear approach for improved pattern recognition.

Each method produces a complete time series reconstruction from **1490 to 1997**.

---

## 📈 Spectral and Scaling Analysis

- The file **`mtm_loglog_psd.py`** estimates the **power-law spectral slope (β)** of the reconstructed streamflow using the **Multitaper Method (MTM)** with a 3×2π taper (time–bandwidth product NW=3).  
- The resulting β quantifies the long-memory and scaling behavior of streamflow variability.

---

## 🤖 Gaussian Process Model

- The **`gaussian_process_reconstruction.py`** (planned or included) uses **Gaussian Process Regression (GPR)** to model uncertainty in the reconstructions.  
- This approach captures probabilistic trends and confidence intervals for each reconstructed year, allowing comparison with deterministic methods.

---

## 📂 File List

| File | Description |
|------|--------------|
| `mtm_spectrum.py` | Full multitaper spectral analysis with AR(1) red-noise significance test |
| `mtm_loglog_psd.py` | Computes log–log MTM spectrum and fits β slope |
| `stepwise_reconstruction.py` | Stepwise linear regression model for streamflow reconstruction |
| `knn_reconstruction.py` | K-Nearest Neighbor reconstruction with PCA inputs |
| `neural_network_reconstruction.py` | Neural network model for nonlinear streamflow reconstruction |
| `gaussian_process_reconstruction.py` | Gaussian Process regression model for uncertainty analysis |
| `reconstructed_streamflow.xlsx` | Data file containing annual reconstructed flows (1490–1997) |

---

## ⚙️ Requirements

All scripts are written in **Python 3.9+** and use the following main libraries:
```bash
numpy
pandas
matplotlib
scipy
scikit-learn
