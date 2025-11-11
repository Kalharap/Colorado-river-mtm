# Colorado River Streamflow Reconstruction

This repository contains Python scripts and data used to reconstruct the natural streamflow of the **Colorado River** using multiple statistical and machine learning methods.  


---

##  Overview

Three different approaches were applied to reconstruct annual streamflow using tree-ring chronologies and principal components:

1. **Stepwise Linear Regression** – baseline statistical reconstruction.
2. **K-Nearest Neighbors (KNN)** – non-parametric method for capturing local relationships.
3. **Neural Network (NN)** – nonlinear approach for improved pattern recognition.

Each method produces a complete time series reconstruction from **1490 to 1995**.

---

##  Spectral and Scaling Analysis

- The file **`Multi-Taper Spectral.py`** estimates the **power-law spectral slope (β)** of the reconstructed streamflow using the **Multitaper Method (MTM)** with a 3×2π taper (time–bandwidth product NW=3).  
- The resulting β quantifies the long-memory and scaling behavior of streamflow variability.

---

## Gaussian Process Model

- The **`gaussian_process_reconstruction.py`** uses **Gaussian Process Regression (GPR)** to model uncertainty in the reconstructions.  
- This approach captures probabilistic trends and confidence intervals for each reconstructed year, allowing comparison with deterministic methods.

---

##  File List

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
