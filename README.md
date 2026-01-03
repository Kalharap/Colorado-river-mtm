# Colorado River Streamflow Reconstruction

This repository contains Python scripts and datasets used to reconstruct the natural streamflow of the Colorado River using multiple statistical and machine-learning approaches. The reconstruction is based on paleoclimate tree-ring proxy data. In addition, the repository implements a Gaussian Process–based framework to generate probabilistic decadal streamflow predictions for the Colorado River, with an emphasis on low-frequency variability and long-term persistence. 


---

##  Overview

Four different approaches were applied to reconstruct annual streamflow using tree-ring chronologies and principal components:

1. **Stepwise Linear Regression** – baseline statistical reconstruction.
2. **PCA Regression** - Dimenstion reduction regression
3. **K-Nearest Neighbors (KNN)** – non-parametric method for capturing local relationships.
4. **Neural Network (NN)** – nonlinear approach for improved pattern recognition.
   

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


