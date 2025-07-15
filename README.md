# 🧠 Official Implementation of Deep Learning-Enabled Single Breath-Hold Abbreviated MRI (DL-SBH-aMRI)

This repository contains the **official implementation** of the project:

> **Development and Validation of a Deep Learning-Enabled Single Breath-Hold Abbreviated MRI (DL-SBH-aMRI) for Hepatocellular Carcinoma Diagnosis: A Multicenter, Prospective, and Retrospective Study**

---

## 📄 Table of Contents

- [Overview](#overview)
- [Representative Results](#representative-results)
- [Citation](#citation)
- [License](#license)

---

## 🧬 Overview

This is the official implementation of  **Deep Learning–Enabled Single Breath-Hold Abbreviated MRI (DL-SBH-aMRI)** protocol for HCC diagnosis.

**Objective:** The goal is to develop a Deep Learning–Enabled Single Breath-Hold Abbreviated MRI (DL-SBH-aMRI) protocol for HCC diagnosis.

**Conclusion:** DL-SBH-aMRI enables gadolinium-free, ultra-fast imaging within a single breath-hold, while preserving full-sequence diagnostic information and achieving performance comparable to conventional MRI, representing a promising and cost-effective alternative for HCC diagnosis.

---

## 🖼 Representative Results

The figure below presents representative **comparisons between DL-SBH-aMRI and complete full-sequence MRI (cMRI)** across four common hepatic lesion types.

<img src="https://github.com/yunfei920406/DL-SBH-aMRI/blob/main/Some%20Representative%20Images/Case.jpg" alt="Representative Cases" width="100%">

### 🔍 Description

- Each panel (**A–D**) shows:
  - **Top row**: Ground-truth cMRI (all sequences acquired).
  - **Bottom row**: DL-SBH-aMRI results (only Pre-T1 acquired; others synthesized by **Li-DiffNet**).
- The same Pre-T1 image is used for alignment across sequences.
- **Lesion types**:
  - **A**: Hepatocellular carcinoma (HCC)
  - **B**: Hepatic hemangioma
  - **C**: Intrahepatic cholangiocarcinoma (ICC)
  - **D**: Hepatic cyst
- **Yellow arrows** mark lesion locations.
- All cases are from the **external validation cohort**.

---

## 📚 Citation

If you use this code or data in your work, please cite:

```bibtex
@article{YourCitation2024,
  title     = {Development and Validation of a Deep Learning-Enabled Single Breath-Hold Abbreviated MRI (DL-SBH-aMRI) for Hepatocellular Carcinoma Diagnosis: A Multicenter, Prospective, and Retrospective Study},
  author    = {Yunfei et al},
  journal   = {Submitted and In Review},
  year      = {2025},
  note      = {Available at: https://github.com/yunfei920406/DL-SBH-aMRI}
}
