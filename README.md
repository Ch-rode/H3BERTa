# H3BERTa: A CDR-H3 specific language model for antibody repertoire analysis
<p align="center">
  <img src="https://github.com/user-attachments/assets/c839ba2c-5958-4f2d-ad73-217839903dd1" width="500">
</p>


Zenodo record (DOI): `10.5281/zenodo.17505849`


This repository contains the code, pretrained models, and datasets used in the study:

> **Chiara Rodella & Thomas Lemmin**  
> *H3BERTa: A CDR-H3 specific language model for antibody repertoire analysis*  
> Institute of Biochemistry and Molecular Medicine, University of Bern (Switzerland)


This repository contains the code, configuration files, datasets, and analyses associated with the H3BERTa paper.
Large artifacts (datasets, trained weights, etc.) are stored on Zenodo and mirrored using the same directory structure as in this GitHub repo.

---

## Overview
```
H3BERTa/
├─ LM/               # H3BERTa language model training, data and analysis
├─ bnAbsclassifier/  # GAN-BERTa and SVM classifiers training and data for HIV-1 broadly neutralizing antibodies (bnAbs)
└─ README.md         # Project description
```
**H3BERTa** is a transformer-based protein language model trained exclusively on the **CDR-H3 region** of antibody heavy chains.  
By focusing on this short but highly variable segment, the model captures **immunologically meaningful sequence features** like:
- B-cell maturation states  
- Comparison of healthy vs. HIV-1 antibody repertoires  
- Classification of **broadly neutralizing antibodies (bnAbs)** using minimal labeled data  

Despite being trained only on CDR-H3 loops, H3BERTa reproduces biologically coherent repertoire structures and supports functional predictions.

---
