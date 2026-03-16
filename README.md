c'est u# CNN-retinopathy — Diabetic Retinopathy Detection

**Project 5 — AI Based Image Processing | ING3 IA A, Group 6 | CY Tech**
Bezet Camille · Brasa Franklin · Kenmogne Loïc · Martins Soares Flavio
*3 December 2025 — submitted to M. Djahid ABDELMOUMENE and Pr. Mai khuong NGUYEN VERGER*

---

## Context

Diabetic Retinopathy (DR) is one of the leading causes of blindness among the working-age population, affecting ~40% of the 425 million diabetics worldwide. Early diagnosis at the Non-Proliferative stage (NPDR) is essential to prevent irreversible vision loss.

> See **Report § Introduction** and **Slides 2–3** for the full medical context (disease mechanism, NPDR/PDR stages, detectable anomalies: microaneurysms, haemorrhages, exudates).

---

## Objective

Faithfully reproduce and evaluate the CNN architecture proposed by **Abed et al. (2020)** — *"Diabetic Retinopathy Diagnosis based on Convolutional Neural Network"* — for binary classification of fundus images (healthy / pathological), then explore extensions.

> See **Report § Project Objective** and **Slide 5**.

---

## Data

| Dataset   | Images | Resolution   | Labels                                  |
|-----------|--------|--------------|-----------------------------------------|
| DiaretDB0 | 130    | 1500×1152 px | Annotated lesions (MA, haemorrhages…)   |
| DiaretDB1 | 89     | 1500×1152 px | All pathological                        |
| DRIMDB    | 216    | 570×760 px   | Image quality (good / bad / outlier)    |

Only **DiaretDB0** is included in this repository (`Datasets/DiaretDB0/`).

> See **Report § Data** (pp. 10–14) and **Slide 6** for technical characteristics and limitations of each dataset (class imbalance, no healthy images in DB1, non-medical nature of DRIMDB).

---

## Preprocessing Pipeline

Implemented in [`CNN_Binaire-sain-pas-sain.ipynb`](CNN_Binaire-sain-pas-sain.ipynb):

1. **Loading & Conversion** — BGR → RGB (`cv2.cvtColor`)
2. **Cropping** — remove dark borders (`crop_image_from_gray`, threshold tol=7)
3. **Resizing** — 224×224 px (`cv2.resize`)
4. **CLAHE** — contrast enhancement on the L channel (LAB colour space, `cv2.createCLAHE`)
5. **Normalisation** — pixel values scaled to [0, 1] (`/ 255.0`)

> See **Report § Pipeline Description** (pp. 15–16) and **Slide 7**.

---

## CNN Architecture

Lightweight sequential architecture inspired by Abed et al., implemented in **Keras/TensorFlow**:

```
Input (224, 224, 3)
  → Conv Block 1 : Conv(8, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
  → Conv Block 2 : Conv(16, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
  → Conv Block 3 : Conv(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
  → Flatten
  → Dense(64, ReLU) + Dropout(0.3)
  → Dense(2, Softmax)
Optimizer : SGD (lr=0.01, momentum=0.9) | Loss : Categorical Crossentropy
```

> See **Report § CNN Architecture** (pp. 16–18) and **Slide 8**.

---

## Results — Healthy / Pathological Classification

80/20 train-test split, evaluated on all 3 datasets:

| Dataset   | Epochs | Abed et al. (2020) | Our project  |
|-----------|--------|--------------------|--------------|
| DiaretDB0 | 1      | 62.67%             | **84.6%**    |
| DiaretDB0 | 20     | 100%               | 84.62%       |
| DiaretDB1 | 1      | 57.9%              | **100%**     |
| DiaretDB1 | 20     | 99.1%              | 100%         |
| DRIMDB    | 1      | 68%                | 40.91%       |
| DRIMDB    | 20     | 100%               | 43.2%        |

**Warning — critical interpretation:** the high scores on DB0 (84.6%) and DB1 (100%) are misleading. The model learns the majority-class bias ("Pathological") rather than developing a genuine discriminative ability. The failure on DRIMDB (43.2% vs. 97.5% in the paper) reveals a methodological inconsistency in the original study (image quality labels ≠ clinical criteria).

> See **Report § Interpretation** (pp. 20–21) and **Slide 10**.

---

## Extension — Classification by "Haemorrhage" Label

To work around the class bias, the problem is reformulated as **binary haemorrhage detection** on DiaretDB0 (29 images with haemorrhages / 101 without).

Notebook: `Classification selon le label "hémorragies".ipynb`

| Epochs | Accuracy   | Loss     | Interpretation                        |
|--------|------------|----------|---------------------------------------|
| 1      | 73.08%     | 0.69     | Model just starting to learn          |
| **13** | **76.92%** | **0.59** | **Optimum — best generalisation**     |
| 20     | 69.23%     | 0.67     | Overfitting — performance degrades    |

The model peaks at 13 epochs then overfits, highlighting the importance of early stopping.

> See **Report § Possible Extensions** (pp. 22–24) and **Slides 11–13**.

---

## Technical Environment

- **Language:** Python
- **Framework:** TensorFlow / Keras
- **Environment:** Jupyter Notebook / Google Colab (GPU)
- **Libraries:** OpenCV, scikit-learn, matplotlib, numpy

> See **Report § Tools and Technical Environment** (p. 21) and **Slide 6**.

---

## Conclusion

This project validates the technical soundness of the CNN architecture from Abed et al., but more importantly highlights the **vulnerability of models to statistical biases** in small, imbalanced medical datasets. Data quality and rigorous problem formulation matter as much as the architecture itself.

> See **Report § Conclusion** (pp. 25–26) and **Slide 14**.

---

## Main Reference

> [15] M.H. Abed, L.A.N. Muhammed, S.H. Toman, *Diabetic Retinopathy Diagnosis based on Convolutional Neural Network*, arXiv:2008.00148, 2020.