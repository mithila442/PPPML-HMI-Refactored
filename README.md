# Federated Learning on RADChestCT with Privacy-Preserving Aggregation

This repository implements privacy-preserving federated learning (FL) algorithms using the **PPPML-HMI** framework on the **RADChestCT** dataset. The core method explored is **CSAPerAvg**, which combines personalized updates and secure aggregation via homomorphic encryption (HE) for decentralized medical AI training.

---

## Overview

**Goal:** Binary classification of lung infection presence from 3D chest CT scans under *non-IID* settings.

**FL Modes Implemented:**
- `FedProx`: Regularized local updates
- `FedOpt`: Server-side adaptive optimization (Adam)
- `CSAPerAvg`: Personalized + encrypted aggregation

---

## Dataset: RADChestCT

- **Modality:** 3D CT volumes (`.npz` originally, converted to `.npy`)
- **Label:** Binary – infection present or not
- **Clients:** 3 simulated clients with varying slice thickness (2mm, 5mm, 10mm)

| Split     | Clients | Slice Thickness       | Label Imbalance         |
|-----------|---------|------------------------|--------------------------|
| c3_split1 | A, B, C | 2mm, 5mm, 10mm         | Non-IID (heterogeneous) |

---

## Key Dependencies

- `torch`
- `numpy`
- `pandas`
- `tenseal`
- `scikit-learn`
- `matplotlib`

---

## Running the Code

```bash
python3 -u TrainPPPML.py \
  --dataset RAD \
  --algorithm CSAPerAvg \
  --batch_size 5 \
  --learning_rate 0.001 \
  --local_epochs 10 \
  --optimizer Adam \
  --numusers 3 \
  --num_global_iters 30 \
  --gpu 0 \
  --times 1 \
  --exp c3_split1
  --HE 1

## Key Insights

- **FedOpt** underperforms due to client drift and lack of personalization.
- **FedProx** improves convergence with a proximal term.
- **CSAPerAvg** shows best accuracy due to dual-phase updates and encrypted aggregation.

---

## Folder Structure

```bash
.
├── TrainPPPML.py
├── models/
├── dataset/
│   └── my_rescaled/     # Contains converted .npy CT volumes
├── result/
│   └── c3_split1/       # Stores checkpoints & logs
├── environment.yml
└── README.md
