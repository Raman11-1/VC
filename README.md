# Visual Computing (VC) — SEM 2 Final Project
### IIT Hyderabad | M.Tech CSE | Course: Visual Computing
**Student:** Raman S. Mankar (`cs25mtech14025@iith.ac.in`)  
**Course Instructor:** Prof. C. Krishna Mohan | **TA:** Ankita Das  
**GitHub:** [https://github.com/Raman11-1/VC](https://github.com/Raman11-1/VC)

---

##  Project Overview

This repository contains the complete implementation for two parts:

| Part | Title | Description |
|------|-------|-------------|
| **Base Paper** | *Vision Transformers with Self-Distilled Registers (PH-Reg)* | Reproduction and implementation of Post Hoc Registers for ViT artifact token removal |
| **Novelty** | *Frequency-Domain Augmentation & Artifact-Aware Adaptive Loss for Post-Hoc Register Learning* | Two principled extensions to PH-Reg: FDA + AAAL |

---

##  Repository Structure

```
VC/
│
├── phreg-final.ipynb            # Base paper implementation (PH-Reg reproduction)
├── phreg-novelity.ipynb         # Novelty implementation (FDA + AAAL)
│
├── image/
│   ├── Base_Paper/              # Result images from base paper implementation
│   │   ├── image1.png           # UMAP feature visualization
│   │   ├── image2.png           # Heatmap comparisons
│   │   ├── image3.png           # Artifact token analysis
│   │   ├── image4.png           # Segmentation results
│   │   ├── image5.png           # Feature maps
│   │   ├── image6.png           # Training curves
│   │   ├── image7.png           # Qualitative results
│   │   └── image8.png           # Additional results
│   │
│   └── novelity/                # Result images from novelty implementation
│       ├── image1.png           # FDA augmentation comparison
│       ├── image2.png           # AAAL loss weight distribution
│       ├── image3.png           # Feature quality comparison
│       └── image4.png           # Artifact score heatmaps
│
├── Novelity_Report.pdf          # Full novelty report (13 pages)
├── 9827_Vision_Transformers_with_.pdf   # Base paper (PH-Reg) — Yan et al.
├── Vc_Final_Presentation.pdf    # Base paper presentation slides
└── VC_Novelity_Presentation.pdf # Novelty presentation slides
```

---

##  Part 1: Base Paper — PH-Reg

### Paper Reference
> **"Vision Transformers with Self-Distilled Registers"**  
> Zipeng Yan*, Yinjie Chen*, Chong Zhou, Bo Dai, Andrew F. Luo  
> University of Hong Kong / Zhejiang University / NTU  
> [Paper PDF included: `9827_Vision_Transformers_with_.pdf`]

### What is PH-Reg?
Vision Transformers (ViTs) generate **artifact tokens** — patch features that are semantically incongruent with their local image content, corrupting dense representations and degrading tasks like segmentation and depth estimation.

PH-Reg solves this via **self-distillation**:
- A **frozen teacher** ViT provides clean targets using spatial-shift augmentation (averaging over n shifted views)
- A **student** ViT with 16 learnable register tokens is trained to match the teacher's clean features
- No labeled data or full retraining needed — only register tokens + positional embedding + last attention block are trainable

### Notebook: `phreg-final.ipynb`

#### Environment Setup (Kaggle)
```python
!pip install open-clip-torch==2.31.0
!pip install transformers==4.37.2
!pip install timm accelerate diffusers
!pip install numpy==1.26.4
!pip install scikit-image scikit-learn imageio opencv-python
!pip install umap-learn numba ftfy regex tqdm
```

#### Dataset
| Dataset | Source | How to Access |
|---------|--------|---------------|
| **Flickr30k** | Kaggle | `ramansss/vt-ph-reg` → `flickr30k-images/` |

> Dataset path on Kaggle: `/kaggle/input/datasets/ramansss/vt-ph-reg/flickr30k-images`  
>  Dataset is **NOT included** in this repo. Access via Kaggle dataset above.

#### Pre-trained Model
| Model | Source | Kaggle Path |
|-------|--------|-------------|
| `OpenAI_CLIP_B_16_Distilled.pth` | Kaggle | `ramanmankar/open-clip/pytorch/default/1/` |

> Model path: `/kaggle/input/models/ramanmankar/open-clip/pytorch/default/1/OpenAI_CLIP_B_16_Distilled.pth`

#### Key Configuration
```python
PRETRAINED_NAME = "ViT-B/16"      # Base CLIP model
RESOLUTION      = 448             # Input resolution
PATCH_SIZE      = 16              # ViT patch size
NUM_PATCHES     = RESOLUTION // PATCH_SIZE  # = 28x28 = 784
NUM_REGISTERS   = 16              # Register tokens
DEVICE          = "cuda"          # Requires GPU
```

#### How to Run
1. Open `phreg-final.ipynb` on **Kaggle** (GPU T4 x2 recommended)
2. Add the Flickr30k dataset from `ramansss/vt-ph-reg`
3. Add the distilled model from `ramanmankar/open-clip`
4. Clone PH-Reg codebase (handled inside notebook):
   ```bash
   !git clone https://github.com/0raiser0/PH-Reg.git
   ```
5. Run cells sequentially

---

##  Part 2: Novelty — FDA + AAAL

### Title
**"Frequency-Domain Augmentation and Artifact-Aware Adaptive Loss for Post-Hoc Register Learning in Vision Transformers"**

### Motivation & Key Problems Solved

| Problem in PH-Reg | Our Solution |
|-------------------|--------------|
| Spatial shifts leave low-frequency artifacts under-perturbed | **FDA**: Perturb phase spectrum in Fourier domain — equally covers all frequencies |
| Uniform loss wastes gradients on ~80% clean patches | **AAAL**: Per-patch weighted loss, focusing on artifact-prone tokens |

### Novel Contribution 1: Frequency-Domain Augmentation (FDA)

**Core Insight:** The 2D DFT decomposes an image as `Î = A · e^(jΦ)` where:
- **Amplitude A** = semantic content → **preserved**
- **Phase Φ** = positional structure (where artifacts originate) → **perturbed**

```
Augmented image: I_FDA = F⁻¹(A · e^j(Φ + ε))
where ε ~ U(-0.15π, 0.15π), low-frequency circle (r=0.10) protected
```

**Result:** Achieves cosine similarity **1.000** to clean reference vs **0.972–0.986** for spatial shifts.

### Novel Contribution 2: Artifact-Aware Adaptive Loss (AAAL)

**Two-part artifact scoring per patch:**
1. **Norm Deviation Score** — Median Absolute Deviation (MAD) detects outliers in both directions (handles both DINOv2 high-norm and CLIP low-norm artifacts)
2. **Local Inconsistency Score** — Cosine distance from 4-neighbor average in spatial grid

```
artifact_score = 0.5 × norm_deviation + 0.5 × local_inconsistency
weight_i = 1 + γ × artifact_score_i     (γ = 2.0)
Loss = Σ(weight_i × per_patch_loss_i) / Σ(weight_i)
```

### Notebook: `phreg-novelity.ipynb`

#### Environment Setup
```python
!pip install git+https://github.com/openai/CLIP.git -q
!pip install open_clip_torch ftfy regex tqdm -q
!pip install umap-learn numba -q
```

#### Dataset
| Dataset | Source | Purpose |
|---------|--------|---------|
| **CIFAR-10** | Auto-downloaded via `torchvision` | Training proxy (500-image subset for demo) |

```python
# CIFAR-10 auto-downloads on first run
dataset = datasets.CIFAR10(root="/kaggle/working/data", train=True, download=True, ...)
```
>  No manual dataset download needed for the novelty notebook — CIFAR-10 is fetched automatically.

#### Key Configuration
```python
PATCH_H      = 14       # 224 / 16
PATCH_W      = 14
N_PATCHES    = 196      # 14 × 14
EMBED_DIM    = 768      # ViT-B hidden dim
N_REGISTERS  = 16

# FDA hyperparameters
phase_noise_scale = 0.15
low_freq_protect  = 0.10

# AAAL hyperparameters
gamma = 2.0   # artifact weight amplifier
alpha = 0.5   # balance norm-deviation vs local-inconsistency
```

#### Student Model Architecture
```
Input Image
    ↓
CLIP ViT-B/16 Patch Embedding (conv1)
    ↓
[CLS | Register×16 | Patch Tokens×196]
    ↓
Transformer (last block unfrozen)
    ↓
Patch Tokens (196 × 768)  ← output
```

**Trainable parameters only:**
- Register tokens (16 × 768)
- Positional embeddings
- Conv patch embedding (conv1)
- Last transformer block

#### How to Run
1. Open `phreg-novelity.ipynb` on **Kaggle** (GPU P100 or T4 recommended)
2. No external dataset needed — CIFAR-10 auto-downloads
3. Run all cells sequentially
4. Training loop runs for 3 epochs with full logging

---

##  Hardware & Software Requirements

| Component | Requirement |
|-----------|-------------|
| Platform | Kaggle Notebooks (recommended) |
| GPU | NVIDIA T4 or P100 (16GB VRAM) |
| Python | 3.10+ |
| PyTorch | 2.x |
| CUDA | 11.8+ |

---

##  Results Summary

### Base Paper (PH-Reg Reproduction)
- Successfully reproduced PH-Reg pipeline on Flickr30k
- Clean UMAP feature maps with visible artifact reduction
- Dense feature heatmaps showing improved text-image localization

### Novelty (FDA + AAAL)
- **FDA cosine similarity:** 1.000 (vs 0.972–0.986 for spatial shifts)
- **AAAL mean artifact score:** 0.19–0.22 per batch
- Focused gradient updates on artifact-prone patches (~20% of tokens)
- Cleaner feature representations on CIFAR-10 proxy

---

##  Submitted Files

| File | Type | Description |
|------|------|-------------|
| `phreg-final.ipynb` | Jupyter Notebook | Base paper implementation |
| `phreg-novelity.ipynb` | Jupyter Notebook | Novelty implementation |
| `Vc_Final_Presentation.pdf` | PDF | Base paper presentation |
| `VC_Novelity_Presentation.pdf` | PDF | Novelty presentation |
| `Novelity_Report.pdf` | PDF | Full novelty technical report |
| `9827_Vision_Transformers_with_.pdf` | PDF | Base paper (PH-Reg) |
| `image/Base_Paper/` | Images | Base paper result visualizations |
| `image/novelity/` | Images | Novelty result visualizations |

---

##  References

1. **Yan et al. (PH-Reg)** — "Vision Transformers with Self-Distilled Registers" *(Base Paper)*
2. **Darcet et al.** — "Vision Transformers Need Registers" — characterized artifact tokens in DINOv2
3. **Wang et al. (SINDER)** — Smoothness prior for DINOv2 outlier repair (high-norm only)
4. **Yang et al. (DVT)** — Dynamic artifact modeling via neural fields
5. **MaskCLIP, SCLIP, ClearCLIP, NACLIP** — Training-free dense CLIP feature improvements
6. **Yang & Soatto** — FDA-style domain adaptation via amplitude spectrum swapping

---

##  Links

- **GitHub Repository:** [https://github.com/Raman11-1/VC](https://github.com/Raman11-1/VC)
- **PH-Reg Official Code:** [https://github.com/0raiser0/PH-Reg](https://github.com/0raiser0/PH-Reg)
- **Kaggle Dataset:** `ramansss/vt-ph-reg` (Flickr30k)
- **Kaggle Model:** `ramanmankar/open-clip` (Distilled CLIP B/16)

---

*IIT Hyderabad — M.Tech CSE — Visual Computing — SEM 2 — April 2026*
