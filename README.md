# phan-tich-du-lieu-lon-mid-term-features

EE104 Feature Engineering — Group Presentation Project

## Project Structure

```
phan-tich-du-lieu-lon-mid-term-features/
│
├── README.md                                    # This file - project overview
│
├── docs/
│   ├── features.pdf                             # Source: EE104 lecture slides (Stanford)
│   └── project-guide.md                         # Team task assignments & timeline
│
├── slides/
│   ├── part1-raw-data-to-vectors/
│   │   └── README.md                            # Person A: Raw Data → Vectors slides
│   │
│   ├── part2-feature-engineering/
│   │   └── README.md                            # Person B: Feature Engineering slides
│   │
│   └── final-presentation.[pptx|pdf]            # Merged final presentation deck
│
└── notebooks/
    ├── demo-structure.md                        # Implementation guide for all 3 notebooks
    ├── requirements.txt                         # Python dependencies
    │
    ├── feature-transforms-demo.ipynb            # Notebook 1: Core transforms (Person C)
    ├── word2vec-demo.ipynb                      # Notebook 2: Automatic embeddings
    ├── stock-price-features.ipynb               # Notebook 3: Real-world application
    │
    └── outputs/                                 # Saved plot images (if needed)
        └── (generated plots)
```

## Notebooks Overview

### 1. `feature-transforms-demo.ipynb` (Main Deliverable)
Demonstrates key feature engineering transforms from EE104 lecture:
- Standardization (Z-scoring)
- Gamma Transform (γ = 0.5, 1.5)
- Clipping/Winsorizing
- Powers Transform (degree 1, 2, 3)
- Positive/Negative Part Split
- Day-of-Week Circular Embedding

**Runtime**: ~2-3 minutes | **Focus**: Visual demonstrations

### 2. `word2vec-demo.ipynb` (Optional/Advanced)
Demonstrates automatic feature learning from text:
- Word similarity and nearest neighbors
- Vector arithmetic and analogies
- 2D visualization of embeddings

**Runtime**: ~5-10 minutes (includes model download) | **Focus**: Automatic embeddings

### 3. `stock-price-features.ipynb` (Application)
Applies all transforms to stock price forecasting:
- Real data from Yahoo Finance
- Systematically applies each transform from the lecture
- Shows complete feature engineering pipeline
- **Scope**: Feature engineering only (no model training)

**Runtime**: ~3-5 minutes | **Focus**: Real-world application

## How to Run

1. Install dependencies:
   ```bash
   cd notebooks/
   pip install -r requirements.txt
   ```

2. Run notebooks in order:
   - Start with `feature-transforms-demo.ipynb` (core concepts)
   - Optional: `word2vec-demo.ipynb` (if time permits during presentation)
   - Finish with `stock-price-features.ipynb` (real-world demo)

3. All notebooks should run top-to-bottom without errors after clean kernel restart

## Implementation Status

- [x] Demo structure documentation (3 notebooks planned)
- [x] Feature Transforms Demo implementation (16 cells)
- [x] Word2Vec Demo implementation (14 cells)
- [x] Stock Price Features implementation (22 cells)
- [ ] Final presentation slides merged

**✅ All notebooks ready for testing and presentation!**

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for complete details.

---

# How to use this project?

Alpha (Α, α) - 	 An-pha
Beta (Β, β) - 	 Bê-ta
Gamma (Γ, γ) - 	 Gam-ma
Delta (Δ, δ) - 	 Đen-ta
Epsilon (Ε, ε) - 	 Ep-si-lon
Zeta (Ζ, ζ) - 	 Dê-ta
Eta (Η, η) - 	 Ê-ta
Theta (Θ, θ) - 	 Thê-ta
Iota (Ι, ι) - 	 I-ô-ta
Kappa (Κ, κ) - 	 Cap-pa
Lambda (Λ, λ) - 	 Lam-đa
Mu (Μ, μ) - 	 Mu
Nu (Ν, ν) - 	 Nu
Xi (Ξ, ξ) - 	 Xi
Omicron (Ο, ο) - 	 Ô-mi-cron
Pi (Π, π) - 	 Pi
Rho (Ρ, ρ) - 	 Rô
Sigma (Σ, σ/ς) - 	 Sig-ma
Tau (Τ, τ) - 	 Tao
Upsilon (Υ, υ) - 	 Úp-si-lon
Phi (Φ, φ) - 	 Phi
Chi (Χ, χ) - 	 Khí
Psi (Ψ, ψ) - 	 Psi
Omega (Ω, ω) - 	 Ô-mê-ga