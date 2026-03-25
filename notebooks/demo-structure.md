# Demo Notebooks Structure Guide

This document provides a step-by-step guide for implementing the demo notebooks.

## Overview

The demo consists of **3 separate Jupyter notebooks**:

1. **`feature-transforms-demo.ipynb`** — Core feature engineering transforms (Person C's main deliverable)
2. **`word2vec-demo.ipynb`** — Word embeddings demonstration (optional/advanced)
3. **`stock-price-features.ipynb`** — Real-world application: Stock price forecasting with feature engineering

---

# Notebook 1: Feature Transforms Demo (`feature-transforms-demo.ipynb`)

## Setup Section

### Cell 1: Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
```

**Purpose**: Load all required libraries for data manipulation and visualization.

---

## Part 1: Basic Transformations

### Cell 2: Z-Scoring / Standardization
**Plot Title**: "Standardization (Z-Scoring)"

**Steps**:
1. Generate sample data: `data = np.random.normal(100, 15, 1000)`
2. Calculate z-scores: `z_scores = (data - np.mean(data)) / np.std(data)`
3. Create side-by-side histograms showing original vs standardized data
4. Add vertical lines for mean and ±1 standard deviation

**Visual Requirements**:
- Two subplots (1 row, 2 columns)
- Left: Original data distribution
- Right: Z-scored data distribution
- Include mean and std annotations

**Verbal Explanation**: "Z-scoring transforms data to have mean 0 and standard deviation 1, making different features comparable regardless of their original scale."

---

### Cell 3: Gamma Transform
**Plot Title**: "Gamma Transform (γ = 0.5 and γ = 1.5)"

**Steps**:
1. Generate input range: `x = np.linspace(0, 1, 100)`
2. Apply gamma transforms: `y_0.5 = x**0.5` and `y_1.5 = x**1.5`
3. Plot all three curves: original (γ=1), γ=0.5, γ=1.5
4. Use different colors/line styles for each

**Visual Requirements**:
- Single plot with three curves
- Legend indicating gamma values
- Grid for readability
- Equal aspect ratio

**Verbal Explanation**: "Gamma transform amplifies small values (γ < 1) or large values (γ > 1), useful for adjusting feature sensitivity in different ranges."

---

### Cell 4: Clipping / Winsorizing
**Plot Title**: "Clipping / Winsorizing"

**Steps**:
1. Generate data with outliers: `data = np.concatenate([np.random.normal(0, 1, 950), np.random.uniform(3, 5, 50)])`
2. Define clipping thresholds: lower = 5th percentile, upper = 95th percentile
3. Apply clipping: `clipped = np.clip(data, lower_threshold, upper_threshold)`
4. Create before/after comparison plots (histograms or scatter)

**Visual Requirements**:
- Two subplots showing original vs clipped distribution
- Highlight clipping boundaries with vertical lines
- Show outliers visually

**Verbal Explanation**: "Clipping limits extreme values to reduce the impact of outliers, preventing them from dominating the model."

---

### Cell 5: Powers Transform
**Plot Title**: "Powers Transform (degree 1, 2, 3)"

**Steps**:
1. Generate input data: `x = np.linspace(0, 2, 100)`
2. Compute powers: `x^1`, `x^2`, `x^3`
3. Plot all three on same axes
4. Optionally add negative powers (x^0.5) for comparison

**Visual Requirements**:
- Single plot with multiple curves
- Clear legend
- Grid enabled
- Different line styles/colors

**Verbal Explanation**: "Power transforms create polynomial features, capturing non-linear relationships between inputs and outputs."

---

## Part 2: Advanced Feature Engineering

### Cell 6: Positive/Negative Split
**Plot Title**: "Positive/Negative Part Split"

**Steps**:
1. Generate data spanning negative to positive: `x = np.linspace(-2, 2, 100)`
2. Create positive part: `x_pos = np.maximum(x, 0)`
3. Create negative part: `x_neg = np.minimum(x, 0)`
4. Plot original, positive part, and negative part

**Visual Requirements**:
- Three curves on same plot
- Use different colors (e.g., green for positive, red for negative, blue for original)
- Include horizontal line at y=0
- Legend

**Verbal Explanation**: "Splitting into positive and negative parts allows the model to learn different behaviors for positive vs negative values of a feature."

---

### Cell 7: Day-of-Week Circular Embedding
**Plot Title**: "Day-of-Week Circular Embedding"

**Steps**:
1. Create days array: `days = np.arange(0, 7)`
2. Convert to circular coordinates:
   - `x = np.cos(2 * np.pi * days / 7)`
   - `y = np.sin(2 * np.pi * days / 7)`
3. Plot points on unit circle
4. Add day labels (Mon, Tue, Wed, etc.)

**Visual Requirements**:
- Scatter plot on circular arrangement
- Equal aspect ratio to show circle properly
- Annotate each point with day name
- Draw unit circle for reference

**Verbal Explanation**: "Circular embedding represents cyclic features like days of the week without imposing false ordering, ensuring Monday and Sunday are treated as adjacent."

---

## Summary Section

### Cell 8: Summary
**Content**:
- List all transforms demonstrated
- Brief recap of when to use each
- Note that Word2Vec is covered in separate notebook
- Stock price forecasting application is in separate notebook

---

## Implementation Checklist (Notebook 1)

- [ ] All cells run from top to bottom without errors
- [ ] Each plot has a title matching the list above
- [ ] All plots are clearly labeled (axes, legends)
- [ ] Verbal explanations prepared (2-3 sentences each)
- [ ] Figure sizes are consistent and readable
- [ ] Code is commented for clarity
- [ ] No hardcoded paths (use relative paths if loading data)
- [ ] Test on clean kernel restart before final submission

---

## Style Guidelines (All Notebooks)

**Matplotlib Settings**:
```python
plt.rcParams['figure.figsize'] = (10, 4)  # Adjust as needed
plt.rcParams['font.size'] = 12
plt.style.use('default')  # Or choose consistent style
```

**Color Scheme**: Use consistent, colorblind-friendly colors throughout.

**Plot Quality**: Use `plt.tight_layout()` before `plt.show()` to prevent label overlap.

---

# Notebook 2: Word2Vec Demo (`word2vec-demo.ipynb`)

## Purpose
Demonstrate automatic feature learning from text using Word2Vec embeddings.

## Cell 1: Introduction
**Content**: 
- Markdown explaining Word2Vec as automatic feature engineering
- Connection to lecture: embeddings learned from data, not hand-crafted

## Cell 2: Setup & Dependencies
```python
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
```

## Cell 3: Load Pretrained Model
**Steps**:
1. Use gensim downloader to get pretrained embeddings (e.g., `glove-wiki-gigaword-100`)
2. Alternative: `word2vec-google-news-300` (larger, better quality)
3. Print model info (vocabulary size, vector dimensions)

**Code Example**:
```python
# Option 1: Lighter model (faster download)
model = api.load("glove-wiki-gigaword-100")

# Option 2: Better quality (larger size)
# model = api.load("word2vec-google-news-300")
```

**Note**: If download fails or takes > 15 minutes, provide fallback instructions.

## Cell 4: Word Similarity Examples
**Title**: "Semantic Similarity Scores"

**Examples**:
```python
examples = [
    ('king', 'queen'),
    ('king', 'man'),
    ('king', 'car'),
    ('computer', 'laptop'),
    ('france', 'paris'),
    ('run', 'walk')
]
```

**Output**: Table showing similarity scores (0-1 scale)

**Verbal Explanation**: "Cosine similarity between word vectors captures semantic relatedness. Similar words have high scores (close to 1), unrelated words have low scores."

## Cell 5: Most Similar Words
**Title**: "Nearest Neighbors in Embedding Space"

**Steps**:
1. Pick 3-4 query words: `['king', 'computer', 'happy']`
2. For each, show top 5 most similar words with scores
3. Format as DataFrame for readability

**Visual**: Display formatted table for each query word

## Cell 6: Analogy Mathematics
**Title**: "Vector Arithmetic: king - man + woman = ?"

**Steps**:
1. Compute: `king - man + woman`
2. Find nearest word to result
3. Show classic analogies:
   - king - man + woman ≈ queen
   - paris - france + germany ≈ berlin
   - good - bad + slow ≈ fast

**Verbal Explanation**: "Word embeddings encode semantic relationships as vector operations. Analogies emerge naturally from the learned representations."

## Cell 7: Visualization (2D PCA)
**Title**: "Word Embeddings Projected to 2D"

**Steps**:
1. Select word groups:
   - Countries: ['france', 'germany', 'italy', 'spain', 'japan']
   - Capitals: ['paris', 'berlin', 'rome', 'madrid', 'tokyo']
   - Animals: ['dog', 'cat', 'lion', 'tiger', 'elephant']
2. Extract embeddings, apply PCA to 2D
3. Scatter plot with color-coded groups
4. Annotate each point with word label

**Visual Requirements**:
- Different colors for each group
- Legend
- Clear labels
- Use `adjustText` or manual positioning to avoid overlap

## Cell 8: Summary
**Content**:
- Word2Vec learns semantic relationships automatically from text co-occurrence
- No manual feature engineering needed for text data
- Embeddings capture meaning through distributional patterns
- Applications: text classification, sentiment analysis, recommendation systems
- **Connection to lecture**: Demonstrates automatic feature engineering from unstructured data

**Verbal Explanation**: "Word2Vec demonstrates that powerful features can be learned automatically from data structure, rather than hand-engineered. This principle extends to images (CNN features), graphs, and other domains."

---

# Notebook 3: Stock Price Forecasting Features (`stock-price-features.ipynb`)

## Purpose
Apply feature engineering transforms from the lecture to a real-world time series problem.

**Scope**: Focus ONLY on feature engineering, not model training (keep it simple and aligned with slide contents).

## Cell 1: Introduction
**Content**:
- Markdown explaining the task: predict tomorrow's stock price movement (up/down)
- Features we'll engineer using transforms from the lecture
- Data source: Yahoo Finance (single stock, e.g., AAPL, 2020-2024)
- **Explicit connection**: "We apply standardization, gamma transform, pos/neg split, binning, and cyclic encoding from the EE104 lecture."

## Cell 2: Setup & Load Data
```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set plotting defaults
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 11
```

**Steps**:
1. Download stock data: `yf.download('AAPL', start='2020-01-01', end='2024-12-31')`
2. Extract relevant columns: Open, High, Low, Close, Volume
3. Display first/last 5 rows
4. Plot raw closing price over time

## Cell 3: Raw Features
**Title**: "Extract Basic Raw Features"

**Steps**:
1. Daily return: `(Close - Close.shift(1)) / Close.shift(1)`
2. Price range: `High - Low`
3. Volume change: `(Volume - Volume.shift(1)) / Volume.shift(1)`
4. Moving averages: 5-day, 20-day MA of Close
5. Create DataFrame with all raw features

**Output**: Display statistics (mean, std, min, max) for each feature

**Verbal Explanation**: "Before engineering, we extract raw features from OHLCV data. These will be transformed in subsequent steps."

## Cell 4: Transform 1 - Standardization
**Title**: "Z-Score Normalization (Slide Transform #1)"

**Steps**:
1. Apply z-scoring to daily return and volume change
2. Plot before/after histograms (side-by-side)
3. Add textbox showing mean≈0, std≈1 after transform

**Connection to Lecture**: "Makes features comparable across different scales. Same transform shown in slide examples."

## Cell 5: Transform 2 - Gamma Transform
**Title**: "Gamma Transform on Returns (γ = 0.5) (Slide Transform #2)"

**Steps**:
1. Apply gamma=0.5 to returns: `sign(return) * abs(return)^0.5`
2. Plot original vs transformed distribution
3. Explain: "Reduces impact of extreme returns, makes small movements more visible"

**Visual**: Overlay histograms with transparency

**Connection to Lecture**: "Gamma < 1 amplifies small values, as shown in lecture plots."

## Cell 6: Transform 3 - Positive/Negative Split
**Title**: "Separate Up Days vs Down Days (Slide Transform #3)"

**Steps**:
1. Create `return_positive = max(return, 0)`
2. Create `return_negative = min(return, 0)`
3. Show time series plot with green (positive) and red (negative) segments

**Connection to Lecture**: "Allows model to learn asymmetric behavior for gains vs losses."

## Cell 7: Transform 4 - Binning/Quantizing
**Title**: "Discretize Volume into Bins (Slide Transform #4)"

**Steps**:
1. Create 5 bins: very_low, low, medium, high, very_high (quintiles)
2. Apply `pd.qcut(Volume, q=5, labels=['very_low', ..., 'very_high'])`
3. One-hot encode the bins
4. Show bar chart of volume distribution across bins

**Connection to Lecture**: "Converts continuous feature to categorical, captures non-linear effects through binning."

## Cell 8: Transform 5 - Interaction Features
**Title**: "Create Interaction: Return × Volume (Slide Transform #5)"

**Steps**:
1. Compute `return_volume_interaction = daily_return * volume_change`
2. Plot scatter: daily_return vs volume_change, color by next-day price movement
3. Add interaction feature column to DataFrame

**Connection to Lecture**: "Product features capture joint effects: large volume + large return may signal trend continuation."

## Cell 9: Transform 6 - Cyclic Time Features
**Title**: "Day-of-Week Circular Embedding (Slide Transform #6)"

**Steps**:
1. Extract day of week (0=Monday, ..., 6=Sunday)
2. Create `day_sin = sin(2π * day / 7)` and `day_cos = cos(2π * day / 7)`
3. Plot circular embedding (scatter on unit circle with day labels)
4. Optional: Color points by average return for that day

**Connection to Lecture**: "Same circular encoding shown in lecture for cyclic features."

## Cell 10: Final Feature Matrix
**Title**: "Complete Engineered Feature Set"

**Steps**:
1. Combine all engineered features into single DataFrame
2. Display first 10 rows with clear column names
3. Print shape: (n_samples, n_features)
4. Show correlation matrix heatmap (optional)

**Output Example**:
```
Feature Matrix Shape: (1200, 15)

Features:
- return_raw
- return_zscore (Transform 1: Standardization)
- return_gamma (Transform 2: Gamma)
- return_pos (Transform 3: Pos/Neg Split)
- return_neg (Transform 3: Pos/Neg Split)
- volume_bin_1...5 (Transform 4: Binning - one-hot encoded)
- return_volume_interaction (Transform 5: Interaction)
- day_sin, day_cos (Transform 6: Cyclic Encoding)
- ma_5, ma_20 (derived features)
```

## Cell 11: Summary & Key Takeaways
**Content**:
- We transformed 3 raw features (price, volume, date) into 15+ engineered features
- Each transform directly corresponds to EE104 lecture content:
  1. **Standardization** → Comparable scales
  2. **Gamma** → Adjust sensitivity to extremes
  3. **Pos/Neg split** → Capture asymmetry
  4. **Binning** → Discretize continuous variables
  5. **Interactions** → Capture joint effects
  6. **Cyclic encoding** → Handle periodic features
- **Out of scope**: Model training (would be next step: logistic regression, random forest)
- **Key insight**: Feature engineering increases model capacity without increasing model complexity

**Verbal Explanation**: "By thoughtfully engineering features using transforms from the lecture, we've prepared the data for effective learning. This demonstrates how the abstract concepts directly apply to real-world forecasting problems."

---

## Implementation Notes (All Notebooks)

1. **Notebook 1** (Feature Transforms Demo):
   - Focus: Visual demonstration of each transform
   - No real data needed (synthetic examples)
   - Priority: Clear, beautiful plots

2. **Notebook 2** (Word2Vec):
   - Focus: Automatic feature learning concept
   - Uses pretrained models (no training required)
   - Keep it concise (5-10 minutes to run)

3. **Notebook 3** (Stock Price Features):
   - Focus: Applying ALL transforms from Notebook 1 to real data
   - Must explicitly reference "Transform #N from slides"
   - No model training (ends at feature matrix)
   - Keep scope narrow: 6 transforms applied systematically

**Consistency**: All three notebooks should use same plotting style, color scheme, and formatting conventions.

---

## Technical Review Notes

### Current Demo (now Notebook 1) - Verified ✓:
1. **Z-Scoring**: Correct implementation, good visualization
2. **Gamma Transform**: Properly handles negative values with `np.sign(x) * np.abs(x)**gamma`
3. **Clipping**: Uses percentiles correctly (5th, 95th)
4. **Powers**: Shows polynomial progression clearly
5. **Pos/Neg Split**: Uses `np.maximum` and `np.minimum` correctly
6. **Circular Embedding**: Proper circular coordinates for days

**No technical issues found.** All transforms are mathematically sound and align with lecture content.

### Recommendations:
- Keep all existing visualizations in Notebook 1 (they're good!)
- Add more explicit connections to slides in Notebook 3
- For Word2Vec, use `glove-wiki-gigaword-100` for faster download
- In Stock notebook, number transforms as "Transform #N" matching slide order
