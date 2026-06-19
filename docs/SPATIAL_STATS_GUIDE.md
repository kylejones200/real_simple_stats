# Spatial Statistics Guide

Spatial data — measurements made at geographic or spatial locations — violates a core assumption of most standard statistical tests: independence. If high pollution in one neighbourhood makes adjacent neighbourhoods more likely to also have high pollution, observations are not independent. Spatial statistics quantifies and models this dependence.

---

## Why spatial autocorrelation matters

**Tobler's First Law of Geography**: everything is related to everything else, but near things are more related than distant things.

When spatial autocorrelation is present and ignored:
- Standard errors from OLS are too small (you have less independent information than your n suggests)
- p-values are too small — more false positives
- Predictions from non-spatial models are less accurate near cluster boundaries

`real_simple_stats` provides two tools: **Moran's I** (a global summary) and **variograms** (a distance-dependent model of autocorrelation).

---

## Moran's I — global spatial autocorrelation

Moran's I is a weighted spatial correlation coefficient. It compares each location's value to the average of its neighbours. The result ranges from −1 to +1:

- **I ≈ +1**: similar values cluster together (high next to high, low next to low)
- **I ≈ 0**: values are distributed as if randomly placed on the map
- **I ≈ −1**: dissimilar values are neighbours (a checkerboard pattern) — rare in practice

Under the null hypothesis of spatial randomness, E[I] = −1/(n−1) ≈ 0.

### Basic usage

```python
import real_simple_stats as rss
import numpy as np

rng = np.random.default_rng(0)
n = 100

# Spatially structured data: two clusters (high in left half, low in right)
x = rng.uniform(0, 100, n)
y = rng.uniform(0, 100, n)
values = np.where(x < 50, rng.normal(10, 1, n), rng.normal(0, 1, n))

r = rss.morans_i(x, y, values, distance_threshold=20)
print(f"Moran's I: {r['moran_i']:.3f}")    # strongly positive
print(f"p-value:   {r['p_value']:.4f}")
print(r["interpretation"])

# Self-explaining version
result = rss.morans_i_explained(x, y, values, distance_threshold=20)
print(result)
result.plot()   # spatial scatter coloured by value
```

### Choosing a distance threshold

The distance threshold defines who counts as a "neighbour." It is the most consequential choice in a Moran's I analysis — the result can change substantially with different thresholds.

**Practical guidance**:

1. **Domain knowledge**: if the process you're studying has a known scale (e.g. disease spreads within 2 km), use that.
2. **Vary and compare**: compute I at several thresholds and plot I vs. threshold. A robust result is stable across a range.
3. **Average neighbour count**: each observation should have at least 3–5 neighbours on average.

```python
from scipy.spatial.distance import cdist
import numpy as np

coords = np.column_stack([x, y])
D = cdist(coords, coords)

for d in [10, 20, 30, 50]:
    avg_n = (D < d).sum(axis=1).mean() - 1
    r = rss.morans_i(x, y, values, distance_threshold=d)
    print(f"threshold={d:3d}  avg_neighbours={avg_n:.1f}  I={r['moran_i']:.3f}  p={r['p_value']:.3f}")
```

### Limitations of global Moran's I

Global Moran's I summarises the *entire map* in a single number. Local clusters can cancel out: a high-high cluster in the north and a low-low cluster in the south produce near-zero global I even though clear spatial structure exists.

For local cluster detection (hotspots and coldspots), use Local Indicators of Spatial Association (LISA / Local Moran's I) — not yet in `rss`, available in `PySAL`.

---

## Variograms — modelling spatial autocorrelation by distance

While Moran's I answers "is there spatial autocorrelation?" the variogram answers "how does spatial autocorrelation decay with distance?" It is the foundation of geostatistical interpolation (kriging).

### The experimental variogram

The semivariance γ(h) measures how different two values are, on average, when they are separated by distance h:

```
γ(h) = ½ × mean of (z(s) − z(s + h))² for all pairs at lag h
```

Large γ(h) means points at distance h tend to have different values. Small γ(h) means they tend to be similar.

```python
# Step 1: compute the experimental variogram
vario = rss.compute_variogram(x, y, values, n_lags=15)

print("Lags:", vario["lags"].round(1))
print("Semivariance:", vario["gamma"].round(3))
print("Pairs per bin:", vario["n_pairs"])
print(f"Total variance: {vario['total_variance']:.3f}")
```

The experimental variogram is a scatter of points — a rough picture. You fit a smooth model to it to get interpretable parameters.

### The three model families

```python
# Step 2: fit a model
fit = rss.fit_variogram(vario["lags"], vario["gamma"], model="spherical")

print(f"Nugget:  {fit['nugget']:.3f}")       # variance at h → 0 (measurement error / micro-scale variation)
print(f"Sill:    {fit['sill']:.3f}")         # total variance (plateau)
print(f"Range:   {fit['range_param']:.1f}")  # distance where autocorrelation effectively vanishes
print(f"RMSE:    {fit['rmse']:.4f}")         # fit quality

# Predict semivariance at a new distance
gamma_at_15 = fit["model_fn"](15.0)
```

**Nugget**: semivariance at zero distance. Theoretically should be zero (identical locations have identical values), but in practice reflects measurement error and sub-sampling-scale variation.

**Sill**: the plateau the variogram approaches at large distances. Equals the total variance of the data when the nugget is zero.

**Range**: the lag at which the sill is (effectively) reached. Beyond this distance, knowing one value tells you nothing about a value that far away — they are spatially uncorrelated. The range defines the scale of spatial structure.

### Choosing a model

All three models share the same three parameters; they differ in shape:

```python
for model in ("spherical", "exponential", "gaussian"):
    fit = rss.fit_variogram(vario["lags"], vario["gamma"], model=model)
    print(f"{model:12s}  RMSE={fit['rmse']:.4f}  range={fit['range_param']:.1f}")
```

| Model | Shape near h=0 | Best for |
|---|---|---|
| **Spherical** | Linear (abrupt transition to sill) | Geology, soil data; most commonly used |
| **Exponential** | Linear (slower, never fully reaches sill) | Data with correlation at all distances |
| **Gaussian** | Parabolic (very smooth near origin) | Highly smooth processes (atmospheric, gravity) |

**Decision**: start with spherical. If the experimental variogram has a very smooth shape near the origin (no sharp initial rise), try Gaussian. If the variogram doesn't clearly plateau, try exponential.

### Full worked example

```python
import real_simple_stats as rss
import numpy as np

rng = np.random.default_rng(1)
n = 120

# Spatially structured data with known range ≈ 25
x = rng.uniform(0, 100, n)
y = rng.uniform(0, 100, n)

# Generate spatially correlated values using a simple approximation
from scipy.spatial.distance import cdist
D = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
C = np.exp(-D / 25)   # exponential covariance with range 25
L = np.linalg.cholesky(C + 1e-8 * np.eye(n))
values = L @ rng.standard_normal(n)

# Step 1: Moran's I
r = rss.morans_i(x, y, values, distance_threshold=25)
print(f"Moran's I: {r['moran_i']:.3f}  (p={r['p_value']:.4f})")

# Step 2: Experimental variogram
vario = rss.compute_variogram(x, y, values, n_lags=15, max_lag=60)

# Step 3: Fit models, compare by RMSE
for model in ("spherical", "exponential", "gaussian"):
    fit = rss.fit_variogram(vario["lags"], vario["gamma"], model=model)
    print(f"{model:12s}  RMSE={fit['rmse']:.4f}  range={fit['range_param']:.1f}")

# Step 4: Use best model
fit = rss.fit_variogram(vario["lags"], vario["gamma"], model="exponential")
print(f"\nRange: {fit['range_param']:.1f} units")   # ≈ 25 if data-generating process recovered
print(f"Sill:  {fit['sill']:.3f}")
```

---

## Common mistakes

**Ignoring the distance threshold**: Moran's I with `distance_threshold=None` uses all pairs as neighbours. For data with clear local clustering, this dilutes the signal — far-apart pairs that are unrelated pull I toward zero.

**Too few lag bins or too many**: too few bins averages over very different distances; too many bins leave few point pairs per bin, making the experimental variogram noisy. `n_lags=10–20` is usually appropriate.

**Fitting a variogram to too few points**: you need at least ~30 point pairs per lag bin for reliable semivariance estimates. Check `vario["n_pairs"]` — bins with fewer than 10 pairs should be treated with caution.

**Interpreting range as a hard cutoff**: the range is where the variogram *effectively* reaches the sill. For the exponential model the sill is never technically reached — the "practical range" is often defined as the distance at which γ(h) = 95% of the sill.

**Using Moran's I when a trend exists**: Moran's I measures spatial autocorrelation in raw values. A directional trend (values systematically higher in the north) produces positive I even if there's no clustering beyond the trend. Detrend first if a gradient is visible.

---

## See also

- [WHICH_TEST.md](WHICH_TEST.md) — spatial branch of the decision tree
- [MATHEMATICAL_FORMULAS.md](MATHEMATICAL_FORMULAS.md) — Moran's I formula, variogram models
- [FAQ.md](FAQ.md) — distance threshold choice, variogram model selection
