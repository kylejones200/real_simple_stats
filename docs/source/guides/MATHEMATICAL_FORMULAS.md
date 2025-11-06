# Mathematical Formulas - LaTeX Notation

Complete mathematical reference for all Real Simple Stats functions with LaTeX formulas.

---

## 📊 Descriptive Statistics

### Mean (Arithmetic Average)

**Formula:**
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Function:** `mean(data)`

**Example:**
```python
import real_simple_stats as rss
data = [1, 2, 3, 4, 5]
result = rss.mean(data)  # 3.0
```

---

### Median

**Formula:**
$$\text{Median} = \begin{cases}
x_{(n+1)/2} & \text{if } n \text{ is odd} \\
\frac{x_{n/2} + x_{(n/2)+1}}{2} & \text{if } n \text{ is even}
\end{cases}$$

**Function:** `median(data)`

---

### Sample Variance

**Formula:**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Function:** `sample_variance(data)`

**Note:** Uses $n-1$ (Bessel's correction) for unbiased estimation.

---

### Sample Standard Deviation

**Formula:**
$$s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

**Function:** `sample_std_dev(data)`

---

### Coefficient of Variation

**Formula:**
$$CV = \frac{s}{\bar{x}} \times 100\%$$

**Function:** `coefficient_of_variation(data)`

**Interpretation:** Relative variability; useful for comparing datasets with different units.

---

### Interquartile Range (IQR)

**Formula:**
$$IQR = Q_3 - Q_1$$

where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile.

**Function:** `interquartile_range(data)`

---

## 📈 Probability Distributions

### Normal Distribution

#### Probability Density Function (PDF)

**Formula:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Function:** `normal_pdf(x, mu, sigma)`

**Parameters:**
- $\mu$ = mean
- $\sigma$ = standard deviation
- $x$ = value

---

#### Cumulative Distribution Function (CDF)

**Formula:**
$$F(x) = P(X \leq x) = \int_{-\infty}^{x} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(t-\mu)^2}{2\sigma^2}} dt$$

**Function:** `normal_cdf(x, mu, sigma)`

---

#### Standard Normal (Z-score)

**Formula:**
$$Z = \frac{X - \mu}{\sigma}$$

**Function:** `z_score(x, mu, sigma)`

**Properties:**
- $Z \sim N(0, 1)$
- $P(|Z| \leq 1.96) \approx 0.95$

---

### Binomial Distribution

#### Probability Mass Function (PMF)

**Formula:**
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$

**Function:** `binomial_probability(n, k, p)`

**Parameters:**
- $n$ = number of trials
- $k$ = number of successes
- $p$ = probability of success

---

#### Mean and Variance

**Formulas:**
$$E[X] = np$$
$$\text{Var}(X) = np(1-p)$$

**Functions:** `binomial_mean(n, p)`, `binomial_variance(n, p)`

---

### Poisson Distribution

#### Probability Mass Function

**Formula:**
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Function:** `poisson_pmf(k, lam)`

**Parameters:**
- $\lambda$ = rate parameter (mean)
- $k$ = number of events

**Properties:**
- $E[X] = \lambda$
- $\text{Var}(X) = \lambda$

---

### Geometric Distribution

#### Probability Mass Function

**Formula:**
$$P(X = k) = (1-p)^{k-1} p$$

**Function:** `geometric_pmf(k, p)`

**Interpretation:** Probability that first success occurs on trial $k$.

**Properties:**
- $E[X] = \frac{1}{p}$
- $\text{Var}(X) = \frac{1-p}{p^2}$

---

### Exponential Distribution

#### Probability Density Function

**Formula:**
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

**Function:** `exponential_pdf(x, lam)`

**Properties:**
- $E[X] = \frac{1}{\lambda}$
- $\text{Var}(X) = \frac{1}{\lambda^2}$
- Memoryless property: $P(X > s + t | X > s) = P(X > t)$

---

## 🧪 Hypothesis Testing

### One-Sample t-Test

**Test Statistic:**
$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

**Function:** `one_sample_t_test(data, mu0)`

**Degrees of Freedom:** $df = n - 1$

**Hypotheses:**
- $H_0: \mu = \mu_0$
- $H_1: \mu \neq \mu_0$ (two-tailed)

---

### Two-Sample t-Test (Independent)

**Test Statistic (Equal Variances):**
$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

where pooled standard deviation:
$$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

**Function:** `two_sample_t_test(data1, data2)`

**Degrees of Freedom:** $df = n_1 + n_2 - 2$

---

### Paired t-Test

**Test Statistic:**
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

where $d_i = x_{1i} - x_{2i}$ are the paired differences.

**Function:** `paired_t_test(data1, data2)`

---

### One-Sample Z-Test

**Test Statistic:**
$$Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

**Function:** `one_sample_z_test(data, mu0, sigma)`

**Note:** Requires known population standard deviation $\sigma$.

---

### Chi-Square Goodness-of-Fit Test

**Test Statistic:**
$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

**Function:** `chi_square_statistic(observed, expected)`

**Parameters:**
- $O_i$ = observed frequency
- $E_i$ = expected frequency
- $k$ = number of categories

**Degrees of Freedom:** $df = k - 1 - p$ (where $p$ = number of estimated parameters)

---

### One-Way ANOVA

**Test Statistic:**
$$F = \frac{MS_{between}}{MS_{within}} = \frac{SS_{between}/(k-1)}{SS_{within}/(N-k)}$$

where:
$$SS_{between} = \sum_{i=1}^{k} n_i(\bar{x}_i - \bar{x})^2$$
$$SS_{within} = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(x_{ij} - \bar{x}_i)^2$$

**Function:** `one_way_anova(groups)`

**Degrees of Freedom:** $df_1 = k-1$, $df_2 = N-k$

---

## 📉 Regression & Correlation

### Pearson Correlation Coefficient

**Formula:**
$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

**Function:** `pearson_correlation(x, y)`

**Properties:**
- $-1 \leq r \leq 1$
- $r = 1$: perfect positive correlation
- $r = -1$: perfect negative correlation
- $r = 0$: no linear correlation

---

### Simple Linear Regression

**Model:**
$$y = \beta_0 + \beta_1 x + \epsilon$$

**Least Squares Estimates:**
$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$

**Function:** `linear_regression(x, y)`

---

### Coefficient of Determination (R²)

**Formula:**
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

**Function:** `coefficient_of_determination(x, y)`

**Interpretation:** Proportion of variance in $y$ explained by $x$.

---

### Multiple Linear Regression

**Model:**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$$

**Matrix Form:**
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

**Least Squares Solution:**
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Function:** `multiple_regression(X, y)`

---

## 🔄 Time Series Analysis

### Simple Moving Average (SMA)

**Formula:**
$$SMA_t = \frac{1}{k}\sum_{i=0}^{k-1} x_{t-i}$$

**Function:** `moving_average(data, window_size, method='simple')`

---

### Exponential Moving Average (EMA)

**Formula:**
$$EMA_t = \alpha x_t + (1-\alpha)EMA_{t-1}$$

where $\alpha = \frac{2}{k+1}$ (smoothing factor)

**Function:** `moving_average(data, window_size, method='exponential')`

---

### Autocorrelation Function (ACF)

**Formula:**
$$\rho_k = \frac{\sum_{t=1}^{n-k}(x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{n}(x_t - \bar{x})^2}$$

**Function:** `autocorrelation(data, max_lag)`

**Interpretation:** Correlation between $x_t$ and $x_{t+k}$.

---

### Linear Trend

**Model:**
$$x_t = \beta_0 + \beta_1 t + \epsilon_t$$

**Function:** `linear_trend(data)`

**Returns:** slope ($\beta_1$), intercept ($\beta_0$), $R^2$

---

### First-Order Differencing

**Formula:**
$$\nabla x_t = x_t - x_{t-1}$$

**Function:** `difference(data, lag=1, order=1)`

**Purpose:** Remove trend, achieve stationarity.

---

## 🎲 Resampling Methods

### Bootstrap Confidence Interval

**Algorithm:**
1. Draw $B$ bootstrap samples with replacement
2. Calculate statistic $\theta^*_b$ for each sample
3. Find percentiles of bootstrap distribution

**Percentile Method:**
$$CI = [\theta^*_{\alpha/2}, \theta^*_{1-\alpha/2}]$$

**Function:** `bootstrap(data, statistic, n_iterations, confidence_level)`

---

### Permutation Test

**Test Statistic:**
$$T_{obs} = f(X_1, X_2)$$

**P-value:**
$$p = \frac{\#\{T_{perm} \geq T_{obs}\}}{B}$$

where $B$ = number of permutations.

**Function:** `permutation_test(data1, data2, statistic, n_permutations)`

---

### Jackknife Standard Error

**Formula:**
$$SE_{jack} = \sqrt{\frac{n-1}{n}\sum_{i=1}^{n}(\theta_{(i)} - \bar{\theta})^2}$$

where $\theta_{(i)}$ is the statistic computed without observation $i$.

**Function:** `jackknife(data, statistic)`

---

## 📊 Effect Sizes

### Cohen's d

**Formula:**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

where:
$$s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

**Function:** `cohens_d(group1, group2, pooled=True)`

**Interpretation:**
- Small: $|d| \approx 0.2$
- Medium: $|d| \approx 0.5$
- Large: $|d| \approx 0.8$

---

### Hedges' g

**Formula:**
$$g = d \times \left(1 - \frac{3}{4(n_1 + n_2) - 9}\right)$$

**Function:** `hedges_g(group1, group2)`

**Note:** Bias-corrected version of Cohen's d for small samples.

---

### Eta-Squared (η²)

**Formula:**
$$\eta^2 = \frac{SS_{between}}{SS_{total}}$$

**Function:** `eta_squared(groups)`

**Interpretation:** Proportion of total variance explained by group membership.

---

### Partial Eta-Squared

**Formula:**
$$\eta_p^2 = \frac{SS_{effect}}{SS_{effect} + SS_{error}}$$

**Function:** `partial_eta_squared(groups)`

---

### Omega-Squared (ω²)

**Formula:**
$$\omega^2 = \frac{SS_{between} - (k-1)MS_{within}}{SS_{total} + MS_{within}}$$

**Function:** `omega_squared(groups)`

**Note:** Less biased than $\eta^2$, especially for small samples.

---

### Cramér's V

**Formula:**
$$V = \sqrt{\frac{\chi^2}{n \times \min(r-1, c-1)}}$$

**Function:** `cramers_v(contingency_table)`

**Range:** $0 \leq V \leq 1$

**Interpretation:**
- Small: $V \approx 0.1$
- Medium: $V \approx 0.3$
- Large: $V \approx 0.5$

---

### Odds Ratio

**Formula (2×2 table):**
$$OR = \frac{a \times d}{b \times c}$$

for table:
$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}$$

**Function:** `odds_ratio(contingency_table)`

**Interpretation:**
- $OR = 1$: no association
- $OR > 1$: positive association
- $OR < 1$: negative association

---

### Relative Risk

**Formula:**
$$RR = \frac{a/(a+b)}{c/(c+d)}$$

**Function:** `relative_risk(contingency_table)`

---

### Cohen's h

**Formula:**
$$h = 2(\arcsin\sqrt{p_1} - \arcsin\sqrt{p_2})$$

**Function:** `cohens_h(p1, p2)`

**Use:** Effect size for difference between two proportions.

---

## 🔬 Power Analysis

### Power for t-Test

**Formula:**
$$\text{Power} = 1 - \beta = P(\text{reject } H_0 | H_1 \text{ true})$$

**Non-centrality Parameter:**
$$\delta = \frac{\mu_1 - \mu_0}{\sigma} \sqrt{n}$$

**Function:** `power_t_test(delta=None, n=None, power=None, sig_level=0.05)`

**Note:** Provide any 3 parameters to solve for the 4th.

---

### Sample Size for t-Test

**Formula:**
$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma^2}{\delta^2}$$

where:
- $z_{1-\alpha/2}$ = critical value for significance level
- $z_{1-\beta}$ = critical value for power
- $\delta$ = effect size

**Function:** `power_t_test(delta, power, sig_level)`

---

### Power for ANOVA

**Effect Size (Cohen's f):**
$$f = \sqrt{\frac{\eta^2}{1-\eta^2}}$$

**Function:** `power_anova(effect_size, k, n=None, power=None, sig_level=0.05)`

---

### Minimum Detectable Effect

**Formula:**
$$MDE = \frac{(z_{1-\alpha/2} + z_{1-\beta})\sigma}{\sqrt{n}}$$

**Function:** `minimum_detectable_effect(n, power, sig_level, sigma)`

---

## 🎯 Bayesian Statistics

### Beta-Binomial Conjugate Update

**Prior:**
$$p \sim \text{Beta}(\alpha, \beta)$$

**Likelihood:**
$$X | p \sim \text{Binomial}(n, p)$$

**Posterior:**
$$p | X \sim \text{Beta}(\alpha + k, \beta + n - k)$$

**Function:** `beta_binomial_update(prior_alpha, prior_beta, successes, trials)`

---

### Normal-Normal Conjugate Update

**Prior:**
$$\mu \sim N(\mu_0, \sigma_0^2)$$

**Likelihood:**
$$X_i | \mu \sim N(\mu, \sigma^2)$$

**Posterior:**
$$\mu | X \sim N(\mu_n, \sigma_n^2)$$

where:
$$\mu_n = \frac{\sigma^2\mu_0 + n\sigma_0^2\bar{x}}{\sigma^2 + n\sigma_0^2}$$

$$\sigma_n^2 = \frac{\sigma^2\sigma_0^2}{\sigma^2 + n\sigma_0^2}$$

**Function:** `normal_normal_update(prior_mean, prior_variance, data, data_variance)`

---

### Gamma-Poisson Conjugate Update

**Prior:**
$$\lambda \sim \text{Gamma}(\alpha, \beta)$$

**Likelihood:**
$$X_i | \lambda \sim \text{Poisson}(\lambda)$$

**Posterior:**
$$\lambda | X \sim \text{Gamma}(\alpha + \sum x_i, \beta + n)$$

**Function:** `gamma_poisson_update(prior_shape, prior_rate, data)`

---

### Credible Interval

**Definition:**
$$P(\theta \in [L, U] | X) = 1 - \alpha$$

**Function:** `credible_interval(distribution, params, credibility)`

**Note:** Bayesian analog of confidence interval.

---

### Highest Density Interval (HDI)

**Definition:**
Shortest interval containing $(1-\alpha)$ of the posterior probability.

**Function:** `highest_density_interval(samples, credibility)`

**Property:** All points inside HDI have higher density than points outside.

---

### Bayes Factor

**Formula:**
$$BF_{10} = \frac{P(D|H_1)}{P(D|H_0)} \times \frac{P(H_1)}{P(H_0)}$$

**Function:** `bayes_factor(likelihood_h1, likelihood_h0, prior_odds)`

**Interpretation (Kass & Raftery):**
- $BF < 1$: Evidence for $H_0$
- $1 < BF < 3$: Barely worth mentioning
- $3 < BF < 10$: Substantial evidence for $H_1$
- $10 < BF < 30$: Strong evidence
- $30 < BF < 100$: Very strong evidence
- $BF > 100$: Decisive evidence

---

## 📐 Multivariate Analysis

### Principal Component Analysis (PCA)

**Objective:**
Find orthogonal directions of maximum variance.

**Eigenvalue Decomposition:**
$$\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$$

where:
- $\mathbf{\Sigma}$ = covariance matrix
- $\mathbf{V}$ = eigenvectors (principal components)
- $\mathbf{\Lambda}$ = diagonal matrix of eigenvalues

**Transformed Data:**
$$\mathbf{Z} = \mathbf{X}\mathbf{V}$$

**Function:** `pca(X, n_components)`

---

### Mahalanobis Distance

**Formula:**
$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

**Function:** `mahalanobis_distance(X, point)`

**Interpretation:** Distance accounting for correlations and scale differences.

---

## 📊 Confidence Intervals

### CI for Mean (Known σ)

**Formula:**
$$\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

**Function:** `confidence_interval_known_std(mean, std_dev, n, confidence)`

---

### CI for Mean (Unknown σ)

**Formula:**
$$\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$

**Function:** `confidence_interval_unknown_std(sample_mean, sample_std, n, confidence)`

---

## 🎲 Central Limit Theorem

### Sampling Distribution of the Mean

**Properties:**
$$E[\bar{X}] = \mu$$
$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$$
$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \text{ (approximately, for large } n\text{)}$$

**Functions:**
- `sampling_distribution_mean(pop_mean)`
- `sampling_distribution_variance(pop_std, sample_size)`
- `clt_probability_greater_than(x, mean, std_dev, n)`
- `clt_probability_less_than(x, mean, std_dev, n)`
- `clt_probability_between(x1, x2, mean, std_dev, n)`

---

## 📚 References

**Notation:**
- $\mu$ = population mean
- $\sigma$ = population standard deviation
- $\bar{x}$ = sample mean
- $s$ = sample standard deviation
- $n$ = sample size
- $\alpha$ = significance level
- $\beta$ = Type II error rate
- $1-\beta$ = statistical power

**Common Critical Values:**
- $z_{0.975} = 1.96$ (95% CI, two-tailed)
- $z_{0.995} = 2.576$ (99% CI, two-tailed)
- $t_{0.975, \infty} \approx 1.96$

---

**See also:**
- [API Comparison](API_COMPARISON.md) - Function lookup
- [Interactive Examples](INTERACTIVE_EXAMPLES.md) - Try it yourself
- [FAQ](FAQ.md) - Common questions
