//! Tests for the pure-Rust core.
//!
//! These check invariants and closed-form identities that hold independently of
//! any reference implementation. Agreement with published values to full double
//! precision is gated separately, from Python, against mpmath and SciPy -- see
//! tests/parity/ in the repository root.

use approx::assert_relative_eq;
use rss_core::{
    descriptive as ds, dist, fit, linalg, noncentral, optimize, resample, rng, special,
};

const TOL: f64 = 1e-12;

// ------------------------------------------------------------------ special --

#[test]
fn erf_is_odd_and_bounded() {
    for i in 0..60 {
        let x = i as f64 * 0.1;
        assert_relative_eq!(special::erf(-x), -special::erf(x), epsilon = 1e-15);
        assert!(special::erf(x) >= 0.0 && special::erf(x) <= 1.0);
    }
}

#[test]
fn erfc_complements_erf() {
    for i in -40..40 {
        let x = i as f64 * 0.15;
        assert_relative_eq!(special::erf(x) + special::erfc(x), 1.0, epsilon = 1e-14);
    }
}

#[test]
fn normal_cdf_and_sf_sum_to_one() {
    for i in -50..50 {
        let x = i as f64 * 0.12;
        assert_relative_eq!(
            special::norm_cdf(x) + special::norm_sf(x),
            1.0,
            epsilon = 1e-14
        );
    }
}

#[test]
fn norm_ppf_inverts_norm_cdf() {
    for i in 1..100 {
        let p = i as f64 / 100.0;
        assert_relative_eq!(special::norm_cdf(special::norm_ppf(p)), p, epsilon = 1e-12);
    }
}

#[test]
fn norm_ppf_known_quantiles() {
    assert_relative_eq!(
        special::norm_ppf(0.975),
        1.959_963_984_540_054,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        special::norm_ppf(0.995),
        2.575_829_303_548_901,
        epsilon = 1e-12
    );
    assert_relative_eq!(special::norm_ppf(0.5), 0.0, epsilon = 1e-15);
}

#[test]
fn ln_gamma_matches_factorials() {
    let mut factorial = 1.0f64;
    for n in 1..15u32 {
        factorial *= n as f64;
        assert_relative_eq!(
            special::ln_gamma(n as f64 + 1.0).exp(),
            factorial,
            max_relative = 1e-12
        );
    }
}

#[test]
fn incomplete_gamma_halves_are_complementary() {
    for i in 1..40 {
        let x = i as f64 * 0.25;
        assert_relative_eq!(
            special::gammainc_p(2.5, x) + special::gammainc_q(2.5, x),
            1.0,
            epsilon = 1e-14
        );
    }
}

#[test]
fn betainc_symmetry() {
    for (a, b, x) in [(2.0, 3.0, 0.3), (0.5, 5.0, 0.7), (10.0, 10.0, 0.5)] {
        assert_relative_eq!(
            special::betainc(a, b, x),
            1.0 - special::betainc(b, a, 1.0 - x),
            epsilon = 1e-13
        );
    }
}

#[test]
fn inverses_round_trip() {
    for (a, b) in [(2.0, 5.0), (0.5, 0.5), (30.0, 2.0)] {
        for p in [1e-8, 0.01, 0.5, 0.99, 1.0 - 1e-8] {
            let x = special::betaincinv(a, b, p);
            assert!((0.0..=1.0).contains(&x), "quantile escaped [0,1]: {x}");

            // Where x saturates against the last representable double below 1
            // (Beta(0.5, 0.5) at p -> 1 does this), no implementation can
            // recover p by round-tripping: consecutive doubles near 1 are
            // 2.2e-16 apart, and I_x moves less than the target tolerance
            // across that gap. SciPy returns the identical x and the identical
            // round-trip value. Check only that x is at the boundary.
            if 1.0 - x <= 1e-15 || x <= 1e-15 {
                continue;
            }
            assert_relative_eq!(special::betainc(a, b, x), p, max_relative = 1e-10);
        }
    }
    for a in [0.5, 2.0, 50.0] {
        for p in [1e-8, 0.25, 0.99] {
            let x = special::gammaincinv(a, p);
            assert!(x >= 0.0 && x.is_finite());
            assert_relative_eq!(special::gammainc_p(a, x), p, max_relative = 1e-10);
        }
    }
}

// --------------------------------------------------------------- distributions

#[test]
fn t_distribution_is_symmetric() {
    for df in [1.0, 5.0, 30.0] {
        for i in 0..30 {
            let x = i as f64 * 0.2;
            assert_relative_eq!(dist::t_cdf(-x, df), dist::t_sf(x, df), epsilon = 1e-14);
        }
    }
}

#[test]
fn t_with_one_df_is_cauchy() {
    for i in -20..20 {
        let x = i as f64 * 0.3;
        let cauchy = 0.5 + x.atan() / std::f64::consts::PI;
        assert_relative_eq!(dist::t_cdf(x, 1.0), cauchy, epsilon = 1e-12);
    }
}

#[test]
fn chi2_with_two_df_is_exponential() {
    for i in 1..30 {
        let x = i as f64 * 0.4;
        assert_relative_eq!(
            dist::chi2_cdf(x, 2.0),
            1.0 - (-x / 2.0).exp(),
            epsilon = 1e-13
        );
    }
}

#[test]
fn cdf_and_sf_are_complementary() {
    for x in [0.5f64, 1.0, 3.0, 8.0] {
        assert_relative_eq!(
            dist::f_cdf(x, 5.0, 9.0) + dist::f_sf(x, 5.0, 9.0),
            1.0,
            epsilon = 1e-13
        );
        assert_relative_eq!(
            dist::chi2_cdf(x, 4.0) + dist::chi2_sf(x, 4.0),
            1.0,
            epsilon = 1e-13
        );
        assert_relative_eq!(
            dist::beta_cdf(x / 10.0, 2.0, 3.0) + dist::beta_sf(x / 10.0, 2.0, 3.0),
            1.0,
            epsilon = 1e-13
        );
    }
}

#[test]
fn binomial_pmf_sums_to_one() {
    for (n, p) in [(10.0, 0.3), (25.0, 0.5), (40.0, 0.85)] {
        let total: f64 = (0..=n as usize)
            .map(|k| dist::binom_pmf(k as f64, n, p))
            .sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-12);
    }
}

#[test]
fn poisson_pmf_sums_to_one() {
    for mu in [0.5f64, 4.0, 20.0] {
        let total: f64 = (0..200).map(|k| dist::poisson_pmf(k as f64, mu)).sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-12);
    }
}

#[test]
fn ppf_inverts_cdf_across_families() {
    for p in [0.05f64, 0.25, 0.5, 0.75, 0.95] {
        assert_relative_eq!(dist::t_cdf(dist::t_ppf(p, 7.0), 7.0), p, epsilon = 1e-11);
        assert_relative_eq!(
            dist::chi2_cdf(dist::chi2_ppf(p, 3.0), 3.0),
            p,
            epsilon = 1e-11
        );
        assert_relative_eq!(
            dist::f_cdf(dist::f_ppf(p, 4.0, 9.0), 4.0, 9.0),
            p,
            epsilon = 1e-11
        );
        assert_relative_eq!(
            dist::beta_cdf(dist::beta_ppf(p, 2.0, 5.0), 2.0, 5.0),
            p,
            epsilon = 1e-11
        );
    }
}

#[test]
fn noncentral_reduces_to_central_at_zero() {
    for x in [-1.0f64, 0.0, 1.5, 3.0] {
        assert_relative_eq!(
            noncentral::nct_cdf(x, 8.0, 0.0),
            dist::t_cdf(x, 8.0),
            epsilon = 1e-11
        );
    }
    for x in [0.5f64, 1.0, 3.0] {
        assert_relative_eq!(
            noncentral::ncf_cdf(x, 3.0, 10.0, 0.0),
            dist::f_cdf(x, 3.0, 10.0),
            epsilon = 1e-11
        );
    }
}

#[test]
fn shapiro_wilk_separates_normal_from_exponential() {
    let mut rng = rng::Sampler::new(7);
    let normal: Vec<f64> = (0..300).map(|_| rng.normal(0.0, 1.0)).collect();
    let skewed: Vec<f64> = (0..300).map(|_| rng.exponential(1.0)).collect();
    let (w_norm, p_norm) = noncentral::shapiro_wilk(&normal);
    let (w_exp, p_exp) = noncentral::shapiro_wilk(&skewed);
    assert!(w_norm > w_exp, "normal data should score higher W");
    assert!(
        p_norm > 0.01,
        "normal sample wrongly rejected: p = {p_norm}"
    );
    assert!(p_exp < 1e-6, "exponential sample not rejected: p = {p_exp}");
}

// ---------------------------------------------------------------- descriptive

#[test]
fn mean_and_variance_are_exact_on_small_input() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_relative_eq!(ds::mean(&x), 3.0, epsilon = TOL);
    assert_relative_eq!(ds::variance(&x, 1), 2.5, epsilon = TOL);
    assert_relative_eq!(ds::variance(&x, 0), 2.0, epsilon = TOL);
    assert_relative_eq!(ds::std_dev(&x, 1), 2.5f64.sqrt(), epsilon = TOL);
}

#[test]
fn pairwise_sum_is_accurate_over_a_million_terms() {
    let x = vec![0.1f64; 1_000_000];
    let naive = x.iter().fold(0.0f64, |a, b| a + b);
    let pairwise = ds::sum(&x);
    let exact = 100_000.0f64;
    assert!(
        (pairwise - exact).abs() <= (naive - exact).abs(),
        "pairwise {pairwise} should be no worse than naive {naive}"
    );
    assert_relative_eq!(pairwise, exact, max_relative = 1e-14);
}

#[test]
fn five_number_summary_uses_the_tukey_convention() {
    let f = ds::five_number_summary(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert_relative_eq!(f.q1, 1.5, epsilon = TOL);
    assert_relative_eq!(f.median, 3.0, epsilon = TOL);
    assert_relative_eq!(f.q3, 4.5, epsilon = TOL);

    let one = ds::five_number_summary(&[7.0]).unwrap();
    assert_relative_eq!(one.q1, 7.0, epsilon = TOL);
    let three = ds::five_number_summary(&[1.0, 5.0, 9.0]).unwrap();
    assert_relative_eq!(three.q1, 1.0, epsilon = TOL);
    assert_relative_eq!(three.q3, 9.0, epsilon = TOL);

    assert!(ds::five_number_summary(&[]).is_none());
}

#[test]
fn median_handles_both_parities() {
    assert_relative_eq!(ds::median(&[3.0, 1.0, 2.0]), 2.0, epsilon = TOL);
    assert_relative_eq!(ds::median(&[4.0, 1.0, 3.0, 2.0]), 2.5, epsilon = TOL);
}

#[test]
fn pearson_r_is_exact_for_a_perfect_line() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];
    assert_relative_eq!(ds::pearson_r(&x, &y), 1.0, epsilon = 1e-14);
    let z = [10.0, 8.0, 6.0, 4.0, 2.0];
    assert_relative_eq!(ds::pearson_r(&x, &z), -1.0, epsilon = 1e-14);
}

#[test]
fn modes_reports_every_tied_value() {
    assert_eq!(ds::modes(&[1.0, 2.0, 2.0, 3.0]), vec![2.0]);
    assert_eq!(ds::modes(&[1.0, 1.0, 2.0, 2.0]), vec![1.0, 2.0]);
    assert_eq!(ds::modes(&[0.0, -0.0, 1.0]).len(), 1);
}

// ------------------------------------------------------------------- resample

#[test]
fn bootstrap_is_reproducible_from_its_seed() {
    let data: Vec<f64> = (1..=200).map(|i| i as f64 * 0.37).collect();
    let a = resample::bootstrap(&data, resample::Stat::Mean, 500, 42);
    let b = resample::bootstrap(&data, resample::Stat::Mean, 500, 42);
    assert_eq!(a, b, "same seed must give identical results");

    let c = resample::bootstrap(&data, resample::Stat::Mean, 500, 43);
    assert_ne!(a, c, "different seeds must diverge");

    assert_relative_eq!(ds::mean(&a), ds::mean(&data), max_relative = 0.02);
}

#[test]
fn quickselect_median_matches_a_full_sort() {
    let mut rng = rng::Sampler::new(3);
    for n in [1usize, 2, 7, 8, 51, 100] {
        let v: Vec<f64> = (0..n).map(|_| rng.normal(0.0, 1.0)).collect();
        let mut buf = v.clone();
        let quick = resample::median_inplace(&mut buf);
        assert_relative_eq!(quick, ds::median(&v), epsilon = 1e-12);
    }
}

#[test]
fn permutation_pvalue_stays_in_range() {
    let a: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..40).map(|i| 3.0 + i as f64 * 0.1).collect();
    let dist = resample::permutation_diff(&a, &b, resample::Stat::Mean, 500, 1);
    let observed = ds::mean(&a) - ds::mean(&b);
    let p = resample::permutation_pvalue(&dist, observed, "two-sided");
    assert!((0.0..=1.0).contains(&p));
    assert!(
        p < 0.05,
        "clearly separated groups should reject, got p = {p}"
    );
}

// --------------------------------------------------------------------- linalg

#[test]
fn inverse_times_original_is_identity() {
    let a = linalg::Matrix::new(3, 3, vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]).unwrap();
    let inv = linalg::inv(&a).unwrap();
    let prod = linalg::matmul(&a, &inv).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(prod.at(i, j), want, epsilon = 1e-12);
        }
    }
}

#[test]
fn singular_matrix_has_no_inverse() {
    let a = linalg::Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
    assert!(linalg::inv(&a).is_none());
}

#[test]
fn sqrtm_squared_recovers_the_matrix() {
    let a = linalg::Matrix::new(3, 3, vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]).unwrap();
    let root = linalg::sqrtm_spd(&a).unwrap();
    let squared = linalg::matmul(&root, &root).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert_relative_eq!(squared.at(i, j), a.at(i, j), epsilon = 1e-11);
        }
    }
}

#[test]
fn eigh_returns_ascending_eigenvalues() {
    let a = linalg::Matrix::new(3, 3, vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]).unwrap();
    let (vals, _) = linalg::eigh(&a).unwrap();
    assert!(vals.windows(2).all(|w| w[0] <= w[1]));
    assert_relative_eq!(vals.iter().sum::<f64>(), 9.0, epsilon = 1e-12);
}

#[test]
fn lstsq_recovers_an_exact_linear_fit() {
    let x = linalg::Matrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
    let y = [3.0, 5.0, 7.0, 9.0];
    let beta = linalg::lstsq(&x, &y).unwrap();
    assert_relative_eq!(beta[0], 1.0, epsilon = 1e-11);
    assert_relative_eq!(beta[1], 2.0, epsilon = 1e-11);
}

// ------------------------------------------------------------------- optimize

#[test]
fn brentq_finds_a_bracketed_root() {
    let root = optimize::brentq(|x| x * x * x - 2.0 * x - 5.0, 1.0, 3.0, 1e-14, 200).unwrap();
    assert_relative_eq!(root, 2.094_551_481_542_326_6, epsilon = 1e-10);
}

#[test]
fn brentq_rejects_an_unbracketed_interval() {
    assert!(optimize::brentq(|x| x * x + 1.0, -1.0, 1.0, 1e-12, 100).is_none());
}

#[test]
fn simplex_projection_is_feasible_and_idempotent() {
    let v = [0.3, -1.2, 4.0, 0.5];
    let p = optimize::project_to_simplex(&v);
    assert_relative_eq!(p.iter().sum::<f64>(), 1.0, epsilon = 1e-14);
    assert!(p.iter().all(|&x| x >= 0.0));
    let again = optimize::project_to_simplex(&p);
    for (a, b) in p.iter().zip(&again) {
        assert_relative_eq!(a, b, epsilon = 1e-14);
    }
}

#[test]
fn simplex_least_squares_recovers_known_weights() {
    let columns = vec![
        (0..30).map(|i| (i as f64).sin()).collect::<Vec<f64>>(),
        (0..30).map(|i| (i as f64).cos()).collect::<Vec<f64>>(),
        (0..30).map(|i| (i as f64) * 0.1).collect::<Vec<f64>>(),
    ];
    let truth = [0.5, 0.2, 0.3];
    let y: Vec<f64> = (0..30)
        .map(|i| (0..3).map(|j| truth[j] * columns[j][i]).sum())
        .collect();

    let w = optimize::simplex_least_squares(&y, &columns, 20_000, 1e-14);
    assert_relative_eq!(w.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    for (got, want) in w.iter().zip(&truth) {
        assert_relative_eq!(got, want, epsilon = 1e-4);
    }
}

// ------------------------------------------------------------------------ fit

#[test]
fn exponential_mle_is_the_sample_mean() {
    let mut rng = rng::Sampler::new(11);
    let t: Vec<f64> = (0..2000).map(|_| rng.exponential(7.0)).collect();
    let r = fit::fit_exponential(&t);
    assert_relative_eq!(r.scale, ds::mean(&t), epsilon = 1e-12);
    assert_relative_eq!(r.scale, 7.0, max_relative = 0.1);
}

#[test]
fn weibull_mle_recovers_its_parameters() {
    let mut rng = rng::Sampler::new(13);
    let t: Vec<f64> = (0..4000).map(|_| rng.weibull(1.7, 25.0)).collect();
    let r = fit::fit_weibull(&t);
    assert_relative_eq!(r.shape.unwrap(), 1.7, max_relative = 0.08);
    assert_relative_eq!(r.scale, 25.0, max_relative = 0.08);
}

#[test]
fn lognormal_mle_recovers_its_parameters() {
    let mut rng = rng::Sampler::new(17);
    let t: Vec<f64> = (0..4000).map(|_| rng.lognormal(2.0, 0.6)).collect();
    let r = fit::fit_lognormal(&t);
    assert_relative_eq!(r.shape.unwrap(), 0.6, max_relative = 0.08);
    assert_relative_eq!(r.scale, 2.0f64.exp(), max_relative = 0.08);
}

#[test]
fn the_correct_family_wins_on_log_likelihood() {
    let mut rng = rng::Sampler::new(19);
    let t: Vec<f64> = (0..3000).map(|_| rng.exponential(12.0)).collect();
    let expo = fit::fit_exponential(&t).log_likelihood;
    let logn = fit::fit_lognormal(&t).log_likelihood;
    assert!(
        expo > logn,
        "exponential data should favour the exponential fit"
    );
}

// ------------------------------------------------------------------------ rng

#[test]
fn streams_are_reproducible_and_distinct() {
    use rand::Rng;
    let draw = |seed: u64, i: u64| rng::stream(seed, i).gen::<f64>();
    let a: Vec<f64> = (0..5).map(|i| draw(99, i)).collect();
    let b: Vec<f64> = (0..5).map(|i| draw(99, i)).collect();
    assert_eq!(a, b, "same (seed, index) must reproduce");
    assert!(a.windows(2).all(|w| w[0] != w[1]));
}

#[test]
fn sampler_respects_its_seed() {
    let mut a = rng::Sampler::new(5);
    let mut b = rng::Sampler::new(5);
    let mut c = rng::Sampler::new(6);
    let xa: Vec<f64> = (0..20).map(|_| a.normal(0.0, 1.0)).collect();
    let xb: Vec<f64> = (0..20).map(|_| b.normal(0.0, 1.0)).collect();
    let xc: Vec<f64> = (0..20).map(|_| c.normal(0.0, 1.0)).collect();
    assert_eq!(xa, xb);
    assert_ne!(xa, xc);
}

#[test]
fn permutation_is_a_permutation() {
    let mut s = rng::Sampler::new(23);
    let mut p = s.permutation(50);
    p.sort_unstable();
    assert_eq!(p, (0..50).collect::<Vec<usize>>());
}

#[test]
fn sampling_without_replacement_returns_distinct_elements() {
    let mut s = rng::Sampler::new(29);
    let pool: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let picked = s.choice_without_replacement(&pool, 8);
    assert_eq!(picked.len(), 8);
    let mut sorted = picked.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    assert_eq!(sorted.len(), 8, "draws must be distinct");
}
