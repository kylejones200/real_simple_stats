//! Special functions: the numerical foundation for every distribution.
//!
//! Algorithms are the standard high-accuracy ones (Lanczos log-gamma, the
//! Numerical Recipes series/continued-fraction pair for the incomplete gamma,
//! Lentz's continued fraction for the incomplete beta, Wichura's AS241 for the
//! normal quantile). Target accuracy is ~1e-14 relative against SciPy.

use std::f64::consts::PI;

pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
pub const SQRT_2PI: f64 = 2.506_628_274_631_000_5;
/// 1/sqrt(2*pi). Multiplying by this is both faster and better rounded than
/// dividing by SQRT_2PI, which costs an extra ulp at the peak of the density.
pub const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
pub const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// Lanczos coefficients (g = 7, n = 9); good to ~15 significant digits.
const LANCZOS_G: f64 = 7.0;
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural log of |Gamma(x)|, for x > 0.
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        if x == 0.0 {
            return f64::INFINITY;
        }
        // Reflection for negative arguments: Gamma(x)Gamma(1-x) = pi/sin(pi x)
        if x < 0.0 {
            let s = (PI * x).sin().abs();
            if s == 0.0 {
                return f64::INFINITY;
            }
            return (PI / s).ln() - ln_gamma(1.0 - x);
        }
        return f64::NAN;
    }
    if x < 0.5 {
        let s = (PI * x).sin();
        return (PI / s.abs()).ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut acc = LANCZOS[0];
    for (i, c) in LANCZOS.iter().enumerate().skip(1) {
        acc += c / (z + i as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    LN_SQRT_2PI + (z + 0.5) * t.ln() - t + acc.ln()
}

/// Gamma(x).
pub fn gamma_fn(x: f64) -> f64 {
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma_fn(1.0 - x))
    } else {
        ln_gamma(x).exp()
    }
}

/// log B(a, b) — the log Beta function.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Regularized lower incomplete gamma P(a, x) = gamma(a,x)/Gamma(a).
pub fn gammainc_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gser(a, x)
    } else {
        1.0 - gcf(a, x)
    }
}

/// Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x), computed without
/// cancellation in the tail.
pub fn gammainc_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gser(a, x)
    } else {
        gcf(a, x)
    }
}

/// Series representation for P(a, x); converges fast when x < a + 1.
fn gser(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..1000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Modified Lentz continued fraction for Q(a, x); converges fast when x >= a + 1.
fn gcf(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Regularized incomplete beta I_x(a, b).
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b)).exp();
    // Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) to stay in the fast-converging regime.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - (b * (1.0 - x).ln() + a * x.ln() - ln_beta(a, b)).exp() * betacf(b, a, 1.0 - x) / b
    }
}

/// Lentz continued fraction for the incomplete beta.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        // odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Error function.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return x; // preserves -0.0
    }
    let ax = x.abs();
    let v = if ax < 1.0 {
        gammainc_p(0.5, ax * ax)
    } else {
        1.0 - gammainc_q(0.5, ax * ax)
    };
    if x < 0.0 {
        -v
    } else {
        v
    }
}

/// Complementary error function, accurate deep into the tail.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < 1.0 {
        return 1.0 - gammainc_p(0.5, x * x);
    }
    // Lentz evaluation of the continued fraction
    //   erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
    // Using the exact 1/sqrt(pi) constant keeps this free of ln_gamma round-off.
    const TINY: f64 = 1e-300;
    const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;
    let mut f = TINY;
    let mut c = f;
    let mut d = 0.0f64;
    for i in 0..300 {
        let (a_i, b_i) = if i == 0 {
            (1.0, x)
        } else {
            (0.5 * i as f64, x)
        };
        d = b_i + a_i * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b_i + a_i / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = c * d;
        f *= del;
        if (del - 1.0).abs() < 1e-17 {
            break;
        }
    }
    // exp(-x^2) is the accuracy bottleneck in the far tail: rounding x*x costs a
    // relative x^2*eps (~9e-14 near x=20). Recover the exact residual of the
    // square with an FMA and fold it back as a first-order factor.
    let x2 = x * x;
    let r = x.mul_add(x, -x2); // exact: x*x - x2
    f * INV_SQRT_PI * (-x2).exp() * (1.0 - r)
}

/// Standard normal CDF.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc_scaled(-x)
}

/// Standard normal survival function (1 - CDF), accurate deep into the upper tail.
pub fn norm_sf(x: f64) -> f64 {
    0.5 * erfc_scaled(x)
}

/// erfc(x / sqrt(2)) with the argument reduction done in double-double.
///
/// Naively computing `erfc(x / SQRT_2)` caps far-tail accuracy at ~2z^2 * eps
/// (~2e-13 near x = 29) because the rounded quotient is already wrong in its
/// last bit and erfc amplifies that by 2z^2. Splitting 1/sqrt(2) into hi+lo and
/// recovering the exact product residual with an FMA removes that ceiling.
fn erfc_scaled(x: f64) -> f64 {
    // 1/sqrt(2) split into a double-double: HI is the nearest f64 (the std
    // constant), LO is the remainder that HI drops.
    const INV_SQRT2_HI: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const INV_SQRT2_LO: f64 = -4.833_646_656_726_457e-17;
    const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

    let z = x * INV_SQRT2_HI;
    if !z.is_finite() || z <= 0.0 {
        // Lower tail and non-finite inputs are well conditioned; no correction needed.
        return erfc(z);
    }
    // Exact residual of the product, plus the neglected low word of 1/sqrt(2).
    let e = x.mul_add(INV_SQRT2_HI, -z) + x * INV_SQRT2_LO;
    let base = erfc(z);
    // First-order correction: d/dz erfc(z) = -2/sqrt(pi) * exp(-z^2)
    let corr = e * TWO_OVER_SQRT_PI * (-z * z).exp();
    base - corr
}

/// Standard normal PDF.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() * FRAC_1_SQRT_2PI
}

/// Standard normal quantile — Wichura's AS241 (PPND16), ~1e-16 accurate.
pub fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 8] = [
        3.387_132_872_796_366_6e0,
        1.331_416_678_917_843_8e2,
        1.971_590_950_306_551_4e3,
        1.373_169_376_550_946e4,
        4.592_195_393_154_987e4,
        6.726_577_092_700_87e4,
        3.343_057_558_358_813e4,
        2.509_080_928_730_122_7e3,
    ];
    const B: [f64; 8] = [
        1.0,
        4.231_333_070_160_091e1,
        6.871_870_074_920_579e2,
        5.394_196_021_424_751e3,
        2.121_379_430_158_659_6e4,
        3.930_789_580_009_271e4,
        2.872_908_573_572_194_3e4,
        5.226_495_278_852_854e3,
    ];
    const C: [f64; 8] = [
        1.423_437_110_749_683_6e0,
        4.630_337_846_156_546,
        5.769_497_221_460_691,
        3.647_848_324_763_204_6e0,
        1.270_458_252_452_368_4e0,
        2.417_807_251_774_506e-1,
        2.272_384_498_926_918_5e-2,
        7.745_450_142_783_414e-4,
    ];
    const D: [f64; 8] = [
        1.0,
        2.053_191_626_637_759,
        1.676_384_830_183_803_8e0,
        6.897_673_349_851e-1,
        1.481_039_764_274_800_8e-1,
        1.519_866_656_361_645_7e-2,
        5.475_938_084_995_345e-4,
        1.050_750_071_644_416_8e-9,
    ];
    const E: [f64; 8] = [
        6.657_904_643_501_104e0,
        5.463_784_911_164_114e0,
        1.784_826_539_917_291_3e0,
        2.965_605_718_285_048_9e-1,
        2.653_218_952_657_612_3e-2,
        1.242_660_947_388_078_4e-3,
        2.711_555_568_743_487_6e-5,
        2.010_334_399_292_288_1e-7,
    ];
    const F: [f64; 8] = [
        1.0,
        5.998_322_065_558_88e-1,
        1.369_298_809_227_358e-1,
        1.487_536_129_085_061_5e-2,
        7.868_691_311_456_133e-4,
        1.846_318_317_510_054_7e-5,
        1.421_511_758_316_446e-7,
        2.044_263_103_389_939_8e-15,
    ];

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * poly(r, &A) / poly(r, &B);
    }
    let r0 = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r0.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        poly(r, &C) / poly(r, &D)
    } else {
        let r = r - 5.0;
        poly(r, &E) / poly(r, &F)
    };
    if q < 0.0 {
        -val
    } else {
        val
    }
}

/// Horner evaluation of a polynomial given coefficients in ascending... (descending here).
#[inline]
fn poly(x: f64, c: &[f64]) -> f64 {
    // Coefficients are listed lowest-order first in the AS241 tables above, so
    // evaluate accordingly.
    let mut acc = 0.0;
    for &ci in c.iter().rev() {
        acc = acc * x + ci;
    }
    acc
}

/// Inverse of the regularized incomplete beta: find x with I_x(a,b) = p.
/// Newton refinement guarded by bisection so it cannot escape the bracket.
pub fn betaincinv(a: f64, b: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    // I_x(a,b) = 1 - I_{1-x}(b,a). When p > 1/2 the residual `betainc(x) - p`
    // is a difference of two numbers near 1 and cancels catastrophically, so
    // solve the mirrored problem, which stays in the well-conditioned tail.
    if p > 0.5 {
        return 1.0 - betaincinv(b, a, 1.0 - p);
    }
    // Initial guess (Abramowitz & Stegun 26.5.22 style), then safeguarded Newton.
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    let mut x = {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut xg = t - (2.30753 + 0.27061 * t) / (1.0 + (0.99229 + 0.04481 * t) * t);
        if p < 0.5 {
            xg = -xg;
        }
        let al = (xg * xg - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = (xg * (al + h).sqrt() / h)
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        let cand = a / (a + b * (2.0 * w).exp());
        if cand.is_finite() && cand > 0.0 && cand < 1.0 {
            cand
        } else {
            0.5
        }
    };
    let lnb = ln_beta(a, b);
    for _ in 0..200 {
        let err = betainc(a, b, x) - p;
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        // Converge on RELATIVE error in p: an absolute 1e-15 test is far too
        // loose when p itself is ~1e-8.
        if err.abs() <= 1e-16 * p {
            break;
        }
        // Newton step: d/dx I_x(a,b) = x^(a-1) (1-x)^(b-1) / B(a,b)
        let d = ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - lnb).exp();
        let mut nx = if d > 0.0 && d.is_finite() {
            x - err / d
        } else {
            f64::NAN
        };
        if !(nx.is_finite() && nx > lo && nx < hi) {
            nx = 0.5 * (lo + hi);
        }
        if (nx - x).abs() <= f64::EPSILON * x.abs() {
            x = nx;
            break;
        }
        x = nx;
    }
    x
}

/// Inverse of the regularized lower incomplete gamma: find x with P(a, x) = p.
pub fn gammaincinv(a: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        // Same cancellation argument as betaincinv: solve Q(a,x) = 1-p instead.
        return gammainccinv(a, 1.0 - p);
    }
    // Bracket by expansion, then safeguarded Newton.
    let mut lo = 0.0f64;
    let mut hi = a.max(1.0);
    while gammainc_p(a, hi) < p {
        lo = hi;
        hi *= 2.0;
        if hi > 1e300 {
            return f64::INFINITY;
        }
    }
    let mut x = 0.5 * (lo + hi);
    let lg = ln_gamma(a);
    for _ in 0..200 {
        let err = gammainc_p(a, x) - p;
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        if err.abs() <= 1e-16 * p {
            break;
        }
        // d/dx P(a,x) = x^(a-1) e^-x / Gamma(a)
        let d = ((a - 1.0) * x.ln() - x - lg).exp();
        let mut nx = if d > 0.0 && d.is_finite() {
            x - err / d
        } else {
            f64::NAN
        };
        if !(nx.is_finite() && nx > lo && nx < hi) {
            nx = 0.5 * (lo + hi);
        }
        if (nx - x).abs() <= f64::EPSILON * x.abs() {
            x = nx;
            break;
        }
        x = nx;
    }
    x
}

/// Inverse of the regularized UPPER incomplete gamma: find x with Q(a, x) = q.
///
/// Kept separate from `gammaincinv` so upper-tail solves never form the
/// cancelling residual `P(a,x) - p` with both terms near 1.
pub fn gammainccinv(a: f64, q: f64) -> f64 {
    if q <= 0.0 {
        return f64::INFINITY;
    }
    if q >= 1.0 {
        return 0.0;
    }
    let mut lo = 0.0f64;
    let mut hi = a.max(1.0);
    while gammainc_q(a, hi) > q {
        lo = hi;
        hi *= 2.0;
        if hi > 1e300 {
            return f64::INFINITY;
        }
    }
    let mut x = 0.5 * (lo + hi);
    let lg = ln_gamma(a);
    for _ in 0..200 {
        let err = gammainc_q(a, x) - q;
        // Q is decreasing in x.
        if err > 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        if err.abs() <= 1e-16 * q {
            break;
        }
        // d/dx Q(a,x) = -x^(a-1) e^-x / Gamma(a)
        let d = -((a - 1.0) * x.ln() - x - lg).exp();
        let mut nx = if d != 0.0 && d.is_finite() {
            x - err / d
        } else {
            f64::NAN
        };
        if !(nx.is_finite() && nx > lo && nx < hi) {
            nx = 0.5 * (lo + hi);
        }
        if (nx - x).abs() <= f64::EPSILON * x.abs() {
            x = nx;
            break;
        }
        x = nx;
    }
    x
}
