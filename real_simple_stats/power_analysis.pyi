"""Type stubs for power_analysis module."""

def power_t_test(
    n: int | None = None,
    delta: float | None = None,
    sd: float = 1.0,
    sig_level: float = 0.05,
    power: float | None = None,
    alternative: str = "two-sided",
) -> dict[str, float]: ...
def power_proportion_test(
    n: int | None = None,
    p1: float | None = None,
    p0: float = 0.5,
    sig_level: float = 0.05,
    power: float | None = None,
    alternative: str = "two-sided",
) -> dict[str, float]: ...
def power_anova(
    n_groups: int,
    n_per_group: int | None = None,
    effect_size: float | None = None,
    sig_level: float = 0.05,
    power: float | None = None,
) -> dict[str, float]: ...
def power_correlation(
    n: int | None = None,
    r: float | None = None,
    sig_level: float = 0.05,
    power: float | None = None,
    alternative: str = "two-sided",
) -> dict[str, float]: ...
def minimum_detectable_effect(
    n: int, sig_level: float = 0.05, power: float = 0.8, test_type: str = "t-test"
) -> float: ...
def sample_size_summary(
    effect_size: float, power: float = 0.8, sig_level: float = 0.05
) -> dict[str, int]: ...
