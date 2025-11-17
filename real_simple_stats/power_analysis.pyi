"""Type stubs for power_analysis module."""

from typing import Optional, Dict

def power_t_test(
    n: Optional[int] = None,
    delta: Optional[float] = None,
    sd: float = 1.0,
    sig_level: float = 0.05,
    power: Optional[float] = None,
    alternative: str = "two-sided",
) -> Dict[str, float]: ...
def power_proportion_test(
    n: Optional[int] = None,
    p1: Optional[float] = None,
    p0: float = 0.5,
    sig_level: float = 0.05,
    power: Optional[float] = None,
    alternative: str = "two-sided",
) -> Dict[str, float]: ...
def power_correlation_test(
    n: Optional[int] = None,
    r: Optional[float] = None,
    sig_level: float = 0.05,
    power: Optional[float] = None,
    alternative: str = "two-sided",
) -> Dict[str, float]: ...

