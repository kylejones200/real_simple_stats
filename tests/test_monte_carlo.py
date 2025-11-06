"""Tests for Monte Carlo simulation methods."""

import pytest
import numpy as np
from real_simple_stats.monte_carlo import (
    geometric_brownian_motion,
    monte_carlo_from_data,
    monte_carlo_integration,
    monte_carlo_probability,
)


class TestGeometricBrownianMotion:
    """Tests for geometric_brownian_motion function."""
    
    def test_basic_simulation(self):
        """Test basic GBM simulation."""
        result = geometric_brownian_motion(
            S0=100,
            mu=0.10,
            sigma=0.20,
            T=1.0,
            n_steps=252,
            n_simulations=100,
            random_seed=42
        )
        
        assert 'paths' in result
        assert 'times' in result
        assert 'final_values' in result
        assert 'mean_path' in result
        assert 'percentiles' in result
        assert 'statistics' in result
        
    def test_paths_shape(self):
        """Test that paths have correct shape."""
        n_steps = 100
        n_simulations = 50
        result = geometric_brownian_motion(
            S0=100,
            mu=0.10,
            sigma=0.20,
            T=1.0,
            n_steps=n_steps,
            n_simulations=n_simulations,
            random_seed=42
        )
        
        assert result['paths'].shape == (n_steps + 1, n_simulations)
        assert len(result['times']) == n_steps + 1
        assert len(result['final_values']) == n_simulations
        
    def test_initial_value(self):
        """Test that all paths start at S0."""
        S0 = 100
        result = geometric_brownian_motion(
            S0=S0,
            mu=0.10,
            sigma=0.20,
            T=1.0,
            n_steps=252,
            n_simulations=100,
            random_seed=42
        )
        
        assert np.all(result['paths'][0] == S0)
        
    def test_positive_paths(self):
        """Test that all paths remain positive."""
        result = geometric_brownian_motion(
            S0=100,
            mu=0.10,
            sigma=0.20,
            T=1.0,
            n_steps=252,
            n_simulations=100,
            random_seed=42
        )
        
        assert np.all(result['paths'] > 0)
        
    def test_percentiles(self):
        """Test that percentiles are in correct order."""
        result = geometric_brownian_motion(
            S0=100,
            mu=0.10,
            sigma=0.20,
            T=1.0,
            n_steps=252,
            n_simulations=1000,
            random_seed=42
        )
        
        p = result['percentiles']
        assert p[5] < p[25] < p[50] < p[75] < p[95]
        
    def test_statistics(self):
        """Test that statistics are reasonable."""
        result = geometric_brownian_motion(
            S0=100,
            mu=0.10,
            sigma=0.20,
            T=1.0,
            n_steps=252,
            n_simulations=1000,
            random_seed=42
        )
        
        stats = result['statistics']
        assert stats['min'] < stats['mean'] < stats['max']
        assert stats['std'] > 0
        assert stats['median'] > 0
        
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        result1 = geometric_brownian_motion(
            S0=100, mu=0.10, sigma=0.20, T=1.0,
            n_steps=100, n_simulations=50, random_seed=42
        )
        result2 = geometric_brownian_motion(
            S0=100, mu=0.10, sigma=0.20, T=1.0,
            n_steps=100, n_simulations=50, random_seed=42
        )
        
        np.testing.assert_array_equal(result1['paths'], result2['paths'])
        
    def test_invalid_S0(self):
        """Test that negative S0 raises error."""
        with pytest.raises(ValueError, match="S0 must be positive"):
            geometric_brownian_motion(
                S0=-100, mu=0.10, sigma=0.20, T=1.0
            )
            
    def test_invalid_sigma(self):
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            geometric_brownian_motion(
                S0=100, mu=0.10, sigma=-0.20, T=1.0
            )
            
    def test_invalid_T(self):
        """Test that non-positive T raises error."""
        with pytest.raises(ValueError, match="T must be positive"):
            geometric_brownian_motion(
                S0=100, mu=0.10, sigma=0.20, T=0
            )


class TestMonteCarloFromData:
    """Tests for monte_carlo_from_data function."""
    
    def test_basic_simulation(self):
        """Test basic simulation from data."""
        data = [100, 102, 101, 105, 103, 107, 110, 108, 112, 115]
        result = monte_carlo_from_data(
            data=data,
            n_steps=30,
            n_simulations=100,
            random_seed=42
        )
        
        assert 'paths' in result
        assert 'parameters' in result
        assert 'mu' in result['parameters']
        assert 'sigma' in result['parameters']
        
    def test_estimated_parameters(self):
        """Test that parameters are estimated correctly."""
        data = [100, 102, 101, 105, 103, 107, 110]
        result = monte_carlo_from_data(
            data=data,
            n_steps=30,
            n_simulations=100,
            random_seed=42
        )
        
        params = result['parameters']
        assert params['S0'] == 110  # Last value
        assert params['sigma'] > 0
        
    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        with pytest.raises(ValueError, match="at least 2 values"):
            monte_carlo_from_data(
                data=[100],
                n_steps=30,
                n_simulations=100
            )


class TestMonteCarloIntegration:
    """Tests for monte_carlo_integration function."""
    
    def test_simple_integral(self):
        """Test integration of x^2 from 0 to 1."""
        # Analytical answer: 1/3
        result = monte_carlo_integration(
            func=lambda x: x**2,
            lower_bounds=0,
            upper_bounds=1,
            n_samples=10000,
            random_seed=42
        )
        
        assert 'integral' in result
        assert 'std_error' in result
        assert 'confidence_interval' in result
        
        # Check that estimate is close to 1/3
        assert abs(result['integral'] - 1/3) < 0.01
        
    def test_constant_function(self):
        """Test integration of constant function."""
        # Integral of 5 from 0 to 2 should be 10
        result = monte_carlo_integration(
            func=lambda x: np.ones_like(x) * 5,
            lower_bounds=0,
            upper_bounds=2,
            n_samples=1000,
            random_seed=42
        )
        
        assert abs(result['integral'] - 10) < 0.5
        
    def test_multidimensional_integral(self):
        """Test 2D integration."""
        # Integral of x*y over [0,1]x[0,1] should be 1/4
        result = monte_carlo_integration(
            func=lambda xy: xy[0] * xy[1],
            lower_bounds=[0, 0],
            upper_bounds=[1, 1],
            n_samples=10000,
            random_seed=42
        )
        
        assert abs(result['integral'] - 0.25) < 0.01
        
    def test_confidence_interval(self):
        """Test that confidence interval contains estimate."""
        result = monte_carlo_integration(
            func=lambda x: x**2,
            lower_bounds=0,
            upper_bounds=1,
            n_samples=1000,
            random_seed=42
        )
        
        ci = result['confidence_interval']
        assert ci[0] < result['integral'] < ci[1]
        
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        result1 = monte_carlo_integration(
            func=lambda x: x**2,
            lower_bounds=0,
            upper_bounds=1,
            n_samples=1000,
            random_seed=42
        )
        result2 = monte_carlo_integration(
            func=lambda x: x**2,
            lower_bounds=0,
            upper_bounds=1,
            n_samples=1000,
            random_seed=42
        )
        
        assert result1['integral'] == result2['integral']


class TestMonteCarloProbability:
    """Tests for monte_carlo_probability function."""
    
    def test_simple_probability(self):
        """Test probability estimation."""
        # P(x < 0.5) for x uniform on [0,1] should be 0.5
        result = monte_carlo_probability(
            condition=lambda x: x < 0.5,
            lower_bounds=0,
            upper_bounds=1,
            n_samples=10000,
            random_seed=42
        )
        
        assert 'probability' in result
        assert 'std_error' in result
        assert 'confidence_interval' in result
        assert 'n_successes' in result
        
        # Check that estimate is close to 0.5
        assert abs(result['probability'] - 0.5) < 0.02
        
    def test_circle_probability(self):
        """Test probability for circle (pi estimation)."""
        # P(x^2 + y^2 <= 1) for x,y in [0,1] estimates pi/4
        result = monte_carlo_probability(
            condition=lambda xy: xy[0]**2 + xy[1]**2 <= 1,
            lower_bounds=[0, 0],
            upper_bounds=[1, 1],
            n_samples=10000,
            random_seed=42
        )
        
        pi_estimate = result['probability'] * 4
        assert abs(pi_estimate - np.pi) < 0.1
        
    def test_probability_bounds(self):
        """Test that probability is between 0 and 1."""
        result = monte_carlo_probability(
            condition=lambda x: x < 0.5,
            lower_bounds=0,
            upper_bounds=1,
            n_samples=1000,
            random_seed=42
        )
        
        assert 0 <= result['probability'] <= 1
        
    def test_confidence_interval_bounds(self):
        """Test that confidence interval is within [0,1]."""
        result = monte_carlo_probability(
            condition=lambda x: x < 0.5,
            lower_bounds=0,
            upper_bounds=1,
            n_samples=1000,
            random_seed=42
        )
        
        ci = result['confidence_interval']
        assert 0 <= ci[0] <= 1
        assert 0 <= ci[1] <= 1
        
    def test_n_successes(self):
        """Test that n_successes is reasonable."""
        result = monte_carlo_probability(
            condition=lambda x: x < 0.5,
            lower_bounds=0,
            upper_bounds=1,
            n_samples=1000,
            random_seed=42
        )
        
        assert 0 <= result['n_successes'] <= result['n_samples']
        assert result['probability'] == result['n_successes'] / result['n_samples']


class TestMonteCarloPerformance:
    """Performance tests for Monte Carlo methods."""
    
    def test_gbm_with_numba(self):
        """Test that GBM runs with Numba optimization."""
        result = geometric_brownian_motion(
            S0=100,
            mu=0.10,
            sigma=0.20,
            T=1.0,
            n_steps=252,
            n_simulations=1000,  # Triggers Numba
            random_seed=42
        )
        
        assert result['paths'].shape == (253, 1000)
        
    def test_gbm_without_numba(self):
        """Test that GBM runs without Numba (small simulations)."""
        result = geometric_brownian_motion(
            S0=100,
            mu=0.10,
            sigma=0.20,
            T=1.0,
            n_steps=252,
            n_simulations=50,  # Below Numba threshold
            random_seed=42
        )
        
        assert result['paths'].shape == (253, 50)
