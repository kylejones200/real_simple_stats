import real_simple_stats as rss

# Each entry: callable fn, module group, human label, explained flag,
# param schema list, and sample data for one-click demo.
TESTS: dict[str, dict] = {
    # ── Explained wrappers ────────────────────────────────────────────────
    "one_sample_t_test_explained": {
        "fn": rss.one_sample_t_test_explained,
        "module": "hypothesis_testing",
        "label": "One-Sample t-Test",
        "explained": True,
        "params": [
            {"name": "data", "type": "array", "label": "Sample values", "required": True},
            {"name": "mu", "type": "float", "label": "Null mean (μ₀)", "required": True, "default": 0.0},
            {"name": "alpha", "type": "float", "label": "α (significance level)", "required": False, "default": 0.05},
        ],
        "sample_data": {"data": [5.2, 5.4, 5.1, 5.5, 5.3, 5.0, 4.9], "mu": 5.0, "alpha": 0.05},
    },
    "one_way_anova_explained": {
        "fn": rss.one_way_anova_explained,
        "module": "hypothesis_testing",
        "label": "One-Way ANOVA",
        "explained": True,
        "params": [
            {"name": "groups", "type": "multi_array", "label": "Groups (one per box)", "required": True},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {
            "groups": [
                [48.2, 52.1, 50.3, 49.8, 51.2],
                [55.6, 58.2, 56.9, 57.4, 59.1],
                [63.5, 65.2, 64.1, 66.3, 62.8],
            ],
            "alpha": 0.05,
        },
    },
    "chi_square_independence_explained": {
        "fn": rss.chi_square_independence_explained,
        "module": "hypothesis_testing",
        "label": "Chi-Square Independence",
        "explained": True,
        "params": [
            {"name": "observed", "type": "array2d", "label": "Contingency table (rows, comma-separated)", "required": True},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {"observed": [[30, 10], [5, 25]], "alpha": 0.05},
    },
    "difference_in_differences_explained": {
        "fn": rss.difference_in_differences_explained,
        "module": "causal_inference",
        "label": "Difference-in-Differences",
        "explained": True,
        "params": [
            {"name": "outcome", "type": "array", "label": "Outcome values", "required": True},
            {"name": "post", "type": "array", "label": "Post-period indicator (0/1)", "required": True},
            {"name": "treated", "type": "array", "label": "Treatment indicator (0/1)", "required": True},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {
            "outcome": [100, 102, 104, 101, 103, 98, 110, 115, 112, 108],
            "post":    [0,   0,   0,   0,   0,   1,  1,   1,   1,   1  ],
            "treated": [0,   0,   0,   1,   1,   0,  0,   1,   1,   1  ],
            "alpha": 0.05,
        },
    },
    "kaplan_meier_explained": {
        "fn": rss.kaplan_meier_explained,
        "module": "survival",
        "label": "Kaplan-Meier Survival",
        "explained": True,
        "params": [
            {"name": "durations", "type": "array", "label": "Time to event or censoring", "required": True},
            {"name": "event_observed", "type": "array", "label": "Event observed (1=yes, 0=censored)", "required": True},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {
            "durations": [5, 12, 8, 20, 3, 15, 9, 18, 6, 25],
            "event_observed": [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
            "alpha": 0.05,
        },
    },
    "morans_i_explained": {
        "fn": rss.morans_i_explained,
        "module": "spatial_stats",
        "label": "Moran's I (Spatial Autocorrelation)",
        "explained": True,
        "params": [
            {"name": "x", "type": "array", "label": "X coordinates", "required": True},
            {"name": "y", "type": "array", "label": "Y coordinates", "required": True},
            {"name": "values", "type": "array", "label": "Values at each location", "required": True},
            {"name": "distance_threshold", "type": "float", "label": "Distance threshold (neighbours)", "required": False, "default": None},
        ],
        "sample_data": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "values": [10, 11, 9, 12, 10, 2, 1, 3, 2, 1],
            "distance_threshold": 3.0,
        },
    },
    "detect_change_points_explained": {
        "fn": rss.detect_change_points_explained,
        "module": "time_series",
        "label": "Change Point Detection",
        "explained": True,
        "params": [
            {"name": "data", "type": "array", "label": "Time series values", "required": True},
            {"name": "n_breaks", "type": "int", "label": "Number of breaks to find", "required": False, "default": 1},
        ],
        "sample_data": {
            "data": [2.1, 2.3, 1.9, 2.2, 2.0, 2.1, 2.3, 2.0, 8.5, 8.2, 8.8, 8.6, 8.4, 8.9, 8.7, 8.5],
            "n_breaks": 1,
        },
    },

    # ── Raw functions ──────────────────────────────────────────────────────
    "one_way_anova": {
        "fn": rss.one_way_anova,
        "module": "hypothesis_testing",
        "label": "One-Way ANOVA (raw)",
        "explained": False,
        "params": [
            {"name": "groups", "type": "multi_array", "label": "Groups (one per box)", "required": True},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {
            "groups": [[48.2, 52.1, 50.3], [55.6, 58.2, 56.9], [63.5, 65.2, 64.1]],
            "alpha": 0.05,
        },
    },
    "chi_square_independence": {
        "fn": rss.chi_square_independence,
        "module": "hypothesis_testing",
        "label": "Chi-Square Independence (raw)",
        "explained": False,
        "params": [
            {"name": "observed", "type": "array2d", "label": "Contingency table (rows, comma-separated)", "required": True},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {"observed": [[30, 10], [5, 25]], "alpha": 0.05},
    },
    "autocorrelation": {
        "fn": rss.autocorrelation,
        "module": "time_series",
        "label": "Autocorrelation (ACF)",
        "explained": False,
        "params": [
            {"name": "data", "type": "array", "label": "Time series values", "required": True},
            {"name": "max_lag", "type": "int", "label": "Max lag", "required": False, "default": 20},
        ],
        "sample_data": {"data": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]},
    },
    "exponential_smoothing": {
        "fn": rss.exponential_smoothing,
        "module": "time_series",
        "label": "Exponential Smoothing",
        "explained": False,
        "params": [
            {"name": "data", "type": "array", "label": "Time series values", "required": True},
            {"name": "alpha", "type": "float", "label": "Smoothing factor α (0–1)", "required": False, "default": 0.3},
        ],
        "sample_data": {"data": [10, 12, 11, 14, 13, 15, 14, 17, 16, 18], "alpha": 0.3},
    },
    "linear_regression": {
        "fn": rss.linear_regression,
        "module": "regression",
        "label": "Simple Linear Regression",
        "explained": False,
        "params": [
            {"name": "X", "type": "array", "label": "Predictor (X)", "required": True},
            {"name": "y", "type": "array", "label": "Response (y)", "required": True},
        ],
        "sample_data": {
            "X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9],
        },
    },
    "pearson_correlation": {
        "fn": rss.pearson_correlation,
        "module": "regression",
        "label": "Pearson Correlation",
        "explained": False,
        "params": [
            {"name": "X", "type": "array", "label": "Variable 1", "required": True},
            {"name": "y", "type": "array", "label": "Variable 2", "required": True},
        ],
        "sample_data": {
            "X": [1, 2, 3, 4, 5],
            "y": [2, 4, 5, 4, 5],
        },
    },
    "cohens_d": {
        "fn": rss.cohens_d,
        "module": "effect_sizes",
        "label": "Cohen's d (Effect Size)",
        "explained": False,
        "params": [
            {"name": "group1", "type": "array", "label": "Group 1", "required": True},
            {"name": "group2", "type": "array", "label": "Group 2", "required": True},
        ],
        "sample_data": {
            "group1": [5.1, 4.9, 5.3, 5.0, 5.2],
            "group2": [5.8, 6.1, 5.9, 6.3, 5.7],
        },
    },
    "bootstrap": {
        "fn": rss.bootstrap,
        "module": "resampling",
        "label": "Bootstrap CI",
        "explained": False,
        "params": [
            {"name": "data", "type": "array", "label": "Data", "required": True},
            {"name": "statistic_fn", "type": "str", "label": "Statistic (mean / median / std)", "required": False, "default": "mean"},
            {"name": "n_bootstrap", "type": "int", "label": "Bootstrap iterations", "required": False, "default": 2000},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {
            "data": [12.5, 13.1, 11.9, 14.2, 12.8, 13.5, 11.5, 14.8],
            "statistic_fn": "mean",
        },
    },
    "kaplan_meier": {
        "fn": rss.kaplan_meier,
        "module": "survival",
        "label": "Kaplan-Meier (raw)",
        "explained": False,
        "params": [
            {"name": "durations", "type": "array", "label": "Durations", "required": True},
            {"name": "event_observed", "type": "array", "label": "Event (1=yes, 0=censored)", "required": True},
        ],
        "sample_data": {
            "durations": [5, 12, 8, 20, 3, 15, 9, 18, 6, 25],
            "event_observed": [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        },
    },
    "morans_i": {
        "fn": rss.morans_i,
        "module": "spatial_stats",
        "label": "Moran's I (raw)",
        "explained": False,
        "params": [
            {"name": "x", "type": "array", "label": "X coordinates", "required": True},
            {"name": "y", "type": "array", "label": "Y coordinates", "required": True},
            {"name": "values", "type": "array", "label": "Values", "required": True},
            {"name": "distance_threshold", "type": "float", "label": "Distance threshold", "required": False, "default": None},
        ],
        "sample_data": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "values": [10, 11, 9, 12, 10, 2, 1, 3, 2, 1],
            "distance_threshold": 3.0,
        },
    },
    "difference_in_differences": {
        "fn": rss.difference_in_differences,
        "module": "causal_inference",
        "label": "Difference-in-Differences (raw)",
        "explained": False,
        "params": [
            {"name": "outcome", "type": "array", "label": "Outcome", "required": True},
            {"name": "post", "type": "array", "label": "Post-period (0/1)", "required": True},
            {"name": "treated", "type": "array", "label": "Treated (0/1)", "required": True},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {
            "outcome": [100, 102, 104, 101, 103, 98, 110, 115, 112, 108],
            "post":    [0,   0,   0,   0,   0,   1,  1,   1,   1,   1  ],
            "treated": [0,   0,   0,   1,   1,   0,  0,   1,   1,   1  ],
        },
    },
    "five_number_summary": {
        "fn": rss.five_number_summary,
        "module": "descriptive",
        "label": "Five-Number Summary",
        "explained": False,
        "params": [
            {"name": "values", "type": "array", "label": "Data values", "required": True},
        ],
        "sample_data": {"values": [3, 7, 8, 5, 12, 14, 21, 13, 18]},
    },
    "power_t_test": {
        "fn": rss.power_t_test,
        "module": "power_analysis",
        "label": "Power: t-Test",
        "explained": False,
        "params": [
            {"name": "n", "type": "int", "label": "Sample size per group", "required": True},
            {"name": "effect_size", "type": "float", "label": "Cohen's d", "required": True},
            {"name": "alpha", "type": "float", "label": "α", "required": False, "default": 0.05},
        ],
        "sample_data": {"n": 30, "effect_size": 0.5, "alpha": 0.05},
    },
}


MODULE_GROUPS = {
    "hypothesis_testing": {"label": "Hypothesis Testing", "color": "indigo"},
    "causal_inference":   {"label": "Causal Inference",   "color": "violet"},
    "survival":           {"label": "Survival Analysis",  "color": "rose"},
    "spatial_stats":      {"label": "Spatial Stats",      "color": "teal"},
    "time_series":        {"label": "Time Series",        "color": "amber"},
    "regression":         {"label": "Regression",         "color": "blue"},
    "effect_sizes":       {"label": "Effect Sizes",       "color": "green"},
    "resampling":         {"label": "Resampling",         "color": "orange"},
    "descriptive":        {"label": "Descriptive",        "color": "slate"},
    "power_analysis":     {"label": "Power Analysis",     "color": "purple"},
}
