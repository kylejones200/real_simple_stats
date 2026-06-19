import { useState } from "react";
import { useNavigate } from "react-router-dom";

interface Node {
  question?: string;
  choices?: { label: string; next: Node }[];
  leaf?: { test: string; name: string; rationale: string };
}

const TREE: Node = {
  question: "What kind of outcome variable do you have?",
  choices: [
    {
      label: "Continuous (numbers: heights, revenues, scores)",
      next: {
        question: "What are you trying to do?",
        choices: [
          {
            label: "Compare one group to a reference value",
            next: { leaf: { test: "one_sample_t_test_explained", name: "One-Sample t-Test", rationale: "Tests whether your sample mean differs from a known reference. Always pair with Cohen's d for effect size." } },
          },
          {
            label: "Compare two independent groups",
            next: { leaf: { test: "two_sample_t_test", name: "Two-Sample t-Test", rationale: "Compares means between two groups. Use Welch (equal_var=False) by default — it's robust to unequal variances." } },
          },
          {
            label: "Compare before/after on the same subjects",
            next: { leaf: { test: "paired_t_test", name: "Paired t-Test", rationale: "Pairing removes between-subject noise, increasing power over a two-sample t-test." } },
          },
          {
            label: "Compare three or more groups",
            next: { leaf: { test: "one_way_anova_explained", name: "One-Way ANOVA", rationale: "Tests whether any of k group means differ. A significant F only tells you some group differs — run post-hoc tests to find which." } },
          },
          {
            label: "Identify a causal effect",
            next: {
              question: "What's your data structure?",
              choices: [
                {
                  label: "Pre/post measurements + a control group",
                  next: { leaf: { test: "difference_in_differences_explained", name: "Difference-in-Differences", rationale: "Compares the change in treated units to the change in controls. Requires parallel pre-treatment trends." } },
                },
                {
                  label: "Treatment assigned by crossing a numerical threshold",
                  next: { leaf: { test: "difference_in_differences", name: "Regression Discontinuity", rationale: "Near the cutoff, units just above and below are approximately exchangeable. Use regression_discontinuity()." } },
                },
                {
                  label: "Repeated observations on the same entities",
                  next: { leaf: { test: "difference_in_differences", name: "Panel Fixed Effects", rationale: "Within-entity demeaning removes time-invariant confounders. Use panel_fixed_effects()." } },
                },
              ],
            },
          },
        ],
      },
    },
    {
      label: "Categorical (yes/no, grades, product types)",
      next: {
        question: "What are you testing?",
        choices: [
          {
            label: "Whether two categorical variables are associated",
            next: { leaf: { test: "chi_square_independence_explained", name: "Chi-Square Independence", rationale: "Tests whether row and column categories are independent. Reports Cramér's V for effect size." } },
          },
          {
            label: "Whether counts match expected proportions",
            next: { leaf: { test: "chi_square_independence_explained", name: "Chi-Square Goodness-of-Fit", rationale: "Compares observed counts to theoretical proportions. Use chi_square_statistic(observed, expected)." } },
          },
        ],
      },
    },
    {
      label: "Time-to-event (days to churn, failure, recovery)",
      next: {
        question: "What do you need?",
        choices: [
          {
            label: "Describe the survival curve (some data may be censored)",
            next: { leaf: { test: "kaplan_meier_explained", name: "Kaplan-Meier", rationale: "The correct way to compute median survival time with right-censored data. Never average durations ignoring censored observations." } },
          },
          {
            label: "Fit a smooth curve and extrapolate beyond observation window",
            next: { leaf: { test: "kaplan_meier", name: "Parametric Survival", rationale: "Use compare_survival_models() to fit Weibull/Lognormal/Exponential/Log-logistic via MLE and rank by AIC." } },
          },
        ],
      },
    },
    {
      label: "Spatial (measurements at geographic locations)",
      next: {
        question: "What spatial question are you asking?",
        choices: [
          {
            label: "Do similar values cluster together in space?",
            next: { leaf: { test: "morans_i_explained", name: "Moran's I", rationale: "Global autocorrelation coefficient. I ≈ +1 = clustering, I ≈ 0 = random, I ≈ −1 = dispersion. Choose distance_threshold carefully." } },
          },
          {
            label: "How does autocorrelation decay with distance?",
            next: { leaf: { test: "morans_i", name: "Variogram", rationale: "Use compute_variogram() then fit_variogram(). The range parameter tells you beyond what distance values are spatially uncorrelated." } },
          },
        ],
      },
    },
    {
      label: "Sequential / time series (ordered over time)",
      next: {
        question: "What do you want to know?",
        choices: [
          {
            label: "When did the mean shift?",
            next: { leaf: { test: "detect_change_points_explained", name: "Change Point Detection", rationale: "Binary segmentation finds breakpoints that most reduce within-segment variance. Validate against known events." } },
          },
          {
            label: "Smooth or forecast the series",
            next: { leaf: { test: "one_sample_t_test_explained", name: "Exponential Smoothing", rationale: "Use exponential_smoothing() for level-only series or double_exponential_smoothing() when there's a trend." } },
          },
          {
            label: "Check for autocorrelation",
            next: { leaf: { test: "one_sample_t_test_explained", name: "ACF / PACF", rationale: "Use autocorrelation(data, max_lag=20) and partial_autocorrelation(data, max_lag=20)." } },
          },
        ],
      },
    },
  ],
};

interface StepProps {
  node: Node;
  onLeaf: (test: string) => void;
}

function Step({ node, onLeaf }: StepProps) {
  const [chosen, setChosen] = useState<Node | null>(null);

  if (node.leaf) {
    return (
      <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-5 mt-4">
        <p className="text-xs font-semibold text-indigo-400 uppercase tracking-wide mb-1">Recommended test</p>
        <h3 className="text-lg font-bold text-indigo-900 mb-2">{node.leaf.name}</h3>
        <p className="text-sm text-indigo-700 mb-4">{node.leaf.rationale}</p>
        <button
          onClick={() => onLeaf(node.leaf!.test)}
          className="bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
        >
          Run this test →
        </button>
      </div>
    );
  }

  return (
    <div className="mt-4">
      <p className="text-sm font-medium text-slate-700 mb-3">{node.question}</p>
      <div className="space-y-2">
        {node.choices?.map((c) => (
          <button
            key={c.label}
            onClick={() => setChosen(c.next)}
            className={`w-full text-left px-4 py-3 rounded-lg text-sm border transition-colors ${
              chosen === c.next
                ? "bg-indigo-600 text-white border-indigo-600"
                : "bg-white text-slate-700 border-slate-200 hover:border-indigo-300 hover:bg-indigo-50"
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>
      {chosen && <Step node={chosen} onLeaf={onLeaf} />}
    </div>
  );
}

export function DecisionTree() {
  const navigate = useNavigate();
  const [started, setStarted] = useState(false);

  const handleLeaf = (testName: string) => {
    navigate(`/run/${testName}`);
  };

  return (
    <div className="max-w-2xl">
      <h2 className="text-xl font-bold text-slate-900 mb-1">Which test should I use?</h2>
      <p className="text-sm text-slate-500 mb-6">
        Answer the questions below. Each leaf maps to an exact function and explains why.
      </p>
      {!started ? (
        <button
          onClick={() => setStarted(true)}
          className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-lg transition-colors"
        >
          Start →
        </button>
      ) : (
        <Step node={TREE} onLeaf={handleLeaf} />
      )}
    </div>
  );
}
