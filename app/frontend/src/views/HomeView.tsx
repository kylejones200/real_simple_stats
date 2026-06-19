import { DecisionTree } from "../components/DecisionTree";
import { useNavigate } from "react-router-dom";
import { useAppStore } from "../stores/useAppStore";

export function HomeView() {
  const { tests } = useAppStore();
  const navigate = useNavigate();

  const explained = tests.filter((t) => t.explained);

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Real Simple Stats</h1>
        <p className="text-slate-500 text-base">
          Statistics that explain themselves — what the test does, what the result means,
          and what it does <em>not</em> mean.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        {/* Decision tree */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6">
          <DecisionTree />
        </div>

        {/* Quick-start: explained functions */}
        <div>
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
            Self-explaining tests
          </h2>
          <div className="space-y-2">
            {explained.map((t) => (
              <button
                key={t.name}
                onClick={() => navigate(`/run/${t.name}`)}
                className="w-full text-left bg-white border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50 rounded-lg px-4 py-3 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-800">{t.label}</span>
                  <span className="text-xs text-indigo-500">→</span>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">{t.module.replace(/_/g, " ")}</p>
              </button>
            ))}
          </div>

          <div className="mt-6">
            <button
              onClick={() => navigate("/run")}
              className="text-sm text-slate-500 hover:text-indigo-600 font-medium"
            >
              Browse all {tests.length} tests →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
