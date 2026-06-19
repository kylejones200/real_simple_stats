import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAppStore } from "../stores/useAppStore";

const MODULE_BADGE: Record<string, string> = {
  hypothesis_testing: "bg-indigo-100 text-indigo-700",
  causal_inference:   "bg-violet-100 text-violet-700",
  survival:           "bg-rose-100 text-rose-700",
  spatial_stats:      "bg-teal-100 text-teal-700",
  time_series:        "bg-amber-100 text-amber-700",
  regression:         "bg-blue-100 text-blue-700",
  effect_sizes:       "bg-green-100 text-green-700",
  resampling:         "bg-orange-100 text-orange-700",
  descriptive:        "bg-slate-100 text-slate-700",
  power_analysis:     "bg-purple-100 text-purple-700",
};

export function TestPicker() {
  const { tests, selectedTest, setTest } = useAppStore();
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  const filtered = tests.filter(
    (t) =>
      t.label.toLowerCase().includes(query.toLowerCase()) ||
      t.module.toLowerCase().includes(query.toLowerCase()) ||
      t.name.toLowerCase().includes(query.toLowerCase())
  );

  const select = (name: string) => {
    setTest(name);
    navigate(`/run/${name}`);
  };

  return (
    <div className="flex flex-col h-full">
      <input
        type="search"
        placeholder="Search tests…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full text-sm border border-slate-300 rounded-lg px-3 py-2 mb-3 focus:outline-none focus:ring-2 focus:ring-indigo-400"
      />
      <div className="overflow-y-auto flex-1 space-y-1">
        {filtered.map((t) => (
          <button
            key={t.name}
            onClick={() => select(t.name)}
            className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors ${
              selectedTest === t.name
                ? "bg-indigo-600 text-white"
                : "hover:bg-slate-100 text-slate-700"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium">{t.label}</span>
              <div className="flex items-center gap-1 shrink-0">
                {t.explained && (
                  <span className="text-xs bg-green-100 text-green-700 px-1.5 py-0.5 rounded font-medium">
                    explained
                  </span>
                )}
                <span className={`text-xs px-1.5 py-0.5 rounded ${
                  selectedTest === t.name
                    ? "bg-indigo-500 text-indigo-100"
                    : (MODULE_BADGE[t.module] ?? "bg-slate-100 text-slate-500")
                }`}>
                  {t.module.replace(/_/g, " ")}
                </span>
              </div>
            </div>
          </button>
        ))}
        {filtered.length === 0 && (
          <p className="text-sm text-slate-400 text-center py-8">No tests match "{query}"</p>
        )}
      </div>
    </div>
  );
}
