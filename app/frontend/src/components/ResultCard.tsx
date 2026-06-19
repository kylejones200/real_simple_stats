import { useState } from "react";
import type { ExplainedResult } from "../types";
import { PlotViewer } from "./PlotViewer";
import { useAppStore } from "../stores/useAppStore";

const MODULE_COLORS: Record<string, string> = {
  hypothesis_testing: "bg-indigo-100 text-indigo-800",
  causal_inference:   "bg-violet-100 text-violet-800",
  survival:           "bg-rose-100 text-rose-800",
  spatial_stats:      "bg-teal-100 text-teal-800",
  time_series:        "bg-amber-100 text-amber-800",
  regression:         "bg-blue-100 text-blue-800",
  effect_sizes:       "bg-green-100 text-green-800",
  resampling:         "bg-orange-100 text-orange-800",
  descriptive:        "bg-slate-100 text-slate-800",
  power_analysis:     "bg-purple-100 text-purple-800",
};

function DecisionBadge({ decision }: { decision: string | null }) {
  if (!decision) return null;
  const reject = decision.toLowerCase().startsWith("reject");
  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold ${
      reject ? "bg-green-100 text-green-800" : "bg-amber-100 text-amber-800"
    }`}>
      {decision}
    </span>
  );
}

function ValuesTable({ values, alpha = 0.05 }: { values: Record<string, unknown>; alpha?: number }) {
  const formatVal = (v: unknown): string => {
    if (v === null || v === undefined) return "—";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (typeof v === "number") {
      if (Number.isInteger(v)) return v.toString();
      return Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(3) : v.toFixed(4);
    }
    if (Array.isArray(v)) {
      if (v.length <= 2) return `(${v.map((x) => (typeof x === "number" ? x.toFixed(4) : x)).join(", ")})`;
      return `[${v.slice(0, 3).map((x) => (typeof x === "number" ? x.toFixed(3) : x)).join(", ")}...]`;
    }
    return String(v);
  };

  const skip = new Set(["decision"]);

  return (
    <table className="w-full text-sm">
      <tbody>
        {Object.entries(values)
          .filter(([k]) => !skip.has(k))
          .map(([key, val]) => {
            const isPval = key === "p_value" && typeof val === "number";
            const pHighlight = isPval
              ? (val as number) < alpha
                ? "text-green-700 font-semibold"
                : "text-red-600 font-semibold"
              : "";
            return (
              <tr key={key} className="border-b border-slate-100 last:border-0">
                <td className="py-1.5 pr-4 text-slate-500 font-mono text-xs w-40">{key}</td>
                <td className={`py-1.5 font-mono ${pHighlight || "text-slate-800"}`}>
                  {formatVal(val)}
                </td>
              </tr>
            );
          })}
      </tbody>
    </table>
  );
}

function Collapsible({ title, children, defaultOpen = false, warning = false }: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  warning?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 hover:bg-slate-100 text-left"
      >
        <span className="flex items-center gap-2 text-sm font-medium text-slate-700">
          {warning && <span className="text-amber-500">⚠</span>}
          {title}
        </span>
        <span className="text-slate-400 text-xs">{open ? "▲" : "▼"}</span>
      </button>
      {open && <div className="px-4 py-3 text-sm text-slate-600 leading-relaxed">{children}</div>}
    </div>
  );
}

interface ResultCardProps {
  result: ExplainedResult;
  testName: string;
  params: Record<string, unknown>;
  module?: string;
}

export function ResultCard({ result, testName, params, module }: ResultCardProps) {
  const { fetchPlot, plotPng, plotLoading } = useAppStore();
  const [showPlot, setShowPlot] = useState(false);

  const moduleColor = module ? (MODULE_COLORS[module] ?? "bg-slate-100 text-slate-700") : "bg-slate-100 text-slate-700";

  const assumptionWarning =
    typeof result.assumptions?.summary === "string" &&
    (result.assumptions.summary.toLowerCase().includes("caution") ||
      result.assumptions.summary.toLowerCase().includes("violat") ||
      result.assumptions.summary.toLowerCase().includes("may not"));

  const handleShowPlot = () => {
    if (!plotPng) fetchPlot(testName, params);
    setShowPlot(true);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">{result.title}</h2>
          {module && (
            <span className={`mt-1 inline-block text-xs font-medium px-2 py-0.5 rounded ${moduleColor}`}>
              {module.replace(/_/g, " ")}
            </span>
          )}
        </div>
        {result.decision && <DecisionBadge decision={result.decision} />}
      </div>

      {/* Question */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-lg px-4 py-3">
        <p className="text-sm italic text-indigo-900">{result.question}</p>
      </div>

      {/* Values table */}
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Result</h3>
        <ValuesTable values={result.values} />
      </div>

      {/* Interpretation — always open */}
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          What this result means
        </h3>
        <p className="text-sm text-slate-700 leading-relaxed">{result.interpretation}</p>
      </div>

      {/* Collapsibles */}
      <Collapsible title="How this test works" defaultOpen={false}>
        <p className="leading-relaxed">{result.intuition}</p>
      </Collapsible>

      <Collapsible
        title="Assumptions"
        defaultOpen={assumptionWarning}
        warning={assumptionWarning}
      >
        {typeof result.assumptions?.summary === "string" && (
          <p className="mb-2">{result.assumptions.summary}</p>
        )}
        {Object.entries(result.assumptions)
          .filter(([k]) => k !== "summary")
          .map(([k, v]) => (
            <div key={k} className="flex gap-2 text-xs font-mono mb-1">
              <span className="text-slate-400 w-32 shrink-0">{k}</span>
              <span>{String(v)}</span>
            </div>
          ))}
      </Collapsible>

      {/* Caveats — always visible */}
      {result.caveats.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
            What this does NOT mean
          </h3>
          {result.caveats.map((c, i) => (
            <div key={i} className="flex gap-3 bg-amber-50 border-l-4 border-amber-400 px-3 py-2 rounded-r-lg">
              <span className="text-amber-500 mt-0.5 shrink-0">•</span>
              <p className="text-sm text-amber-900">{c}</p>
            </div>
          ))}
        </div>
      )}

      {/* Next steps */}
      {result.next_steps.length > 0 && (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Next steps</h3>
          <ol className="space-y-1">
            {result.next_steps.map((s, i) => (
              <li key={i} className="flex gap-2 text-sm text-slate-700">
                <span className="text-indigo-400 font-bold shrink-0">{i + 1}.</span>
                {s.replace(/^→\s*/, "")}
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Plot */}
      {result.has_plot && (
        <div>
          {!showPlot ? (
            <button
              onClick={handleShowPlot}
              className="flex items-center gap-2 text-sm text-indigo-600 hover:text-indigo-800 font-medium"
            >
              <span>📊</span> Show plot
            </button>
          ) : (
            <PlotViewer png={plotPng} loading={plotLoading} />
          )}
        </div>
      )}
    </div>
  );
}
