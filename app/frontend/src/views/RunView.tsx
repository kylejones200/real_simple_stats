import { useEffect } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { useAppStore } from "../stores/useAppStore";
import { TestPicker } from "../components/TestPicker";
import { ParamForm } from "../components/ParamForm";
import { ResultCard } from "../components/ResultCard";
import type { ExplainedResult } from "../types";

export function RunView() {
  const { testName } = useParams<{ testName?: string }>();
  const [searchParams] = useSearchParams();
  const moduleFilter = searchParams.get("module");

  const {
    tests, selectedTest, result, loading, error,
    setTest, runExplained, runRaw,
  } = useAppStore();

  useEffect(() => {
    if (testName && testName !== selectedTest) {
      setTest(testName);
    }
  }, [testName]);

  const meta = tests.find((t) => t.name === (selectedTest ?? testName));

  const handleSubmit = (params: Record<string, unknown>) => {
    if (!meta) return;
    if (meta.explained) {
      runExplained(meta.name, params);
    } else {
      runRaw(meta.name, params);
    }
  };

  const isExplained = (r: unknown): r is ExplainedResult =>
    typeof r === "object" && r !== null && "title" in r && "caveats" in r;

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left: test picker */}
      <div className="w-72 shrink-0 border-r border-slate-200 bg-white p-4 overflow-y-auto">
        <TestPicker />
      </div>

      {/* Right: form + result */}
      <div className="flex-1 overflow-y-auto">
        {!meta ? (
          <div className="flex items-center justify-center h-full text-slate-400 text-sm">
            Select a test to get started
          </div>
        ) : (
          <div className="p-6 max-w-3xl mx-auto">
            <div className="bg-white border border-slate-200 rounded-2xl p-6 mb-6">
              <h2 className="text-lg font-bold text-slate-900 mb-1">{meta.label}</h2>
              <p className="text-xs text-slate-400 mb-4">
                {meta.explained ? "✦ Self-explaining result" : "Raw result (dict)"} · {meta.module.replace(/_/g, " ")}
              </p>
              <ParamForm
                params={meta.params}
                sampleData={meta.sample_data}
                onSubmit={handleSubmit}
                loading={loading}
              />
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6">
                <p className="text-sm text-red-700 font-medium">Error</p>
                <p className="text-sm text-red-600 mt-1 font-mono">{error}</p>
              </div>
            )}

            {result && isExplained(result) && (
              <div className="bg-white border border-slate-200 rounded-2xl p-6">
                <ResultCard
                  result={result}
                  testName={meta.name}
                  params={{}}
                  module={meta.module}
                />
              </div>
            )}

            {result && !isExplained(result) && (
              <div className="bg-white border border-slate-200 rounded-2xl p-6">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">Result</h3>
                <pre className="text-xs font-mono text-slate-700 bg-slate-50 rounded-lg p-4 overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
