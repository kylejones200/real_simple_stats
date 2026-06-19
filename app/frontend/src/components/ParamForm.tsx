import { useState } from "react";
import type { ParamSchema } from "../types";

function parseArray(raw: string): number[] {
  return raw
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => !isNaN(n));
}

interface ParamFormProps {
  params: ParamSchema[];
  sampleData: Record<string, unknown> | undefined;
  onSubmit: (values: Record<string, unknown>) => void;
  loading: boolean;
}

export function ParamForm({ params, sampleData, onSubmit, loading }: ParamFormProps) {
  const initState = (): Record<string, unknown> => {
    const s: Record<string, unknown> = {};
    for (const p of params) {
      if (p.type === "multi_array") {
        s[p.name] = [[], []] as number[][];
      } else if (p.type === "array" || p.type === "array2d") {
        s[p.name] = "";
      } else {
        s[p.name] = p.default ?? "";
      }
    }
    return s;
  };

  const [rawValues, setRawValues] = useState<Record<string, unknown>>(initState);
  const [multiGroups, setMultiGroups] = useState<Record<string, string[]>>(() => {
    const init: Record<string, string[]> = {};
    for (const p of params) {
      if (p.type === "multi_array") init[p.name] = ["", "", ""];
    }
    return init;
  });

  const loadSample = () => {
    if (!sampleData) return;
    const next = { ...rawValues };
    for (const p of params) {
      const val = sampleData[p.name];
      if (val === undefined) continue;
      if (p.type === "multi_array") {
        const groups = val as unknown[][];
        setMultiGroups((prev) => ({
          ...prev,
          [p.name]: groups.map((g) => (g as number[]).join(", ")),
        }));
      } else if (p.type === "array" || p.type === "array2d") {
        next[p.name] = Array.isArray(val)
          ? (val as number[][]).map
            ? (val as number[]).join(", ")
            : (val as number[][]).map((row) => (row as number[]).join(", ")).join("\n")
          : String(val);
      } else {
        next[p.name] = val;
      }
    }
    setRawValues(next);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const out: Record<string, unknown> = {};
    for (const p of params) {
      const raw = rawValues[p.name];
      if (p.type === "array") {
        out[p.name] = parseArray(raw as string);
      } else if (p.type === "array2d") {
        out[p.name] = (raw as string)
          .trim()
          .split("\n")
          .map((row) => parseArray(row));
      } else if (p.type === "multi_array") {
        out[p.name] = (multiGroups[p.name] ?? []).map(parseArray);
      } else if (p.type === "float") {
        const v = parseFloat(raw as string);
        out[p.name] = isNaN(v) ? undefined : v;
      } else if (p.type === "int") {
        const v = parseInt(raw as string);
        out[p.name] = isNaN(v) ? undefined : v;
      } else if (p.type === "bool") {
        out[p.name] = Boolean(raw);
      } else {
        out[p.name] = raw === "" ? undefined : raw;
      }
    }
    // Remove undefined optional params
    const cleaned = Object.fromEntries(
      Object.entries(out).filter(([, v]) => v !== undefined && v !== "")
    );
    onSubmit(cleaned);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {sampleData && (
        <button
          type="button"
          onClick={loadSample}
          className="text-xs text-indigo-600 hover:text-indigo-800 font-medium border border-indigo-200 rounded px-3 py-1"
        >
          Load sample data
        </button>
      )}

      {params.map((p) => {
        if (p.type === "multi_array") {
          const groups = multiGroups[p.name] ?? [];
          return (
            <div key={p.name}>
              <label className="block text-sm font-medium text-slate-700 mb-1">{p.label}</label>
              {groups.map((g, i) => (
                <div key={i} className="flex gap-2 mb-2">
                  <textarea
                    rows={2}
                    placeholder={`Group ${i + 1}: 5.1, 5.3, 5.0, …`}
                    value={g}
                    onChange={(e) => {
                      const next = [...groups];
                      next[i] = e.target.value;
                      setMultiGroups((prev) => ({ ...prev, [p.name]: next }));
                    }}
                    className="flex-1 text-sm font-mono border border-slate-300 rounded px-3 py-2 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-400"
                  />
                  {groups.length > 2 && (
                    <button
                      type="button"
                      onClick={() =>
                        setMultiGroups((prev) => ({
                          ...prev,
                          [p.name]: groups.filter((_, j) => j !== i),
                        }))
                      }
                      className="text-red-400 hover:text-red-600 text-xs px-1"
                    >
                      ✕
                    </button>
                  )}
                </div>
              ))}
              <button
                type="button"
                onClick={() =>
                  setMultiGroups((prev) => ({ ...prev, [p.name]: [...groups, ""] }))
                }
                className="text-xs text-slate-500 hover:text-slate-700 border border-slate-200 rounded px-2 py-1"
              >
                + Add group
              </button>
            </div>
          );
        }

        if (p.type === "array" || p.type === "array2d") {
          return (
            <div key={p.name}>
              <label className="block text-sm font-medium text-slate-700 mb-1">{p.label}</label>
              <textarea
                rows={p.type === "array2d" ? 4 : 2}
                placeholder={p.type === "array2d" ? "One row per line: 30, 10\n5, 25" : "5.1, 4.9, 5.3, 5.0, …"}
                value={rawValues[p.name] as string}
                onChange={(e) => setRawValues((prev) => ({ ...prev, [p.name]: e.target.value }))}
                className="w-full text-sm font-mono border border-slate-300 rounded px-3 py-2 resize-y focus:outline-none focus:ring-2 focus:ring-indigo-400"
              />
            </div>
          );
        }

        if (p.type === "bool") {
          return (
            <div key={p.name} className="flex items-center gap-3">
              <input
                type="checkbox"
                id={p.name}
                checked={!!rawValues[p.name]}
                onChange={(e) => setRawValues((prev) => ({ ...prev, [p.name]: e.target.checked }))}
                className="w-4 h-4 accent-indigo-600"
              />
              <label htmlFor={p.name} className="text-sm font-medium text-slate-700">{p.label}</label>
            </div>
          );
        }

        return (
          <div key={p.name}>
            <label className="block text-sm font-medium text-slate-700 mb-1">{p.label}</label>
            <input
              type={p.type === "float" || p.type === "int" ? "number" : "text"}
              step={p.type === "float" ? "any" : undefined}
              value={rawValues[p.name] as string | number}
              onChange={(e) => setRawValues((prev) => ({ ...prev, [p.name]: e.target.value }))}
              className="w-full text-sm font-mono border border-slate-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </div>
        );
      })}

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-medium py-2.5 rounded-lg transition-colors"
      >
        {loading ? "Running…" : "Run"}
      </button>
    </form>
  );
}
