import { create } from "zustand";
import type { TestMeta, ExplainedResult, ModuleGroup } from "../types";

interface AppStore {
  tests: TestMeta[];
  modules: Record<string, ModuleGroup>;
  selectedTest: string | null;
  result: ExplainedResult | Record<string, unknown> | null;
  plotPng: string | null;
  loading: boolean;
  plotLoading: boolean;
  error: string | null;

  loadTests: () => Promise<void>;
  setTest: (name: string | null) => void;
  runExplained: (testName: string, params: Record<string, unknown>) => Promise<void>;
  runRaw: (testName: string, params: Record<string, unknown>) => Promise<void>;
  fetchPlot: (testName: string, params: Record<string, unknown>) => Promise<void>;
  clearResult: () => void;
}

export const useAppStore = create<AppStore>((set) => ({
  tests: [],
  modules: {},
  selectedTest: null,
  result: null,
  plotPng: null,
  loading: false,
  plotLoading: false,
  error: null,

  loadTests: async () => {
    const [testsRes, modsRes] = await Promise.all([
      fetch("/api/tests"),
      fetch("/api/modules"),
    ]);
    const tests = await testsRes.json();
    const modules = await modsRes.json();
    set({ tests, modules });
  },

  setTest: (name) => set({ selectedTest: name, result: null, plotPng: null, error: null }),

  runExplained: async (testName, params) => {
    set({ loading: true, error: null, result: null, plotPng: null });
    try {
      const res = await fetch(`/api/explain/${testName}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Request failed");
      }
      const data = await res.json();
      set({ result: data.result, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  runRaw: async (testName, params) => {
    set({ loading: true, error: null, result: null, plotPng: null });
    try {
      const res = await fetch(`/api/run/${testName}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Request failed");
      }
      const data = await res.json();
      set({ result: data.result, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  fetchPlot: async (testName, params) => {
    set({ plotLoading: true });
    try {
      const res = await fetch(`/api/plot/${testName}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params }),
      });
      if (!res.ok) throw new Error("Plot request failed");
      const data = await res.json();
      set({ plotPng: data.png, plotLoading: false });
    } catch (e) {
      set({ plotLoading: false, error: (e as Error).message });
    }
  },

  clearResult: () => set({ result: null, plotPng: null, error: null }),
}));
