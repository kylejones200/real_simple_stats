export interface ParamSchema {
  name: string;
  type: "array" | "array2d" | "multi_array" | "float" | "int" | "bool" | "str";
  label: string;
  default?: unknown;
  required?: boolean;
}

export interface TestMeta {
  name: string;
  label: string;
  module: string;
  explained: boolean;
  params: ParamSchema[];
  has_sample_data: boolean;
  sample_data?: Record<string, unknown>;
}

export interface ExplainedResult {
  title: string;
  question: string;
  values: Record<string, unknown>;
  intuition: string;
  interpretation: string;
  assumptions: Record<string, unknown>;
  caveats: string[];
  next_steps: string[];
  decision: string | null;
  has_plot: boolean;
}

export type RunResult = ExplainedResult | Record<string, unknown>;

export interface ModuleGroup {
  label: string;
  color: string;
}
