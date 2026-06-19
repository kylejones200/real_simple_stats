import { NavLink } from "react-router-dom";
import { useAppStore } from "../stores/useAppStore";

const MODULE_DOT: Record<string, string> = {
  hypothesis_testing: "bg-indigo-500",
  causal_inference:   "bg-violet-500",
  survival:           "bg-rose-500",
  spatial_stats:      "bg-teal-500",
  time_series:        "bg-amber-500",
  regression:         "bg-blue-500",
  effect_sizes:       "bg-green-500",
  resampling:         "bg-orange-500",
  descriptive:        "bg-slate-400",
  power_analysis:     "bg-purple-500",
};

export function Sidebar() {
  const { tests, modules } = useAppStore();

  const byModule = tests.reduce<Record<string, number>>((acc, t) => {
    acc[t.module] = (acc[t.module] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <aside className="w-56 shrink-0 border-r border-slate-200 bg-white flex flex-col h-full">
      <div className="px-4 py-5 border-b border-slate-100">
        <span className="font-bold text-slate-900 text-sm tracking-tight">real simple stats</span>
        <p className="text-xs text-slate-400 mt-0.5">statistics that explain themselves</p>
      </div>

      <nav className="flex-1 overflow-y-auto py-3">
        <NavLink
          to="/"
          className={({ isActive }) =>
            `flex items-center gap-2 px-4 py-2 text-sm ${
              isActive ? "text-indigo-700 font-medium bg-indigo-50" : "text-slate-600 hover:bg-slate-50"
            }`
          }
        >
          <span>🌳</span> Which test?
        </NavLink>
        <NavLink
          to="/run"
          className={({ isActive }) =>
            `flex items-center gap-2 px-4 py-2 text-sm ${
              isActive ? "text-indigo-700 font-medium bg-indigo-50" : "text-slate-600 hover:bg-slate-50"
            }`
          }
        >
          <span>▶</span> Run a test
        </NavLink>

        <div className="mt-4 px-4 mb-1">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Modules</p>
        </div>
        {Object.entries(byModule).map(([mod, count]) => (
          <NavLink
            key={mod}
            to={`/run?module=${mod}`}
            className="flex items-center gap-2 px-4 py-1.5 text-xs text-slate-500 hover:text-slate-800 hover:bg-slate-50"
          >
            <span className={`w-2 h-2 rounded-full shrink-0 ${MODULE_DOT[mod] ?? "bg-slate-300"}`} />
            <span className="flex-1">{modules[mod]?.label ?? mod.replace(/_/g, " ")}</span>
            <span className="text-slate-300">{count}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
