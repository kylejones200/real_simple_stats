interface PlotViewerProps {
  png: string | null;
  loading: boolean;
}

export function PlotViewer({ png, loading }: PlotViewerProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 bg-slate-50 rounded-lg border border-slate-200">
        <span className="text-slate-400 text-sm animate-pulse">Generating plot…</span>
      </div>
    );
  }
  if (!png) return null;
  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden bg-white">
      <img
        src={`data:image/png;base64,${png}`}
        alt="Statistical plot"
        className="w-full h-auto"
      />
    </div>
  );
}
