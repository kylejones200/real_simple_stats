import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { HomeView } from "./views/HomeView";
import { RunView } from "./views/RunView";
import { useAppStore } from "./stores/useAppStore";

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  );
}

export default function App() {
  const loadTests = useAppStore((s) => s.loadTests);

  useEffect(() => {
    loadTests();
  }, []);

  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<HomeView />} />
          <Route path="/run" element={<RunView />} />
          <Route path="/run/:testName" element={<RunView />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
