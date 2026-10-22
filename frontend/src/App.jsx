import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Truck, LayoutDashboard, BarChart3, Moon, Sun } from "lucide-react";
import Dashboard from "./pages/Dashboard";
import ShipmentDetail from "./pages/ShipmentDetail";
import Analytics from "./pages/Analytics";

function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        className="container"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.25 }}
      >
        <Routes location={location}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/shipment/:id" element={<ShipmentDetail />} />
          <Route path="/analytics" element={<Analytics />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  );
}

export default function App() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("theme") || "light"; } catch { return "light"; }
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch { /* ignore */ }
  }, [theme]);

  return (
    <BrowserRouter>
      <div className="blobs">
        <div className="blob b1" /><div className="blob b2" /><div className="blob b3" /><div className="blob b4" />
      </div>
      <nav className="nav">
        <div className="logo">
          <div className="logo-icon"><Truck size={18} /></div> Supply Prescript
        </div>
        <NavLink to="/" end className="nav-link"><LayoutDashboard size={16} /> Dashboard</NavLink>
        <NavLink to="/analytics" className="nav-link"><BarChart3 size={16} /> Analytics</NavLink>
        <button className="theme-btn" onClick={() => setTheme(theme === "light" ? "dark" : "light")} title="Toggle theme">
          {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
        </button>
      </nav>
      <AnimatedRoutes />
    </BrowserRouter>
  );
}