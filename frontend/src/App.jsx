import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import ShipmentDetail from "./pages/ShipmentDetail";
import Analytics from "./pages/Analytics";

export default function App() {
  return (
    <BrowserRouter>
      <nav style={{ display: "flex", gap: 20, padding: 12, borderBottom: "1px solid #888" }}>
        <strong>Supply Prescript</strong>
        <Link to="/">Dashboard</Link>
        <Link to="/analytics">Analytics</Link>
      </nav>
      <div style={{ padding: 20 }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/shipment/:id" element={<ShipmentDetail />} />
          <Route path="/analytics" element={<Analytics />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}