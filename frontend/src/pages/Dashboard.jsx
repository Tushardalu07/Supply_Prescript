import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Package, Gauge, Clock, Building2, ArrowRight } from "lucide-react";
import { getAlerts } from "../services/api";

const color = { High: "var(--high)", Medium: "var(--medium)", Low: "var(--low)" };
const days = (n) => `${n} ${n === 1 ? "day" : "days"}`;

export default function Dashboard() {
  const [alerts, setAlerts] = useState([]);
  const [filter, setFilter] = useState("All");

  useEffect(() => { getAlerts().then(setAlerts); }, []);

  const shown = filter === "All" ? alerts : alerts.filter((a) => a.risk_level === filter);
  const high = alerts.filter((a) => a.risk_level === "High").length;
  const avg = alerts.length ? Math.round((alerts.reduce((s, a) => s + a.delay_probability, 0) / alerts.length) * 100) : 0;

  const stats = [
  { label: "Active Alerts", value: alerts.length, icon: Package, g: "g1" },
  { label: "High Risk", value: high, icon: AlertTriangle, g: "g2" },
  { label: "Avg Delay Probability", value: `${avg}%`, icon: Gauge, g: "g3" },
];

  return (
    <>
     <div className="hero">
  <div>
    <h1>Shipment Delay Alerts</h1>
    <p>AI-predicted delays with smart, cost-aware recommendations</p>
  </div>
  <div className="hero-pill"><span className="live-dot" /> Live monitoring</div>
</div>

      <div className="stats">
        {stats.map((s, i) => (
          <motion.div key={s.label} className={`stat ${s.g}`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }} whileHover={{ y: -4, scale: 1.02 }}>
  <div className="stat-icon"><s.icon size={22} /></div>
  <div><b>{s.value}</b><span>{s.label}</span></div>
</motion.div>
        ))}
      </div>

      <div className="filters">
        {["All", "High", "Medium", "Low"].map((f) => (
          <button key={f} className={`chip ${filter === f ? "on" : ""}`} onClick={() => setFilter(f)}>{f}</button>
        ))}
      </div>

      <AnimatePresence mode="popLayout">
        {shown.map((a, i) => (
          <motion.div key={a.shipment_id} layout initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95 }} transition={{ delay: i * 0.08 }} whileHover={{ y: -3 }}>
            <Link to={`/shipment/${a.shipment_id}`} className="alert" style={{ "--c": color[a.risk_level] }}>
              <div className="alert-top">
                <h3>{a.shipment_id} · {a.product}</h3>
                <span className={`badge ${a.risk_level === "High" ? "pulse" : ""}`}>{a.risk_level} Risk</span>
              </div>
              <div className="meta">
                <span><Building2 size={15} /> {a.supplier}</span>
                <span><Clock size={15} /> Expected delay: {days(a.expected_delay_days)}</span>
              </div>
              <div className="bar-label"><span>Delay probability</span><b>{Math.round(a.delay_probability * 100)}%</b></div>
              <div className="bar">
                <motion.div initial={{ width: 0 }} animate={{ width: `${a.delay_probability * 100}%` }} transition={{ duration: 0.9, delay: 0.2 + i * 0.08 }} />
              </div>
              <div className="go">View recommendations <ArrowRight size={16} /></div>
            </Link>
          </motion.div>
        ))}
      </AnimatePresence>
    </>
  );
}