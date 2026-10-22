import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, Plane, Factory, CalendarClock, Sparkles, Rocket, CheckCircle2 } from "lucide-react";
import { getRecommendations, executeDecision } from "../services/api";

const icons = { "Air Freight": Plane, "Secondary Supplier": Factory, "Delay Launch": CalendarClock };
const theme = { "Air Freight": "#0ea5e9", "Secondary Supplier": "#a855f7", "Delay Launch": "#f59e0b" };

export default function ShipmentDetail() {
  const { id } = useParams();
  const [options, setOptions] = useState([]);
  const [selected, setSelected] = useState(null);
  const [confirm, setConfirm] = useState(false);
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => { getRecommendations(id).then((d) => setOptions(d.options)); }, [id]);

  const best = options.length ? Math.max(...options.map((o) => o.score)) : 0;
  const chosen = options.find((o) => o.id === selected);

  const run = async () => {
    setLoading(true);
    const res = await executeDecision(id, selected);
    setLoading(false);
    if (res.status === "success") setDone(true);
  };

  return (
    <>
      <Link to="/" className="back"><ArrowLeft size={16} /> Back to alerts</Link>
      <h1 className="page-title">Recommendations for {id}</h1>
      <p className="sub">Compare cost and delivery time, then pick an action</p>

      <div className="options">
        {options.map((o, i) => {
          const Icon = icons[o.type] || Rocket;
          return (
            <motion.div key={o.id} style={{ "--o": theme[o.type] || "#6366f1" }} className={`option ${selected === o.id ? "sel" : ""}`}
              initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.12 }}
              whileHover={{ y: -6, scale: 1.02 }} whileTap={{ scale: 0.98 }}
              onClick={() => !done && setSelected(o.id)}>
              {o.score === best && <div className="tag"><Sparkles size={12} /> Best match</div>}
              <h3><span className="o-icon"><Icon size={22} /></span> {o.type}</h3>
              <div className="row"><span>Extra cost</span><b>₹{o.extra_cost.toLocaleString("en-IN")}</b></div>
              <div className="row"><span>Delivery time</span><b>{o.delivery_days} {o.delivery_days === 1 ? "day" : "days"}</b></div>
              <div className="row"><span>Score</span><b>{Math.round(o.score * 100)}/100</b></div>
            </motion.div>
          );
        })}
      </div>

      {done ? (
        <motion.p initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="good" style={{ marginTop: 28, display: "flex", gap: 8, alignItems: "center" }}>
          <CheckCircle2 size={20} /> Decision executed: {chosen?.type}
        </motion.p>
      ) : (
        <motion.button className="btn" disabled={!selected} onClick={() => setConfirm(true)} whileHover={{ scale: selected ? 1.04 : 1 }} whileTap={{ scale: 0.96 }}>
          <Rocket size={18} /> Execute Decision
        </motion.button>
      )}

      <AnimatePresence>
        {confirm && (
          <motion.div className="overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => !loading && setConfirm(false)}>
            <motion.div className="modal" initial={{ scale: 0.85, y: 30 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, opacity: 0 }} onClick={(e) => e.stopPropagation()}>
              {done ? (
                <>
                  <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 260 }}>
                    <CheckCircle2 size={64} color="#10b981" />
                  </motion.div>
                  <h3>Decision executed!</h3>
                  <p>{chosen?.type} has been saved for {id}.</p>
                  <div className="btns"><Link to="/analytics" className="btn">View Analytics</Link></div>
                </>
              ) : (
                <>
                  <h3>Confirm decision</h3>
                  <p>Execute <b>{chosen?.type}</b> for {id}? Extra cost: ₹{chosen?.extra_cost.toLocaleString("en-IN")}</p>
                  <div className="btns">
                    <button className="btn ghost" onClick={() => setConfirm(false)}>Cancel</button>
                    <button className="btn" onClick={run} disabled={loading}>{loading ? "Executing..." : "Confirm"}</button>
                  </div>
                </>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}