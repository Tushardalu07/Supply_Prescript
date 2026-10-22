import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer } from "recharts";
import { getOutcomes } from "../services/api";

export default function Analytics() {
  const [data, setData] = useState([]);
  useEffect(() => { getOutcomes().then(setData); }, []);

  const totalCost = data.reduce((s, d) => s + d.actual_cost, 0);
  const onTime = data.filter((d) => d.actual_days <= d.predicted_days).length;

  const cards = [
    { label: "Decisions Executed", value: data.length },
    { label: "Total Extra Cost", value: `₹${totalCost.toLocaleString("en-IN")}` },
    { label: "On-time / Better", value: `${onTime}/${data.length}` },
  ];

  return (
    <>
      <h1 className="page-title">Decision Outcomes</h1>
      <p className="sub">Predicted vs actual results of executed decisions</p>

      <div className="stats">
        {cards.map((c, i) => (
          <motion.div key={c.label} className={`stat g${i + 1}`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }} whileHover={{ y: -4, scale: 1.02 }}>
            <div><b>{c.value}</b><span>{c.label}</span></div>
          </motion.div>
        ))}
      </div>

      <motion.div className="panel" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
        <h3>Predicted vs Actual Cost</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="shipment_id" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="predicted_cost" name="Predicted" fill="#f9a8d4" radius={[8, 8, 0, 0]} />
            <Bar dataKey="actual_cost" name="Actual" fill="#6366f1" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      <motion.div className="panel" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.45 }}>
        <h3>Delivery Time Comparison</h3>
        <table>
          <thead><tr><th>Shipment</th><th>Decision</th><th>Predicted</th><th>Actual</th><th>Result</th></tr></thead>
          <tbody>
            {data.map((d) => (
              <tr key={d.shipment_id}>
                <td>{d.shipment_id}</td><td>{d.decision}</td>
                <td>{d.predicted_days} days</td><td>{d.actual_days} days</td>
                <td className={d.actual_days <= d.predicted_days ? "good" : "bad"}>
                  {d.actual_days <= d.predicted_days ? "On track" : `+${d.actual_days - d.predicted_days}d late`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>
    </>
  );
}