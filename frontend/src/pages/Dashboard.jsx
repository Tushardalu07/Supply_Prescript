import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getAlerts } from "../services/api";

const colors = { High: "#dc2626", Medium: "#f59e0b", Low: "#16a34a" };

export default function Dashboard() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    getAlerts().then(setAlerts);
  }, []);

  return (
    <div>
      <h2>Shipment Delay Alerts</h2>
      {alerts.map((a) => (
        <Link key={a.shipment_id} to={`/shipment/${a.shipment_id}`} style={{ textDecoration: "none", color: "inherit" }}>
          <div style={{ border: `2px solid ${colors[a.risk_level]}`, borderRadius: 8, padding: 16, margin: "12px 0", textAlign: "left" }}>
            <strong>{a.shipment_id}</strong> - {a.product}
            <span style={{ float: "right", color: colors[a.risk_level], fontWeight: "bold" }}>{a.risk_level} Risk</span>
            <div>Supplier: {a.supplier}</div>
           <div>Delay Probability: {(a.delay_probability * 100).toFixed(0)}% | Expected Delay: {a.expected_delay_days} {a.expected_delay_days === 1 ? "day" : "days"}</div>
          </div>
        </Link>
      ))}
    </div>
  );
}