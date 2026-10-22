import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRecommendations, executeDecision } from "../services/api";

export default function ShipmentDetail() {
  const { id } = useParams();
  const [options, setOptions] = useState([]);
  const [selected, setSelected] = useState(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    getRecommendations(id).then((d) => setOptions(d.options));
  }, [id]);

  const handleExecute = async () => {
    if (!selected) return;
    if (!window.confirm("Is decision ko execute karna hai?")) return;
    const res = await executeDecision(id, selected);
    setMessage(res.status === "success" ? "Decision executed successfully!" : "Something went wrong");
  };

  return (
    <div style={{ textAlign: "left" }}>
      <Link to="/">← Back</Link>
      <h2>Recommendations for {id}</h2>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        {options.map((o) => (
          <div
            key={o.id}
            onClick={() => setSelected(o.id)}
            style={{ border: selected === o.id ? "3px solid #2563eb" : "1px solid #888", borderRadius: 8, padding: 16, width: 220, cursor: "pointer" }}
          >
            <h3>{o.type}</h3>
            <p>Extra Cost: ₹{o.extra_cost.toLocaleString()}</p>
            <p>Delivery: {o.delivery_days} days</p>
            <p>Score: {o.score}</p>
          </div>
        ))}
      </div>
      <button onClick={handleExecute} disabled={!selected} style={{ marginTop: 20, padding: "10px 20px" }}>
        Execute Decision
      </button>
      {message && <p style={{ color: "green" }}>{message}</p>}
    </div>
  );
}