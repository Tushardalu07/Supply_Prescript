import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from "recharts";
import { getOutcomes } from "../services/api";

export default function Analytics() {
  const [data, setData] = useState([]);

  useEffect(() => {
    getOutcomes().then(setData);
  }, []);

  return (
    <div>
      <h2>Decision Outcomes: Predicted vs Actual</h2>
      <BarChart width={650} height={320} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="shipment_id" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="predicted_cost" fill="#93c5fd" name="Predicted Cost" />
        <Bar dataKey="actual_cost" fill="#2563eb" name="Actual Cost" />
      </BarChart>
    </div>
  );
}