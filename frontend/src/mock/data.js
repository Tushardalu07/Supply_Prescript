export const alerts = [
  { shipment_id: "SH-1021", supplier: "ABC Ltd", product: "Electronics Batch A", delay_probability: 0.82, expected_delay_days: 6, risk_level: "High" },
  { shipment_id: "SH-1022", supplier: "Global Parts Co", product: "Steel Components", delay_probability: 0.55, expected_delay_days: 3, risk_level: "Medium" },
  { shipment_id: "SH-1023", supplier: "FastTrade Inc", product: "Packaging Material", delay_probability: 0.21, expected_delay_days: 1, risk_level: "Low" },
];

export const recommendations = {
  "SH-1021": [
    { id: 1, type: "Air Freight", extra_cost: 12000, delivery_days: 2, score: 0.91 },
    { id: 2, type: "Secondary Supplier", extra_cost: 7000, delivery_days: 4, score: 0.78 },
    { id: 3, type: "Delay Launch", extra_cost: 0, delivery_days: 9, score: 0.55 },
  ],
  "SH-1022": [
    { id: 1, type: "Air Freight", extra_cost: 9000, delivery_days: 2, score: 0.80 },
    { id: 2, type: "Secondary Supplier", extra_cost: 5000, delivery_days: 3, score: 0.84 },
    { id: 3, type: "Delay Launch", extra_cost: 0, delivery_days: 7, score: 0.50 },
  ],
  "SH-1023": [
    { id: 1, type: "Delay Launch", extra_cost: 0, delivery_days: 2, score: 0.88 },
    { id: 2, type: "Secondary Supplier", extra_cost: 3000, delivery_days: 2, score: 0.60 },
  ],
};

export const outcomes = [
  { shipment_id: "SH-1001", decision: "Air Freight", predicted_cost: 11000, actual_cost: 11500, predicted_days: 2, actual_days: 2 },
  { shipment_id: "SH-1002", decision: "Secondary Supplier", predicted_cost: 7000, actual_cost: 6800, predicted_days: 4, actual_days: 5 },
  { shipment_id: "SH-1003", decision: "Delay Launch", predicted_cost: 0, actual_cost: 0, predicted_days: 9, actual_days: 8 },
];