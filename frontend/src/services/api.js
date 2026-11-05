import axios from "axios";

const BASE_URL = "http://localhost:8000";

// Fetch shipments for Dashboard
export const getAlerts = async () => {
  const response = await axios.get(BASE_URL + "/shipments");
  return response.data.shipments;
};

export const getShipments = async () => {
  const response = await axios.get(BASE_URL + "/shipments");
  return response.data;
};

// Create a shipment
export const createShipment = async (shipment) => {
  const response = await axios.post(BASE_URL + "/shipments", shipment);
  return response.data;
};

// Fetch recommendations
export const getRecommendations = async (id) => {
  const response = await axios.get(
    BASE_URL + "/shipments/" + encodeURIComponent(id) + "/recommendations"
  );
  return response.data;
};

// Execute selected decision
export const executeDecision = async (id, decisionId) => {
  const response = await axios.post(
    BASE_URL + "/shipments/" + encodeURIComponent(id) + "/execute",
    { decision_id: decisionId }
  );
  return response.data;
};

// Fetch decision outcomes for Analytics
export const getOutcomes = async () => {
  try {
    const response = await axios.get(BASE_URL + "/outcomes");
    return response.data.outcomes || [];
  } catch (error) {
    console.error("Unable to fetch decision outcomes:", error);
    return [];
  }
};