import axios from "axios";
import { alerts, recommendations, outcomes } from "../mock/data";

const USE_MOCK = true;
const BASE_URL = "http://localhost:8000/api";

export const getAlerts = async () =>
  USE_MOCK ? alerts : (await axios.get(`${BASE_URL}/shipments/alerts`)).data;

export const getRecommendations = async (id) =>
  USE_MOCK
    ? { options: recommendations[id] || [] }
    : (await axios.get(`${BASE_URL}/shipments/${id}/recommendations`)).data;

export const executeDecision = async (shipment_id, option_id) =>
  USE_MOCK
    ? { status: "success", shipment_id, option_id }
    : (await axios.post(`${BASE_URL}/decisions/execute`, { shipment_id, option_id })).data;

export const getOutcomes = async () =>
  USE_MOCK ? outcomes : (await axios.get(`${BASE_URL}/decisions/outcomes`)).data;