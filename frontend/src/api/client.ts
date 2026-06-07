import axios from 'axios'

const BASE_URL = '/api'

export const submitQuery = async (query: string, top_k: number = 5) => {
  const response = await axios.post(`${BASE_URL}/query`, { query, top_k })
  return response.data
}

export const checkHealth = async () => {
  const response = await axios.get(`${BASE_URL}/health`)
  return response.data
}

export const getLogs = async (n: number = 10) => {
  const response = await axios.get(`${BASE_URL}/logs?n=${n}`)
  return response.data
}

export const getStats = async () => {
  const response = await axios.get(`${BASE_URL}/stats`)
  return response.data
}
