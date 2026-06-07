export interface RAGResponse {
  answer: string
  retrieval_latency_ms: number
  llm_latency_ms: number
  total_latency_ms: number
  low_similarity: boolean
  context_count: number
}

export interface NonRAGResponse {
  answer: string
  llm_latency_ms: number
}

export interface MLPrediction {
  priority: number
  priority_label: string
  confidence: number
  urgent_probability: number
  latency_ms: number
}

export interface LLMZeroShot {
  priority: number
  priority_label: string
  latency_ms: number
}

export interface SourceTicket {
  text: string
  priority: number
  similarity_score: number
  tweet_id: number
}

export interface QueryResponse {
  query: string
  rag: RAGResponse | null
  non_rag: NonRAGResponse | null
  ml_prediction: MLPrediction | null
  llm_zeroshot: LLMZeroShot | null
  error: string | null
}
