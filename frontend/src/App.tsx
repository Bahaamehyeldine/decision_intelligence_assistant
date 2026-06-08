import { useState, useEffect } from 'react'
import { submitQuery, checkHealth } from './api/client'

interface RAGResult {
  answer: string
  retrieval_latency_ms: number
  llm_latency_ms: number
  total_latency_ms: number
  low_similarity: boolean
  context_count: number
}

interface NonRAGResult {
  answer: string
  llm_latency_ms: number
}

interface MLResult {
  priority: number
  priority_label: string
  confidence: number
  urgent_probability: number
  latency_ms: number
}

interface ZeroShotResult {
  priority: number
  priority_label: string
  latency_ms: number
}

interface QueryResult {
  query: string
  rag: RAGResult | null
  non_rag: NonRAGResult | null
  ml_prediction: MLResult | null
  llm_zeroshot: ZeroShotResult | null
  error: string | null
}

function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState('checking...')

  useEffect(() => {
    checkHealth()
      .then((h: {status: string}) => setHealth(h.status))
      .catch(() => setHealth('unhealthy'))
  }, [])

  const handleSubmit = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await submitQuery(query)
      setResult(response as QueryResult)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Something went wrong'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold text-white">Decision Intelligence Assistant</h1>
            <p className="text-sm text-gray-400">RAG + ML Support Ticket Analysis</p>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${health === 'healthy' ? 'bg-green-400' : 'bg-red-400'}`} />
            <span className="text-sm text-gray-400">{health}</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 mb-8">
          <label className="block text-sm font-medium text-gray-300 mb-2">Support Query</label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe your support issue..."
            rows={3}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none"
          />
          <div className="mt-4 flex justify-end">
            <button
              onClick={handleSubmit}
              disabled={loading || !query.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
            >
              {loading ? 'Analyzing...' : 'Analyze Query'}
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {loading && (
          <div className="text-center py-12">
            <div className="inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4" />
            <p className="text-gray-400">Running RAG retrieval, LLM generation, and ML prediction...</p>
          </div>
        )}

        {result && (
          <div className="space-y-6">
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Priority Prediction Comparison</h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-400 mb-1">ML Classifier</p>
                  <div className="flex items-center justify-between">
                    <span className={`text-lg font-bold uppercase ${result.ml_prediction?.priority_label === 'urgent' ? 'text-red-400' : 'text-green-400'}`}>
                      {result.ml_prediction?.priority_label}
                    </span>
                    <span className="text-sm text-gray-400">{result.ml_prediction?.latency_ms}ms</span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">
                    Confidence: {((result.ml_prediction?.confidence ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>

                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-400 mb-1">LLM Zero-shot</p>
                  <div className="flex items-center justify-between">
                    <span className={`text-lg font-bold uppercase ${result.llm_zeroshot?.priority_label === 'urgent' ? 'text-red-400' : 'text-green-400'}`}>
                      {result.llm_zeroshot?.priority_label}
                    </span>
                    <span className="text-sm text-gray-400">{result.llm_zeroshot?.latency_ms}ms</span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">
                    Agreement: {result.ml_prediction?.priority === result.llm_zeroshot?.priority ? '✓ Both agree' : '✗ Disagree'}
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <div className="flex justify-between items-center mb-3">
                  <h2 className="text-lg font-semibold text-white">RAG Answer</h2>
                  <span className="text-xs bg-blue-900/50 text-blue-300 px-2 py-1 rounded">{result.rag?.total_latency_ms}ms total</span>
                </div>
                <p className="text-gray-300 text-sm leading-relaxed">{result.rag?.answer}</p>
                <div className="mt-3 flex gap-4 text-xs text-gray-500">
                  <span>Retrieval: {result.rag?.retrieval_latency_ms}ms</span>
                  <span>LLM: {result.rag?.llm_latency_ms}ms</span>
                  <span>Context: {result.rag?.context_count} tickets</span>
                </div>
                {result.rag?.low_similarity && (
                  <p className="mt-2 text-xs text-yellow-500">⚠ Low similarity — limited relevant context found</p>
                )}
              </div>

              <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <div className="flex justify-between items-center mb-3">
                  <h2 className="text-lg font-semibold text-white">Non-RAG Answer</h2>
                  <span className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">{result.non_rag?.llm_latency_ms}ms</span>
                </div>
                <p className="text-gray-300 text-sm leading-relaxed">{result.non_rag?.answer}</p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
