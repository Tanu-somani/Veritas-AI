"use client";

import { useState } from "react";

export default function Home() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    label: string;
    confidence: number;
    explanation: string | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction from the server");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <div className="header">
        <h1>Veritas AI</h1>
        <p>Enterprise-grade Fake News Detection powered by advanced Natural Language Processing</p>
      </div>

      <div className="glass-panel">
        <textarea
          placeholder="Paste the news article text here to analyze its authenticity..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
        />
        
        <button 
          className="btn-primary" 
          onClick={handlePredict}
          disabled={!text.trim() || loading}
        >
          {loading ? <span className="loader"></span> : "Analyze Authenticity"}
        </button>

        {error && (
          <div style={{ color: 'var(--error-color)', marginTop: '1rem', textAlign: 'center' }}>
            {error}
          </div>
        )}

        {result && (
          <div className="result-card">
            <div className="result-header">
              <span className="result-title">Analysis Result</span>
              <span className={`result-label ${result.label === 'REAL' ? 'label-real' : 'label-fake'}`}>
                {result.label}
              </span>
            </div>
            
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                <span>Confidence Score</span>
                <span>{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="confidence-bar-container">
                <div 
                  className="confidence-bar" 
                  style={{ 
                    width: `${result.confidence * 100}%`,
                    background: result.label === 'REAL' ? 'var(--success-color)' : 'var(--error-color)'
                  }}
                />
              </div>
            </div>

            {result.explanation && (
              <div className="explanation-section">
                <h3>AI Explanation</h3>
                <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                  {result.explanation}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
