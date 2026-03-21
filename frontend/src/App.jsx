import { useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/predict";

const EXAMPLES = {
  Spanish: "Mi luz inteligente no funciona",
  English: "I was charged twice this month",
  German: "Wo ist mein Paket",
};

function confidenceBand(value) {
  if (value >= 0.75) return { label: "High", tone: "ok" };
  if (value >= 0.5) return { label: "Medium", tone: "warn" };
  return { label: "Low", tone: "danger" };
}

export default function App() {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const band = useMemo(() => {
    if (!result) return null;
    return confidenceBand(Number(result.confidence || 0));
  }, [result]);

  async function onClassify() {
    setError("");
    setResult(null);

    if (!message.trim()) {
      setError("Please enter a message first.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`API error ${response.status}: ${text}`);
      }

      const payload = await response.json();
      setResult(payload);
    } catch (err) {
      setError(err.message || "Could not connect to API.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="glow glow-a" />
      <div className="glow glow-b" />

      <main className="shell">
        <header className="hero">
          <p className="eyebrow">Global Support AI</p>
          <h1>Smart Router React Console</h1>
          <p>
            Route multilingual tickets to Technical Support, Billing Inquiry, or
            Order Tracking in real time.
          </p>
        </header>

        <section className="panel input-panel">
          <div className="panel-top">
            <h2>Ticket Message</h2>
            <span className="api-pill">API: {API_URL}</span>
          </div>

          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type a customer message..."
            rows={7}
          />

          <div className="quick-row">
            {Object.entries(EXAMPLES).map(([label, text]) => (
              <button
                key={label}
                type="button"
                className="ghost-btn"
                onClick={() => setMessage(text)}
              >
                {label} Example
              </button>
            ))}
          </div>

          <button className="primary-btn" onClick={onClassify} disabled={loading}>
            {loading ? "Classifying..." : "Classify Ticket"}
          </button>

          {error && <p className="error-msg">{error}</p>}
        </section>

        <section className="panel result-panel">
          <h2>Prediction</h2>

          {!result && <p className="muted">Submit a message to see prediction output.</p>}

          {result && (
            <>
              <div className="result-grid">
                <article className="metric-card">
                  <p className="metric-label">Predicted Category</p>
                  <p className="metric-value">{result.category}</p>
                </article>

                <article className="metric-card">
                  <p className="metric-label">Confidence</p>
                  <p className="metric-value">{(result.confidence * 100).toFixed(2)}%</p>
                </article>

                <article className={`metric-card band ${band.tone}`}>
                  <p className="metric-label">Confidence Band</p>
                  <p className="metric-value">{band.label}</p>
                </article>
              </div>

              <article className="clean-box">
                <p className="metric-label">Cleaned Text</p>
                <p>{result.cleaned_text}</p>
              </article>

              {band.label === "Low" && (
                <p className="review-note">Low confidence: send this ticket for manual review.</p>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
