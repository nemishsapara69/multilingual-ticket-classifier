import { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/predict";

const EXAMPLES = {
  Spanish: "Mi luz inteligente no funciona",
  English: "I was charged twice this month",
  German: "Wo ist mein Paket",
};

const ACCENTS = {
  aurora: { label: "Aurora", cyan: "#26b3e0", amber: "#ff9c4d" },
  emerald: { label: "Emerald", cyan: "#2dc78f", amber: "#a7e049" },
  ruby: { label: "Ruby", cyan: "#df5a8a", amber: "#ff8f6a" },
};

const TICKET_HISTORY = [
  { id: "GT-1209", channel: "Web", language: "EN", category: "Order Tracking", status: "Queued", age: "02m" },
  { id: "GT-1208", channel: "Email", language: "ES", category: "Technical Support", status: "Assigned", age: "07m" },
  { id: "GT-1207", channel: "Phone", language: "DE", category: "Billing Inquiry", status: "Queued", age: "11m" },
  { id: "GT-1206", channel: "Web", language: "EN", category: "Technical Support", status: "Escalated", age: "18m" },
];

function confidenceBand(value) {
  if (value >= 0.75) return { label: "High", tone: "ok" };
  if (value >= 0.5) return { label: "Medium", tone: "warn" };
  return { label: "Low", tone: "danger" };
}

function slaTarget(category, confidenceLabel) {
  if (confidenceLabel === "Low") return "Manual triage within 15 min";
  if (category === "Technical Support") return "Engineer queue: 30 min";
  if (category === "Billing Inquiry") return "Billing queue: 2 hrs";
  return "Logistics queue: 1 hr";
}

export default function App() {
  const [message, setMessage] = useState("");
  const [priority, setPriority] = useState("Normal");
  const [channel, setChannel] = useState("Web Portal");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState("Checking");
  const [agentNote, setAgentNote] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [accent, setAccent] = useState("aurora");

  const band = useMemo(() => {
    if (!result) return null;
    return confidenceBand(Number(result.confidence || 0));
  }, [result]);

  useEffect(() => {
    const controller = new AbortController();
    const healthUrl = API_URL.replace(/\/predict$/, "/health");

    async function checkApi() {
      try {
        const response = await fetch(healthUrl, { signal: controller.signal });
        if (!response.ok) throw new Error("API unavailable");
        const payload = await response.json();
        setApiStatus(payload.model_loaded ? "Online" : "Model Not Loaded");
      } catch {
        setApiStatus("Offline");
      }
    }

    checkApi();
    return () => controller.abort();
  }, []);

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
    <div
      className="page"
      style={{
        "--accent-cyan": ACCENTS[accent].cyan,
        "--accent-amber": ACCENTS[accent].amber,
      }}
    >
      <div className="glow glow-a" />
      <div className="glow glow-b" />

      <main className="shell">
        <header className="hero">
          <p className="eyebrow">Global Support Operations</p>
          <h1>Smart Router React Console</h1>
          <p>
            Route multilingual tickets to Technical Support, Billing Inquiry, or
            Order Tracking in real time.
          </p>
          <div className="hero-strip">
            <span className={`status-dot ${apiStatus === "Online" ? "up" : "down"}`}>
              API: {apiStatus}
            </span>
            <span className="status-chip">Model: multilingual-ticket-classifier</span>
            <span className="status-chip">Routing mode: Real-time</span>
            <label className="accent-picker">
              Theme Accent
              <select value={accent} onChange={(e) => setAccent(e.target.value)}>
                {Object.entries(ACCENTS).map(([key, value]) => (
                  <option key={key} value={key}>{value.label}</option>
                ))}
              </select>
            </label>
          </div>
        </header>

        <section className={`workspace-grid ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
          <aside className="panel sidebar-panel">
            <button
              type="button"
              className="collapse-btn"
              onClick={() => setSidebarCollapsed((v) => !v)}
            >
              {sidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
            </button>

            <h2>Queue Monitor</h2>
            <ul className="queue-list">
              <li>
                <span>All Open Tickets</span>
                <strong>128</strong>
              </li>
              <li>
                <span>Technical Support</span>
                <strong>54</strong>
              </li>
              <li>
                <span>Billing Inquiry</span>
                <strong>37</strong>
              </li>
              <li>
                <span>Order Tracking</span>
                <strong>31</strong>
              </li>
              <li>
                <span>Manual Review</span>
                <strong>6</strong>
              </li>
            </ul>

            <h3>Live Routing Rules</h3>
            <div className="rule-stack">
              <span className="rule-chip">Model inference</span>
              <span className="rule-chip">Confidence thresholds</span>
              <span className="rule-chip">Keyword fallback</span>
            </div>
          </aside>

          <div className="main-stack">
            <section className="panel input-panel">
              <div className="panel-top">
                <h2>Ticket Message</h2>
                <span className="api-pill">API: {API_URL}</span>
              </div>

              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type a customer message..."
                rows={6}
              />

              <div className="meta-row">
                <label>
                  Priority
                  <select value={priority} onChange={(e) => setPriority(e.target.value)}>
                    <option>Low</option>
                    <option>Normal</option>
                    <option>High</option>
                    <option>Urgent</option>
                  </select>
                </label>
                <label>
                  Channel
                  <select value={channel} onChange={(e) => setChannel(e.target.value)}>
                    <option>Web Portal</option>
                    <option>Email</option>
                    <option>Mobile App</option>
                    <option>Phone Agent</option>
                  </select>
                </label>
              </div>

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

                    <article className="metric-card">
                      <p className="metric-label">Recommended Queue</p>
                      <p className="metric-value">{result.category}</p>
                    </article>

                    <article className="metric-card">
                      <p className="metric-label">SLA Target</p>
                      <p className="metric-value small">{slaTarget(result.category, band.label)}</p>
                    </article>

                    <article className="metric-card">
                      <p className="metric-label">Ticket Meta</p>
                      <p className="metric-value small">{priority} | {channel}</p>
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

            <section className="panel history-panel">
              <h2>Recent Ticket Feed</h2>
              <div className="table-wrap">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Channel</th>
                      <th>Lang</th>
                      <th>Category</th>
                      <th>Status</th>
                      <th>Age</th>
                    </tr>
                  </thead>
                  <tbody>
                    {TICKET_HISTORY.map((row) => (
                      <tr key={row.id}>
                        <td>{row.id}</td>
                        <td>{row.channel}</td>
                        <td>{row.language}</td>
                        <td>{row.category}</td>
                        <td>
                          <span className={`status-tag ${row.status.toLowerCase()}`}>{row.status}</span>
                        </td>
                        <td>{row.age}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>

          <aside className="panel ops-panel">
            <h2>Agent Ops</h2>

            <h3>Routing Timeline</h3>
            <ol className="timeline">
              <li>
                <span>Received from {channel}</span>
                <small>Now</small>
              </li>
              <li>
                <span>Text normalized and cleaned</span>
                <small>+100 ms</small>
              </li>
              <li>
                <span>Model + fallback scoring complete</span>
                <small>+220 ms</small>
              </li>
              <li>
                <span>Queue recommendation generated</span>
                <small>+280 ms</small>
              </li>
            </ol>

            <h3>Agent Notes</h3>
            <textarea
              className="notes-box"
              value={agentNote}
              onChange={(e) => setAgentNote(e.target.value)}
              placeholder="Add optional handling notes for downstream support team..."
              rows={5}
            />
          </aside>
        </section>
      </main>
    </div>
  );
}
