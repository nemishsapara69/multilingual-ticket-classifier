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

function timelineTimeLabel(elapsedMs, index) {
  if (index === 0 || elapsedMs <= 0) return "Now";
  return `+${elapsedMs} ms`;
}

export default function App() {
  const [authEnabled, setAuthEnabled] = useState(false);
  const [authToken, setAuthToken] = useState(() => localStorage.getItem("mtc_token") || "");
  const [authUser, setAuthUser] = useState("");
  const [authRole, setAuthRole] = useState("");
  const [loginUsername, setLoginUsername] = useState("admin");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState("");

  const [message, setMessage] = useState("");
  const [priority, setPriority] = useState("Normal");
  const [channel, setChannel] = useState("Web Portal");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState("Checking");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [accent, setAccent] = useState("aurora");
  const [selectedTicketId, setSelectedTicketId] = useState(null);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [ticketLoading, setTicketLoading] = useState(false);
  const [threadMessages, setThreadMessages] = useState([]);
  const [threadInput, setThreadInput] = useState("");
  const [threadLoading, setThreadLoading] = useState(false);
  const [threadSending, setThreadSending] = useState(false);
  const [dashboard, setDashboard] = useState({
    queue: {
      total_open: 0,
      technical_support: 0,
      billing_inquiry: 0,
      order_tracking: 0,
      manual_review: 0,
    },
    recent_tickets: [],
  });

  function authHeaders() {
    if (!authToken) {
      return { "Content-Type": "application/json" };
    }

    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${authToken}`,
    };
  }

  const band = useMemo(() => {
    if (!result) return null;
    return confidenceBand(Number(result.confidence || 0));
  }, [result]);

  const resultTimeline = useMemo(() => {
    if (result?.processing_timeline?.length) {
      return result.processing_timeline;
    }

    return [
      { label: `Received from ${channel}`, elapsed_ms: 0 },
      { label: "Text normalized and cleaned", elapsed_ms: 0 },
      { label: "Model inference complete", elapsed_ms: 0 },
      { label: "Queue recommendation generated", elapsed_ms: 0 },
    ];
  }, [result, channel]);

  const displayedTimeline = useMemo(() => {
    if (selectedTicket?.processing_timeline?.length) {
      return selectedTicket.processing_timeline;
    }
    return resultTimeline;
  }, [selectedTicket, resultTimeline]);

  async function loadTicketMessages(ticketId) {
    if (!ticketId) return;

    setThreadLoading(true);
    try {
      const messagesUrl = API_URL.replace(/\/predict$/, `/tickets/${encodeURIComponent(ticketId)}/messages`);
      const response = await fetch(messagesUrl, { headers: authHeaders() });
      if (response.status === 401 || response.status === 403) {
        setAuthToken("");
        localStorage.removeItem("mtc_token");
        throw new Error("Authentication required");
      }
      if (!response.ok) throw new Error("Failed to load conversation thread");

      const payload = await response.json();
      setThreadMessages(payload.messages || []);
    } catch {
      setThreadMessages([]);
    } finally {
      setThreadLoading(false);
    }
  }

  async function loadTicketDetails(ticketId) {
    if (!ticketId) return;

    setSelectedTicketId(ticketId);
    setTicketLoading(true);

    try {
      const ticketUrl = API_URL.replace(/\/predict$/, `/tickets/${encodeURIComponent(ticketId)}`);
      const response = await fetch(ticketUrl, { headers: authHeaders() });
      if (response.status === 401 || response.status === 403) {
        setAuthToken("");
        localStorage.removeItem("mtc_token");
        throw new Error("Authentication required");
      }
      if (!response.ok) throw new Error("Failed to load ticket details");
      const payload = await response.json();
      setSelectedTicket(payload);
      await loadTicketMessages(ticketId);
    } catch {
      setSelectedTicket(null);
      setThreadMessages([]);
    } finally {
      setTicketLoading(false);
    }
  }

  async function sendThreadMessage() {
    const ticketId = selectedTicketId;
    const message = threadInput.trim();
    if (!ticketId || !message) return;

    setThreadSending(true);
    try {
      const messagesUrl = API_URL.replace(/\/predict$/, `/tickets/${encodeURIComponent(ticketId)}/messages`);
      const response = await fetch(messagesUrl, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ message }),
      });

      if (response.status === 401 || response.status === 403) {
        setAuthToken("");
        localStorage.removeItem("mtc_token");
        throw new Error("Authentication required");
      }
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Failed to send message");
      }

      setThreadInput("");
      await loadTicketMessages(ticketId);
    } catch (err) {
      setError(err.message || "Could not send thread message.");
    } finally {
      setThreadSending(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    const healthUrl = API_URL.replace(/\/predict$/, "/health");
    const dashboardUrl = API_URL.replace(/\/predict$/, "/dashboard");
    const authConfigUrl = API_URL.replace(/\/predict$/, "/auth/config");
    const meUrl = API_URL.replace(/\/predict$/, "/auth/me");

    async function loadAuthConfig() {
      try {
        const response = await fetch(authConfigUrl, { signal: controller.signal });
        if (!response.ok) return;
        const payload = await response.json();
        setAuthEnabled(Boolean(payload.enabled));
      } catch {
        // Keep auth disabled in unavailable cases.
      }
    }

    async function loadMe() {
      if (!authToken) return;
      try {
        const response = await fetch(meUrl, {
          signal: controller.signal,
          headers: { Authorization: `Bearer ${authToken}` },
        });
        if (!response.ok) {
          setAuthToken("");
          localStorage.removeItem("mtc_token");
          return;
        }
        const payload = await response.json();
        setAuthUser(payload.username || "");
        setAuthRole(payload.role || "");
      } catch {
        // Ignore me endpoint errors.
      }
    }

    async function fetchDashboard() {
      try {
        const response = await fetch(dashboardUrl, {
          signal: controller.signal,
          headers: authHeaders(),
        });
        if (response.status === 401 || response.status === 403) {
          return;
        }
        if (!response.ok) return;
        const payload = await response.json();
        setDashboard(payload);

        if (selectedTicketId) {
          await loadTicketDetails(selectedTicketId);
        }
      } catch {
        // Keep previous snapshot if refresh fails.
      }
    }

    async function checkApi() {
      try {
        const response = await fetch(healthUrl, { signal: controller.signal });
        if (!response.ok) throw new Error("API unavailable");
        const payload = await response.json();
        setApiStatus(payload.model_loaded ? "Online" : "Model Not Loaded");
        if (payload.model_loaded) {
          await fetchDashboard();
        }
      } catch {
        setApiStatus("Offline");
      }
    }

    loadAuthConfig();
    loadMe();
    checkApi();

    const poll = setInterval(fetchDashboard, 10000);
    return () => {
      clearInterval(poll);
      controller.abort();
    };
  }, [authToken, selectedTicketId]);

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
        headers: authHeaders(),
        body: JSON.stringify({ text: message, priority, channel }),
      });

      if (response.status === 401 || response.status === 403) {
        setAuthToken("");
        localStorage.removeItem("mtc_token");
        throw new Error("Your session expired. Please login again.");
      }

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`API error ${response.status}: ${text}`);
      }

      const payload = await response.json();
      setResult(payload);

      try {
        const dashboardUrl = API_URL.replace(/\/predict$/, "/dashboard");
        const dashboardResponse = await fetch(dashboardUrl, { headers: authHeaders() });
        if (dashboardResponse.ok) {
          const dashboardPayload = await dashboardResponse.json();
          setDashboard(dashboardPayload);
          if (dashboardPayload.recent_tickets.length > 0) {
            await loadTicketDetails(dashboardPayload.recent_tickets[0].id);
          }
        }
      } catch {
        // Ignore dashboard refresh errors.
      }
    } catch (err) {
      setError(err.message || "Could not connect to API.");
    } finally {
      setLoading(false);
    }
  }

  async function onLogin(event) {
    event.preventDefault();
    setLoginError("");

    if (!loginUsername.trim() || !loginPassword.trim()) {
      setLoginError("Enter username and password.");
      return;
    }

    setLoginLoading(true);
    try {
      const loginUrl = API_URL.replace(/\/predict$/, "/auth/login");
      const response = await fetch(loginUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: loginUsername.trim(), password: loginPassword }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Login failed");
      }

      const payload = await response.json();
      setAuthToken(payload.access_token || "");
      setAuthUser(payload.username || "");
      setAuthRole(payload.role || "");
      localStorage.setItem("mtc_token", payload.access_token || "");
      setLoginPassword("");
    } catch (err) {
      setLoginError(err.message || "Could not sign in.");
    } finally {
      setLoginLoading(false);
    }
  }

  function onLogout() {
    setAuthToken("");
    setAuthUser("");
    setAuthRole("");
    localStorage.removeItem("mtc_token");
  }

  const requiresLogin = authEnabled && !authToken;

  if (requiresLogin) {
    return (
      <div className="page">
        <div className="glow glow-a" />
        <div className="glow glow-b" />
        <main className="shell auth-shell">
          <section className="panel auth-panel">
            <p className="eyebrow">Secure Access</p>
            <h1>Smart Router Console Login</h1>
            <p className="muted">Sign in with your role account to access ticket routing operations.</p>
            <form onSubmit={onLogin} className="auth-form">
              <label>
                Username
                <input
                  value={loginUsername}
                  onChange={(e) => setLoginUsername(e.target.value)}
                  placeholder="admin"
                />
              </label>
              <label>
                Password
                <input
                  type="password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  placeholder="Enter password"
                />
              </label>
              <button className="primary-btn" disabled={loginLoading}>
                {loginLoading ? "Signing in..." : "Sign In"}
              </button>
            </form>
            {loginError && <p className="error-msg">{loginError}</p>}
            <p className="muted creds-note">Default demo users: admin / agent / viewer</p>
          </section>
        </main>
      </div>
    );
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
            {authEnabled && (
              <span className="status-chip">User: {authUser || "Unknown"} ({authRole || "n/a"})</span>
            )}
            <span className="status-chip">Model: multilingual-ticket-classifier</span>
            <span className="status-chip">Routing mode: Real-time</span>
            {authEnabled && (
              <button type="button" className="ghost-btn logout-btn" onClick={onLogout}>Logout</button>
            )}
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
                <strong>{dashboard.queue.total_open}</strong>
              </li>
              <li>
                <span>Technical Support</span>
                <strong>{dashboard.queue.technical_support}</strong>
              </li>
              <li>
                <span>Billing Inquiry</span>
                <strong>{dashboard.queue.billing_inquiry}</strong>
              </li>
              <li>
                <span>Order Tracking</span>
                <strong>{dashboard.queue.order_tracking}</strong>
              </li>
              <li>
                <span>Manual Review</span>
                <strong>{dashboard.queue.manual_review}</strong>
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

                    <article className="metric-card">
                      <p className="metric-label">Processing Time</p>
                      <p className="metric-value small">
                        {result.total_processing_ms ? `${result.total_processing_ms} ms` : "Pending"}
                      </p>
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
                    {dashboard.recent_tickets.map((row) => (
                      <tr
                        key={row.id}
                        className={`clickable-row ${selectedTicketId === row.id ? "is-active" : ""}`}
                        onClick={() => loadTicketDetails(row.id)}
                      >
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
                    {dashboard.recent_tickets.length === 0 && (
                      <tr>
                        <td colSpan="6" className="empty-row">No live tickets yet. Classify one message to populate this feed.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </div>

          <aside className="panel ops-panel">
            <h2>Agent Ops</h2>

            <h3>Routing Timeline</h3>
            {selectedTicket && (
              <p className="timeline-meta">
                Selected: {selectedTicket.id} | Total: {selectedTicket.total_processing_ms} ms
              </p>
            )}
            {ticketLoading && <p className="muted">Loading selected ticket timeline...</p>}
            <ol className="timeline">
              {displayedTimeline.map((step, index) => (
                <li key={`${step.label}-${index}`}>
                  <span>{step.label}</span>
                  <small>{timelineTimeLabel(Number(step.elapsed_ms || 0), index)}</small>
                </li>
              ))}
            </ol>

            <h3>Conversation Thread</h3>
            {!selectedTicketId && <p className="muted">Select a ticket from feed to open conversation.</p>}

            {selectedTicketId && (
              <>
                <div className="thread-box">
                  {threadLoading && <p className="muted">Loading conversation...</p>}
                  {!threadLoading && threadMessages.length === 0 && (
                    <p className="muted">No messages yet for this ticket.</p>
                  )}
                  {threadMessages.map((msg) => (
                    <article key={msg.id} className={`thread-message ${msg.role}`}>
                      <p className="thread-head">{msg.sender} ({msg.role})</p>
                      <p className="thread-body">{msg.message}</p>
                    </article>
                  ))}
                </div>

                {authRole !== "viewer" && (
                  <div className="thread-compose">
                    <textarea
                      className="notes-box"
                      value={threadInput}
                      onChange={(e) => setThreadInput(e.target.value)}
                      placeholder="Reply to this ticket conversation..."
                      rows={3}
                    />
                    <button className="ghost-btn" type="button" onClick={sendThreadMessage} disabled={threadSending}>
                      {threadSending ? "Sending..." : "Send Reply"}
                    </button>
                  </div>
                )}
              </>
            )}
          </aside>
        </section>
      </main>
    </div>
  );
}
