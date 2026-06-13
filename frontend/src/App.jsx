import { useEffect, useRef, useState } from "react";

const LOADING_STEPS = [
  "Scanning package metadata",
  "Checking dependency reputation",
  "Analyzing security signals",
  "Generating risk insights",
];

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const FEATURE_MEANINGS = {
  days_since_created: "Newer packages have less trust history.",
  days_since_last_update: "Long inactivity can indicate abandonment.",
  num_versions: "Version history helps show package maturity.",
  release_velocity: "Burst releases can be suspicious.",
  num_maintainers: "More maintainers usually reduce single-point control risk.",
  has_any_install_hook: "Install lifecycle scripts are a common malware execution path.",
  has_postinstall: "Postinstall scripts are a common malware execution path.",
  has_github_repo: "Missing repository lowers verification confidence.",
  stargazers_count: "Stars are a rough adoption and trust signal.",
  forks_count: "Forks indicate ecosystem interest and review depth.",
  contributor_count: "Higher contributor count implies broader ownership.",
  weekly_downloads: "Very low download counts can indicate a throwaway package.",
  typosquat_min_distance: "Small edit distance to a popular package name is suspicious.",
  script_suspicion_score: "Dangerous patterns found in install scripts.",
  maintainer_min_account_age_days: "Newly created maintainer accounts are higher risk.",
};

function normalizePackageName(value) {
  return value.trim();
}

function validatePackageName(value) {
  const trimmed = normalizePackageName(value);

  if (!trimmed) return "Enter a package name.";
  if (/\s/.test(trimmed)) return "Package names cannot contain spaces.";
  return "";
}

function getRiskTone(riskLevel = "") {
  const normalized = String(riskLevel).toUpperCase();
  if (normalized.includes("HEALTHY") || normalized.includes("LOW")) return "safe";
  if (normalized.includes("MODERATE") || normalized.includes("MEDIUM") || normalized.includes("CAUTION")) return "warning";
  return "danger";
}

function getRiskLabel(score, riskLevel) {
  if (riskLevel) return String(riskLevel).replaceAll("_", " ");
  if (score < 0.33) return "LOW";
  if (score < 0.66) return "MEDIUM";
  return "HIGH";
}

function scoreToRiskColor(score) {
  if (score < 0.33) return "var(--success)";
  if (score < 0.66) return "var(--warning)";
  return "var(--danger)";
}

function formatFeatureValue(value) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    if (Math.abs(value) < 1) return value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
    return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  return String(value);
}

function formatReport(result) {
  const factors = (result.explanations || []).slice(0, 4);
  return [
    `Package: ${result.package}`,
    `Risk score: ${(result.score * 100).toFixed(1)}%`,
    `Risk level: ${getRiskLabel(result.score, result.risk_level)}`,
    "",
    "Top factors:",
    ...factors.map((factor) => `- ${factor.feature}: ${factor.shap_value}`),
  ].join("\n");
}

function getAlternativeName(result) {
  if (!result) return "";
  if (typeof result.suggested_alternative === "string") return result.suggested_alternative;
  if (result.suggested_alternative?.package) return result.suggested_alternative.package;
  if (result.suggestion?.name) return result.suggestion.name;
  return "";
}

function buildShareUrl(packageName) {
  return `${window.location.origin}/?pkg=${encodeURIComponent(packageName)}`;
}

async function readResponseBody(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  return response.text();
}

function Gauge({ score, riskLevel }) {
  const size = 200;
  const stroke = 9;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const progress = Math.max(0, Math.min(1, score));
  const dashOffset = circumference - progress * circumference;
  const color = scoreToRiskColor(score);
  const percent = Math.round(progress * 100);

  return (
    <div className="gauge-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="gauge-svg" aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} className="gauge-track" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          className="gauge-progress"
          strokeWidth={stroke}
          fill="none"
          stroke={color}
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
        />
      </svg>
      <div className="gauge-label">
        <span className="gauge-value">
          {percent === 0 ? "✓" : `${percent}%`}
        </span>
        <span className="gauge-subtitle">{getRiskLabel(score, riskLevel)}</span>
      </div>
    </div>
  );
}

function FactorBars({ items }) {
  if (!items?.length) {
    return <p className="muted">No explanation factors available for this package.</p>;
  }

  const maxAbs = Math.max(...items.map((item) => Math.abs(item.shap_value || 0)), 1e-9);

  return (
    <div className="bars">
      {items.map((item) => {
        const value = item.shap_value || 0;
        const widthPct = Math.max(10, Math.round((Math.abs(value) / maxAbs) * 100));
        const positive = value >= 0;
        const meaning = FEATURE_MEANINGS[item.feature] || "Derived from package and GitHub metadata.";

        return (
          <div className="bar-row" key={`${item.feature}-${value}`}>
            <div className="bar-meta">
              <span className="feature-pill" data-meaning={meaning}>
                <span className="mono feature-name">{item.feature}</span>
              </span>
              <span className={`mono feature-value ${positive ? "positive" : "negative"}`}>
                {positive ? "↑" : "↓"} {value.toFixed(3)}
              </span>
            </div>
            <div className="bar-track">
              <div className={`bar-fill ${positive ? "bar-pos" : "bar-neg"}`} style={{ width: `${widthPct}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

const SCOPE_LETTERS = [
  { letter: "S", word: "Security" },
  { letter: "C", word: "Check for" },
  { letter: "O", word: "Open-source" },
  { letter: "P", word: "Package" },
  { letter: "E", word: "Ecosystems" },
];

function SplashScreen({ onDone }) {
  const [fading, setFading] = useState(false);

  useEffect(() => {
    const fadeTimer = setTimeout(() => setFading(true), 2400);
    const doneTimer = setTimeout(onDone, 3000);
    return () => { clearTimeout(fadeTimer); clearTimeout(doneTimer); };
  }, [onDone]);

  return (
    <div className={`splash ${fading ? "splash-fade" : ""}`}>
      <div className="splash-inner">
        <div className="splash-acronym">
          {SCOPE_LETTERS.map(({ letter, word }, i) => (
            <div key={letter} className="splash-row" style={{ animationDelay: `${i * 110}ms` }}>
              <span className="splash-letter">{letter}</span>
              <span className="splash-word">{word}</span>
            </div>
          ))}
        </div>
        <p className="splash-sub">NPM Dependency Security Scanner</p>
      </div>
    </div>
  );
}

function RecentDialog({ open, onClose, onClear, recentAnalyses, onSelectRecent }) {
  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <section className="glass card modal-panel" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <h2>Recent Analyses</h2>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {recentAnalyses.length > 0 && (
              <button type="button" className="icon-button" onClick={onClear}>Clear</button>
            )}
            <button type="button" className="icon-button" onClick={onClose}>Close</button>
          </div>
        </div>
        <p className="muted">Saved in your browser. Click any package to restore results.</p>
        {recentAnalyses.length > 0 ? (
          <div className="recent-grid">
            {recentAnalyses.map((entry) => {
              const tone = getRiskTone(entry.risk_level);
              return (
                <button
                  type="button"
                  key={`${entry.package}-${entry.viewedAt}`}
                  className={`recent-card recent-${tone}`}
                  onClick={() => {
                    onSelectRecent(entry);
                    onClose();
                  }}
                >
                  <span className="recent-name">{entry.package}</span>
                  <span className="recent-score mono">{Math.round(entry.score * 100)}%</span>
                  <span className="recent-dot" aria-hidden="true" />
                </button>
              );
            })}
          </div>
        ) : (
          <p className="muted">No saved analyses yet.</p>
        )}
      </section>
    </div>
  );
}

function getRiskBadgeClass(riskLevel = "") {
  const r = riskLevel.toUpperCase();
  if (r === "HEALTHY") return "badge-healthy";
  if (r === "MEDIUM")  return "badge-medium";
  if (r === "HIGH")    return "badge-high";
  if (r === "CRITICAL") return "badge-critical";
  return "badge-unknown";
}

function DashboardView({ onOpenPackage }) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [depCount, setDepCount] = useState(0);
  const [rows, setRows] = useState(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setInput(ev.target.result);
    reader.readAsText(file);
    e.target.value = "";
  }

  async function handleScan() {
    setError("");
    setRows(null);

    let parsed;
    try {
      parsed = JSON.parse(input);
    } catch {
      setError("Invalid JSON — paste or upload a package.json file.");
      return;
    }

    const pkgCount = Object.keys({
      ...(parsed.dependencies || {}),
      ...(parsed.devDependencies || {}),
    }).length;
    setDepCount(pkgCount);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/dashboard/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ package_json: parsed }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      setRows(await res.json());
    } catch (err) {
      setError(err.message || "Scan failed.");
    } finally {
      setLoading(false);
    }
  }

  const summary = rows
    ? rows.reduce(
        (acc, row) => {
          const r = (row.risk_level || "UNKNOWN").toUpperCase();
          acc.counts[r] = (acc.counts[r] || 0) + 1;
          acc.totalCves += row.cves.length;
          return acc;
        },
        { counts: {}, totalCves: 0 }
      )
    : null;

  return (
    <section className="dashboard-view">
      <div className="glass card reveal">
        <h2>Dependency Vulnerability Scan</h2>
        <p className="muted">Upload or paste your <span className="mono">package.json</span> to scan all dependencies for known CVEs and ML risk scores.</p>
        <div className="dash-upload-row">
          <input ref={fileInputRef} type="file" accept=".json,application/json" className="dash-file-input" onChange={handleFileUpload} />
          <button type="button" className="secondary-button" onClick={() => fileInputRef.current?.click()}>
            Upload package.json
          </button>
          <span className="muted">or paste below</span>
        </div>
        <textarea
          className="dash-textarea"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={'{\n  "dependencies": {\n    "lodash": "^4.17.20"\n  }\n}'}
          rows={10}
          spellCheck={false}
        />
        <button className="search-button" onClick={handleScan} disabled={loading || !input.trim()}>
          {loading ? `Scanning ${depCount} package${depCount !== 1 ? "s" : ""}…` : "Scan Project"}
        </button>
      </div>

      {error && (
        <div className="glass card error-card reveal">
          <p>{error}</p>
        </div>
      )}

      {rows && summary && (
        <>
          <div className="glass card dash-summary reveal">
            <div className="dash-summary-stats">
              {summary.counts.CRITICAL > 0 && <span className="dash-stat dash-stat-critical">{summary.counts.CRITICAL} critical</span>}
              {summary.counts.HIGH > 0 && <span className="dash-stat dash-stat-high">{summary.counts.HIGH} high</span>}
              {summary.counts.MEDIUM > 0 && <span className="dash-stat dash-stat-medium">{summary.counts.MEDIUM} medium</span>}
              {summary.counts.HEALTHY > 0 && <span className="dash-stat dash-stat-healthy">{summary.counts.HEALTHY} healthy</span>}
              <div className="dash-stat-divider" />
              <span className="dash-stat dash-stat-cve">{summary.totalCves} CVE{summary.totalCves !== 1 ? "s" : ""} found</span>
              <span className="muted" style={{ fontSize: "0.82rem" }}>across {rows.length} package{rows.length !== 1 ? "s" : ""}</span>
            </div>
          </div>

          <div className="glass card reveal">
            <div className="dash-table-wrap">
              <table className="dash-table">
                <thead>
                  <tr>
                    <th>Package</th>
                    <th>Version</th>
                    <th>Risk</th>
                    <th>Score</th>
                    <th>CVEs</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.package}>
                      <td className="mono">{row.package}</td>
                      <td className="mono muted">{row.version || "—"}</td>
                      <td>
                        <span className={`risk-badge ${getRiskBadgeClass(row.risk_level)}`}>
                          {row.risk_level || "UNKNOWN"}
                        </span>
                      </td>
                      <td className="mono">{row.score != null ? `${(row.score * 100).toFixed(1)}%` : "—"}</td>
                      <td>
                        {row.cves.length === 0 ? (
                          <span className="muted">none</span>
                        ) : (
                          <details className="cve-details">
                            <summary className="cve-count">{row.cves.length} CVE{row.cves.length !== 1 ? "s" : ""}</summary>
                            <ul className="cve-list">
                              {row.cves.map((cve) => (
                                <li key={cve.id}>
                                  <span className="mono cve-id">{cve.id}</span>
                                  {cve.summary && <span className="cve-summary"> — {cve.summary}</span>}
                                </li>
                              ))}
                            </ul>
                          </details>
                        )}
                      </td>
                      <td>
                        <button type="button" className="dash-inspect-btn" onClick={() => onOpenPackage(row.package)}>
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [activeTab, setActiveTab] = useState("scan");
  const [packageName, setPackageName] = useState("react-domm");
  const [loading, setLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [error, setError] = useState("");
  const [notFound, setNotFound] = useState(null); // { name, suggestion? }
  const [result, setResult] = useState(null);
  const [recentAnalyses, setRecentAnalyses] = useState(() => {
    try { return JSON.parse(localStorage.getItem("scope_recent") || "[]"); }
    catch { return []; }
  });
  const [recentOpen, setRecentOpen] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const suggestTimer = useRef(null);

  useEffect(() => {
    if (!loading) return undefined;
    const id = setInterval(() => setStepIndex((prev) => (prev + 1) % LOADING_STEPS.length), 1050);
    return () => clearInterval(id);
  }, [loading]);

  useEffect(() => {
    try { localStorage.setItem("scope_recent", JSON.stringify(recentAnalyses)); }
    catch { /* storage full or unavailable */ }
  }, [recentAnalyses]);

  useEffect(() => {
    const pkg = new URLSearchParams(window.location.search).get("pkg");
    if (pkg) {
      setPackageName(pkg);
      analyzePackage(pkg);
    }
  }, []);

  async function analyzePackage(rawPackageName) {
    const trimmed = normalizePackageName(rawPackageName);
    const validationError = validatePackageName(trimmed);

    if (validationError) {
      setError(validationError);
      setLoading(false);
      return null;
    }

    setPackageName(trimmed);
    setError("");
    setNotFound(null);
    setResult(null);
    setLoading(true);
    setStepIndex(0);

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ package_name: trimmed }),
      });

      const body = await readResponseBody(response);

      if (response.status === 404) {
        // Package doesn't exist on npm — show dedicated not-found state
        setNotFound({
          name: trimmed,
          suggestion: body?.suggestion || null,
        });
        return null;
      }

      if (!response.ok) {
        const message = typeof body === "string" ? body : body?.detail || "Analysis failed.";
        throw new Error(`HTTP ${response.status}: ${message}`);
      }

      setResult(body);
      setRecentAnalyses((current) => {
        const deduped = current.filter((entry) => entry.package !== body.package);
        return [{ ...body, viewedAt: Date.now() }, ...deduped].slice(0, 5);
      });
      return body;
    } catch (err) {
      setResult(null);
      if (err.name === "TypeError" || /fetch/i.test(err.message || "")) {
        setError("Cannot reach server. Is the API running?");
      } else {
        setError(err.message || "Something went wrong.");
      }
      return null;
    } finally {
      setLoading(false);
    }
  }

  function onPackageInput(e) {
    const val = e.target.value;
    setPackageName(val);
    setShowSuggestions(false);
    clearTimeout(suggestTimer.current);
    if (val.trim().length < 2) { setSuggestions([]); return; }
    suggestTimer.current = setTimeout(async () => {
      try {
        const res = await fetch(
          `https://registry.npmjs.org/-/v1/search?text=${encodeURIComponent(val.trim())}&size=6`
        );
        const data = await res.json();
        setSuggestions((data.objects || []).map((o) => o.package.name));
        setShowSuggestions(true);
      } catch { setSuggestions([]); }
    }, 280);
  }

  function onSuggestionPick(name) {
    setPackageName(name);
    setSuggestions([]);
    setShowSuggestions(false);
    analyzePackage(name);
  }

  async function onSubmit(event) {
    event.preventDefault();
    await analyzePackage(packageName);
  }

  async function onPackageKeyDown(event) {
    if (event.key !== "Enter") return;
    event.preventDefault();
    await analyzePackage(packageName);
  }

  async function handleCopyReport(data) {
    await navigator.clipboard.writeText(formatReport(data));
  }

  async function handleCopyShareUrl(packageToShare) {
    await navigator.clipboard.writeText(buildShareUrl(packageToShare));
  }

  async function handleAnalyzeAlternative(alternativeName) {
    await analyzePackage(alternativeName);
  }

  function handleSelectRecent(entry) {
    setResult(entry);
    setPackageName(entry.package);
    setError("");
  }

  const summarySignals = result
    ? [
        {
          label: "Package",
          value: result.package,
          meaning: "Exact package name sent to the model for scoring.",
        },
        {
          label: "Risk level",
          value: getRiskLabel(result.score, result.risk_level),
          meaning: "Label bucket derived from the model score.",
        },
        {
          label: "Risk score",
          value: result.score.toFixed(3),
          meaning: "Probability-like risk output from 0 to 1.",
        },
        {
          label: "Install hook",
          value: result.features?.has_any_install_hook ? "yes" : "no",
          meaning: FEATURE_MEANINGS.has_any_install_hook,
        },
        {
          label: "Maintainers",
          value: formatFeatureValue(result.features?.num_maintainers),
          meaning: FEATURE_MEANINGS.num_maintainers,
        },
        {
          label: "GitHub repo",
          value: result.features?.has_github_repo ? "present" : "missing",
          meaning: FEATURE_MEANINGS.has_github_repo,
        },
      ]
    : [];

  const topFactors = result?.explanations ? result.explanations.slice(0, 4) : [];
  const alternativeName = getAlternativeName(result);
  const riskTone = getRiskTone(result?.risk_level || "");

  return (
    <div className="page">
      {showSplash && <SplashScreen onDone={() => setShowSplash(false)} />}
      <div className="bg-noise" aria-hidden="true" />
      <div className="bg-blob bg-blob-a" aria-hidden="true" />
      <div className="bg-blob bg-blob-b" aria-hidden="true" />

      <nav className="navbar">
        <span className="nav-brand">SCOPE</span>
        <div className="nav-divider" />
        <div className="nav-links">
          <button type="button" className={`nav-link ${activeTab === "scan" ? "nav-link-active" : ""}`} onClick={() => setActiveTab("scan")}>
            Package Scan
          </button>
          <button type="button" className={`nav-link ${activeTab === "dashboard" ? "nav-link-active" : ""}`} onClick={() => setActiveTab("dashboard")}>
            Dashboard
          </button>
        </div>
        <div className="nav-divider" />
        <button type="button" className="nav-recent" onClick={() => setRecentOpen(true)}>
          Recent {recentAnalyses.length > 0 && <span className="nav-recent-count">{recentAnalyses.length}</span>}
        </button>
      </nav>

      <main className="shell">
        <header className="hero reveal">
          <p className="kicker">SCOPE Intelligence</p>
          <h1>{activeTab === "dashboard" ? "Dependency Dashboard" : "Package Trust Analysis"}</h1>
          <p className="subtitle">Minimal, explainable security insight for NPM dependencies.</p>
        </header>

        {activeTab === "dashboard" && <DashboardView />}

        {activeTab === "scan" && <>
        <section className="glass card search-card reveal reveal-delay-1">
          <form onSubmit={onSubmit} className="search-form">
            <label htmlFor="package-input" className="label">
              Package
            </label>
            <div className="search-row" style={{ position: "relative" }}>
              <input
                id="package-input"
                value={packageName}
                onChange={onPackageInput}
                onKeyDown={onPackageKeyDown}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
                onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
                placeholder="Enter package name (e.g. react-domm)"
                className="search-input"
                autoComplete="off"
                required
              />
              {showSuggestions && suggestions.length > 0 && (
                <ul className="suggestions-list">
                  {suggestions.map((s) => (
                    <li key={s} className="suggestions-item" onMouseDown={() => onSuggestionPick(s)}>
                      {s}
                    </li>
                  ))}
                </ul>
              )}
              <button type="submit" className="search-button" disabled={loading}>
                {loading ? "Analyzing" : "Analyze"}
              </button>
            </div>
          </form>
        </section>

        {loading && (
          <section className="glass card loading-card reveal reveal-delay-2">
            <p className="loading-text">
              {LOADING_STEPS[stepIndex]}
              <span className="cursor">|</span>
            </p>
          </section>
        )}

        {error && (
          <section className="glass card error-card reveal reveal-delay-2">
            <p>{error}</p>
          </section>
        )}

        {notFound && !loading && (
          <section className="glass card error-card reveal reveal-delay-2">
            <p>
              <strong>&ldquo;{notFound.name}&rdquo;</strong> was not found on npm. Check the
              spelling, or this package may have been unpublished.
            </p>
            {notFound.suggestion && (
              <p>
                Did you mean{" "}
                <button
                  type="button"
                  className="inline-link"
                  onClick={() => analyzePackage(notFound.suggestion.name)}
                >
                  {notFound.suggestion.name}
                </button>
                ?
              </p>
            )}
          </section>
        )}

        {result && !loading && (
          <section className="results reveal reveal-delay-2">
            <article className={`glass card score-card risk-${riskTone}`}>
              <div className="section-heading">
                <div>
                  <h2>Risk Insights</h2>
                  <p className="muted">Main score first. Hover any signal for a short interpretation.</p>
                </div>
                <div className="result-actions">
                  <button type="button" className="secondary-button" onClick={() => handleCopyReport(result)}>
                    Copy Report
                  </button>
                  <button type="button" className="secondary-button" onClick={() => handleCopyShareUrl(result.package)}>
                    Share URL
                  </button>
                </div>
              </div>

              <div className="score-layout">
                <Gauge score={result.score} riskLevel={result.risk_level} />
                <div className="signal-grid">
                  {summarySignals.map((signal) => (
                    <article className="signal-card" key={signal.label} data-meaning={signal.meaning}>
                      <p className="signal-label">{signal.label}</p>
                      <p className="signal-value mono">{signal.value}</p>
                    </article>
                  ))}
                </div>
              </div>

              {alternativeName && (
                <div className="typosquat-note">
                  <span>May be typosquatting. Did you mean:</span>
                  <button type="button" className="inline-link" onClick={() => handleAnalyzeAlternative(alternativeName)}>
                    {alternativeName}?
                  </button>
                </div>
              )}
            </article>

            {result.llm_verdict && (
              <article className="glass card verdict-card">
                <h2>Security Analysis</h2>
                <p className="verdict-text">{result.llm_verdict}</p>
                <p className="muted verdict-note">Generated for HIGH / CRITICAL packages only</p>
              </article>
            )}

            <article className="glass card chart-card">
              <h2>Top Risk Factors</h2>
              <p className="muted">These are the strongest model factors for this prediction.</p>
              <FactorBars items={topFactors} />
            </article>
          </section>
        )}
        </>}
      </main>

      <RecentDialog
        open={recentOpen}
        onClose={() => setRecentOpen(false)}
        onClear={() => setRecentAnalyses([])}
        recentAnalyses={recentAnalyses}
        onSelectRecent={handleSelectRecent}
      />
    </div>
  );
}
