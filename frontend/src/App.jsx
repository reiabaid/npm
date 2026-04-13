import { useEffect, useState } from "react";

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
  has_postinstall: "Postinstall scripts are a common malware execution path.",
  has_github_repo: "Missing repository lowers verification confidence.",
  stargazers_count: "Stars are a rough adoption and trust signal.",
  forks_count: "Forks indicate ecosystem interest and review depth.",
  contributor_count: "Higher contributor count implies broader ownership.",
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
  return `https://scope.dev/check/${encodeURIComponent(packageName)}`;
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
        <span className="gauge-value">{percent}</span>
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

function RecentDialog({ open, onClose, recentAnalyses, onSelectRecent }) {
  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <section className="glass card modal-panel" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <h2>Recent Analyses</h2>
          <button type="button" className="icon-button" onClick={onClose}>
            Close
          </button>
        </div>
        <p className="muted">Click any package to reopen saved results instantly.</p>
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

export default function App() {
  const [packageName, setPackageName] = useState("react-domm");
  const [loading, setLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [recentOpen, setRecentOpen] = useState(false);

  useEffect(() => {
    if (!loading) return undefined;
    const id = setInterval(() => setStepIndex((prev) => (prev + 1) % LOADING_STEPS.length), 1050);
    return () => clearInterval(id);
  }, [loading]);

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
    setLoading(true);
    setStepIndex(0);

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ package_name: trimmed }),
      });

      const body = await readResponseBody(response);

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
        setError("Cannot reach server");
      } else {
        setError(err.message || "Something went wrong.");
      }
      return null;
    } finally {
      setLoading(false);
    }
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
          label: "Model score",
          value: result.score.toFixed(3),
          meaning: "Probability-like risk output from 0 to 1.",
        },
        {
          label: "Postinstall",
          value: result.features?.has_postinstall ? "yes" : "no",
          meaning: FEATURE_MEANINGS.has_postinstall,
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
      <div className="bg-noise" aria-hidden="true" />
      <div className="bg-blob bg-blob-a" aria-hidden="true" />
      <div className="bg-blob bg-blob-b" aria-hidden="true" />

      <main className="shell">
        <header className="hero reveal">
          <div className="top-actions">
            <button type="button" className="secondary-button" onClick={() => setRecentOpen(true)}>
              Recent Analyses ({recentAnalyses.length})
            </button>
          </div>
          <p className="kicker">SCOPE Intelligence</p>
          <h1>Package Trust Analysis</h1>
          <p className="subtitle">Minimal, explainable security insight for NPM dependencies.</p>
        </header>

        <section className="glass card search-card reveal reveal-delay-1">
          <form onSubmit={onSubmit} className="search-form">
            <label htmlFor="package-input" className="label">
              Package
            </label>
            <div className="search-row">
              <input
                id="package-input"
                value={packageName}
                onChange={(e) => setPackageName(e.target.value)}
                onKeyDown={onPackageKeyDown}
                placeholder="Enter package name (e.g. react-domm)"
                className="search-input"
                required
              />
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

            <article className="glass card chart-card">
              <h2>Top Risk Factors</h2>
              <p className="muted">These are the strongest model factors for this prediction.</p>
              <FactorBars items={topFactors} />
            </article>
          </section>
        )}
      </main>

      <RecentDialog
        open={recentOpen}
        onClose={() => setRecentOpen(false)}
        recentAnalyses={recentAnalyses}
        onSelectRecent={handleSelectRecent}
      />
    </div>
  );
}
