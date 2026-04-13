import { useEffect, useMemo, useState } from "react";

const LOADING_STEPS = [
  "Scanning package metadata",
  "Checking dependency reputation",
  "Analyzing security signals",
  "Generating risk insights",
];

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function normalizePackageName(value) {
  return value.trim();
}

function validatePackageName(value) {
  const trimmed = normalizePackageName(value);

  if (!trimmed) {
    return "Enter a package name.";
  }

  if (/\s/.test(trimmed)) {
    return "Package names cannot contain spaces.";
  }

  return "";
}

function getRiskTone(riskLevel = "") {
  const normalized = String(riskLevel).toUpperCase();

  if (normalized.includes("HEALTHY") || normalized.includes("LOW")) {
    return "safe";
  }

  if (normalized.includes("MODERATE") || normalized.includes("MEDIUM") || normalized.includes("CAUTION")) {
    return "warning";
  }

  return "danger";
}

function getRiskLabel(score, riskLevel) {
  if (riskLevel) {
    return String(riskLevel).replaceAll("_", " ");
  }

  if (score < 0.33) return "Low Risk";
  if (score < 0.66) return "Moderate Risk";
  return "High Risk";
}

function formatReport(result) {
  const lines = [
    `Package: ${result.package}`,
    `Risk score: ${(result.score * 100).toFixed(1)}%`,
    `Risk level: ${getRiskLabel(result.score, result.risk_level)}`,
  ];

  if (result.features) {
    lines.push("", "Features:");
    Object.entries(result.features).forEach(([key, value]) => {
      lines.push(`- ${key}: ${value}`);
    });
  }

  if (Array.isArray(result.explanations) && result.explanations.length) {
    lines.push("", "Top factors:");
    result.explanations.slice(0, 4).forEach((factor) => {
      lines.push(`- ${factor.feature}: ${factor.shap_value}`);
    });
  }

  return lines.join("\n");
}

function getAlternativeName(result) {
  if (!result) return "";

  if (typeof result.suggested_alternative === "string") {
    return result.suggested_alternative;
  }

  if (result.suggested_alternative?.package) {
    return result.suggested_alternative.package;
  }

  if (result.suggestion?.name) {
    return result.suggestion.name;
  }

  return "";
}

function buildShareUrl(packageName) {
  return `https://scope.dev/check/${encodeURIComponent(packageName)}`;
}

async function readResponseBody(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

function scoreToRiskColor(score) {
  if (score < 0.33) return "var(--success)";
  if (score < 0.66) return "var(--warning)";
  return "var(--danger)";
}

function riskBadge(score, level) {
  return getRiskLabel(score, level);
}

function Gauge({ score, riskLevel }) {
  const size = 176;
  const stroke = 8;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const progress = Math.max(0, Math.min(1, score));
  const dashOffset = circumference - progress * circumference;
  const color = scoreToRiskColor(score);
  const percent = Math.round(progress * 100);

  return (
    <div className="gauge-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="gauge-svg" aria-hidden="true">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          className="gauge-track"
          strokeWidth={stroke}
          fill="none"
        />
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
        <span className="gauge-subtitle">{riskBadge(score, riskLevel)}</span>
      </div>
    </div>
  );
}

function FactorBars({ items }) {
  if (!items?.length) {
    return <p className="muted">No factor details were returned for this package.</p>;
  }

  const maxAbs = Math.max(...items.map((item) => Math.abs(item.shap_value || 0)), 1e-9);

  return (
    <div className="bars">
      {items.slice(0, 6).map((item) => {
        const value = item.shap_value || 0;
        const widthPct = Math.max(8, Math.round((Math.abs(value) / maxAbs) * 100));
        const positive = value >= 0;
        const direction = positive ? "↑" : "↓";

        return (
          <div className="bar-row" key={`${item.feature}-${value}`}>
            <div className="bar-meta">
              <span className="mono feature-name">{item.feature}</span>
              <span className={`mono feature-value ${positive ? "positive" : "negative"}`}>
                {direction} {value.toFixed(3)}
              </span>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${positive ? "bar-pos" : "bar-neg"}`}
                style={{ width: `${widthPct}%` }}
                title={item.description || ""}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function renderFactors(factors) {
  return <FactorBars items={factors} />;
}

function renderResults(data, onCopyReport, onCopyShareUrl, onAnalyzeAlternative) {
  const riskTone = getRiskTone(data.risk_level);
  const alternativeName = getAlternativeName(data);
  const factors = Array.isArray(data.explanations) ? data.explanations.slice(0, 4) : [];

  return (
    <section className="results reveal reveal-delay-2">
      <article className={`glass card score-card risk-${riskTone}`}>
        <h2>Risk Insights</h2>
        <div className="score-layout">
          <Gauge score={data.score} riskLevel={data.risk_level} />
          <div className="score-details">
            <p className="score-line">
              <span className="label">Package</span>
              <span className="mono">{data.package}</span>
            </p>
            <p className="score-line">
              <span className="label">Risk Level</span>
              <span>{riskBadge(data.score, data.risk_level)}</span>
            </p>
            <p className="score-line">
              <span className="label">Model Score</span>
              <span className="mono">{data.score.toFixed(3)}</span>
            </p>

            <div className="result-actions">
              <button type="button" className="secondary-button" onClick={() => onCopyReport(data)}>
                Copy Report
              </button>
              <button type="button" className="secondary-button" onClick={() => onCopyShareUrl(data.package)}>
                Share URL
              </button>
            </div>
          </div>
        </div>

        {alternativeName && (
          <div className="typosquat-note">
            <span>May be typosquatting. Did you mean:</span>
            <button type="button" className="inline-link" onClick={() => onAnalyzeAlternative(alternativeName)}>
              {alternativeName}?
            </button>
          </div>
        )}
      </article>

      <article className="glass card chart-card">
        <h2>Top Risk Factors</h2>
        {renderFactors(factors)}
      </article>
    </section>
  );
}

function renderRecentAnalysisCard(entry, onSelect) {
  const tone = getRiskTone(entry.risk_level);

  return (
    <button type="button" className={`recent-card recent-${tone}`} onClick={() => onSelect(entry)}>
      <span className="recent-name">{entry.package}</span>
      <span className="recent-score mono">{Math.round(entry.score * 100)}%</span>
      <span className="recent-dot" aria-hidden="true" />
    </button>
  );
}

export default function App() {
  const [packageName, setPackageName] = useState("react-domm");
  const [loading, setLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [recentAnalyses, setRecentAnalyses] = useState([]);

  useEffect(() => {
    if (!loading) return undefined;

    const id = setInterval(() => {
      setStepIndex((prev) => (prev + 1) % LOADING_STEPS.length);
    }, 1100);

    return () => clearInterval(id);
  }, [loading]);

  const insights = useMemo(() => {
    if (!result?.explanations?.length) return [];
    return result.explanations.slice(0, 4);
  }, [result]);

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
        const message = typeof body === "string" ? body : body?.detail || "Analysis failed. Please try again.";
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

  return (
    <div className="page">
      <div className="bg-noise" aria-hidden="true" />
      <div className="bg-blob bg-blob-a" aria-hidden="true" />
      <div className="bg-blob bg-blob-b" aria-hidden="true" />

      <main className="shell">
        <header className="hero reveal">
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
            <p className="loading-text">{LOADING_STEPS[stepIndex]}<span className="cursor">|</span></p>
          </section>
        )}

        {error && (
          <section className="glass card error-card reveal reveal-delay-2">
            <p>{error}</p>
          </section>
        )}

        {result && !loading && (
          renderResults(result, handleCopyReport, handleCopyShareUrl, handleAnalyzeAlternative)
        )}

        {recentAnalyses.length > 0 && (
          <section className="glass card recent-section reveal reveal-delay-2">
            <div className="recent-header">
              <h2>Recent Analyses</h2>
              <p className="muted">Last 5 packages analyzed in this session.</p>
            </div>

            <div className="recent-grid">
              {recentAnalyses.map((entry) => renderRecentAnalysisCard(entry, handleSelectRecent))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
