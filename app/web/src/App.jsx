import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Cpu,
  FileVideo,
  Gauge,
  ImageIcon,
  Layers3,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  SlidersHorizontal,
  Table2,
  UploadCloud
} from "lucide-react";

const imageExtensions = new Set([".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"]);
const videoExtensions = new Set([".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"]);

function extensionOf(file) {
  const name = file?.name || "";
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

function detectKind(file) {
  if (!file) return "";
  const extension = extensionOf(file);
  if (imageExtensions.has(extension)) return "image";
  if (videoExtensions.has(extension)) return "video";
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("video/")) return "video";
  return "";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatScore(value) {
  return typeof value === "number" ? `${Math.round(value * 1000) / 10}%` : "-";
}

function formatSeconds(value) {
  return typeof value === "number" ? `${value.toFixed(value >= 10 ? 1 : 2)}s` : "-";
}

function verdictClass(status = "") {
  const normalized = status.toLowerCase();
  if (normalized.includes("real") || normalized === "ok") return "good";
  if (normalized.includes("uncertain")) return "warn";
  if (normalized.includes("fake") || normalized.includes("suspicious") || normalized.includes("error")) return "bad";
  return "neutral";
}

function ModelStatusRow({ item }) {
  const ready = Boolean(item.available);
  return (
    <div className="model-status-row">
      <span
        className={`model-light ${ready ? "ready" : "missing"}`}
        aria-label={ready ? "Ready" : "Missing"}
        title={item.message || (ready ? "Ready" : "Missing")}
      />
      <div>
        <strong>{item.model_name}</strong>
      </div>
    </div>
  );
}

function ModelResult({ result }) {
  const state = verdictClass(result.status);
  return (
    <div className="model-result">
      <div>
        <strong>{result.model_name || result.model_id || "Model"}</strong>
      </div>
      <span className={`status-pill ${state}`}>{result.status || "-"}</span>
      <span className="score-chip">{formatScore(result.fake_score)}</span>
    </div>
  );
}

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="metric">
      <Icon size={18} aria-hidden="true" />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PipelineMatrix({ pipelines, models }) {
  if (!pipelines.length) {
    return (
      <div className="no-score">
        <AlertTriangle size={18} aria-hidden="true" />
        <span>No preprocessing pipeline returned a score.</span>
      </div>
    );
  }

  return (
    <div className="pipeline-scroll">
      <table className="pipeline-matrix">
        <thead>
          <tr>
            <th scope="col">Pipeline</th>
            <th scope="col">Preprocessing</th>
            <th scope="col">Purpose</th>
            {models.map((item) => (
              <th key={item.model_id} scope="col">{item.model_name}</th>
            ))}
            <th scope="col">Average</th>
          </tr>
        </thead>
        <tbody>
          {pipelines.map((pipeline) => {
            const averageState = verdictClass(pipeline.summary?.status);
            return (
              <tr key={pipeline.pipeline_id}>
                <th scope="row">
                  <strong>{pipeline.pipeline_name}</strong>
                </th>
                <td>{pipeline.preprocessing}</td>
                <td>{pipeline.purpose}</td>
                {models.map((modelItem) => {
                  const modelResult = (pipeline.results || []).find((item) => item.model_id === modelItem.model_id);
                  const state = verdictClass(modelResult?.status);
                  return (
                    <td key={`${pipeline.pipeline_id}-${modelItem.model_id}`}>
                      <span className={`matrix-score ${state}`}>{formatScore(modelResult?.fake_score)}</span>
                      <small>{modelResult?.status || "-"}</small>
                    </td>
                  );
                })}
                <td className="average-cell">
                  <span className={`matrix-score ${averageState}`}>{formatScore(pipeline.summary?.fake_score)}</span>
                  <small>{pipeline.summary?.status || "-"}</small>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState("checking");
  const [models, setModels] = useState([]);
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("available");
  const [imageRunMode, setImageRunMode] = useState("standard");
  const [videoPreset, setVideoPreset] = useState("quick");
  const [videoFrames, setVideoFrames] = useState("");
  const [includeDetails, setIncludeDetails] = useState(false);
  const [toast, setToast] = useState("");
  const [toastError, setToastError] = useState(false);
  const [result, setResult] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const fileInputRef = useRef(null);

  const kind = detectKind(file);
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  async function checkHealth() {
    try {
      const response = await fetch("/health", { cache: "no-store" });
      if (!response.ok) throw new Error("Backend offline");
      setHealth("online");
    } catch {
      setHealth("offline");
    }
  }

  async function loadModels(inputKind = "") {
    const suffix = inputKind ? `?input_type=${encodeURIComponent(inputKind)}` : "";
    const response = await fetch(`/models${suffix}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load model list");
    const payload = await response.json();
    const nextModels = payload.models || [];
    setModels(nextModels);
    return nextModels;
  }

  useEffect(() => {
    checkHealth();
    loadModels().catch((error) => {
      setToast(error.message);
      setToastError(true);
    });
  }, []);

  async function handleFileChange(event) {
    const selected = event.target.files?.[0] || null;
    const selectedKind = detectKind(selected);
    setFile(selected);
    setResult(null);
    if (selectedKind !== "image") setImageRunMode("standard");

    if (!selected) {
      setToast("");
      await loadModels();
      return;
    }
    if (!selectedKind) {
      setToast("Choose a supported image or video file.");
      setToastError(true);
      return;
    }

    setToast("");
    setToastError(false);
    try {
      const nextModels = await loadModels(selectedKind);
      const selectedModel = nextModels.find((item) => item.model_id === model);
      if (selectedModel && !selectedModel.available) setModel("available");
    } catch (error) {
      setToast(error.message);
      setToastError(true);
    }
  }

  function resetForm() {
    setFile(null);
    setModel("available");
    setImageRunMode("standard");
    setVideoPreset("quick");
    setVideoFrames("");
    setIncludeDetails(false);
    setToast("");
    setToastError(false);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    loadModels().catch((error) => {
      setToast(error.message);
      setToastError(true);
    });
  }

  async function analyzeSelectedFile(event) {
    event.preventDefault();
    if (!file || !kind) {
      setToast("Choose a supported image or video file.");
      setToastError(true);
      return;
    }

    const data = new FormData();
    data.append("file", file);
    data.append("model", model);
    data.append("include_details", includeDetails ? "true" : "false");
    if (kind === "video") {
      data.append("video_preset", videoPreset);
      if (videoFrames.trim()) data.append("video_frames", videoFrames.trim());
    }

    const usePipelines = kind === "image" && imageRunMode === "pipelines";

    setSubmitting(true);
    setToast(usePipelines ? "Running preprocessing comparison..." : "Uploading and running analysis...");
    setToastError(false);
    try {
      const endpoint = usePipelines ? "/analyze-image-pipelines" : kind === "image" ? "/analyze-image" : "/analyze-video";
      const response = await fetch(endpoint, { method: "POST", body: data });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Analysis failed");
      setResult(payload);
      setToast("Analysis complete.");
    } catch (error) {
      setToast(error.message || "Analysis failed");
      setToastError(true);
    } finally {
      setSubmitting(false);
    }
  }

  const uniqueModels = new Map();
  for (const item of models) {
    const existing = uniqueModels.get(item.model_id);
    if (!existing) {
      uniqueModels.set(item.model_id, { ...item });
    } else {
      existing.available = existing.available || item.available;
      if (existing.message !== "ready" && item.message === "ready") existing.message = item.message;
    }
  }

  const modelList = Array.from(uniqueModels.values());
  const readyCount = modelList.filter((item) => item.available).length;
  const summary = result?.summary || {};
  const resultRows = Array.isArray(result?.results) ? result.results : [];
  const isPipelineResult = result?.analysis_type === "preprocessing_comparison";
  const pipelineRows = Array.isArray(result?.pipelines) ? result.pipelines : [];
  const pipelineModels = Array.isArray(result?.models) ? result.models : [];
  const modelRunCount = isPipelineResult
    ? `${pipelineRows.length || 0} x ${pipelineModels.length || 0}`
    : resultRows.length || "-";
  const fakeScore = typeof summary.fake_score === "number" ? Math.max(0, Math.min(100, summary.fake_score * 100)) : 0;
  const statusClass = verdictClass(summary.status);

  return (
    <main className="app-shell">
      <header className="masthead">
        <div>
          <h1>Deepfake Model Tester</h1>
        </div>

        <div className="top-stats" aria-live="polite">
          <div className={`health-badge ${health}`}>
            <Activity size={16} aria-hidden="true" />
            <span>{health === "online" ? "Online" : health === "offline" ? "Offline" : "Checking"}</span>
          </div>
          <div className="stat-badge">
            <Cpu size={16} aria-hidden="true" />
            <span>{readyCount}/{modelList.length || 0} ready</span>
          </div>
        </div>
      </header>

      <section className="workbench">
        <form className="action-dock" onSubmit={analyzeSelectedFile}>
          <div className="dock-title">
            <ShieldCheck size={21} aria-hidden="true" />
            <h2>Run</h2>
          </div>

          <label className={`drop-zone ${file && kind ? "loaded" : ""}`} htmlFor="fileInput">
            <input
              ref={fileInputRef}
              id="fileInput"
              type="file"
              accept="image/*,video/*"
              onChange={handleFileChange}
            />
            <span className="drop-icon">
              {kind === "video" ? <FileVideo size={28} aria-hidden="true" /> : kind === "image" ? <ImageIcon size={28} aria-hidden="true" /> : <UploadCloud size={28} aria-hidden="true" />}
            </span>
            <strong>{file ? file.name : "Select media"}</strong>
            {file && kind && <span>{`${kind} - ${formatBytes(file.size)}`}</span>}
          </label>

          <div className="control-group">
            <label htmlFor="modelSelect">Model set</label>
            <select id="modelSelect" className="select" value={model} onChange={(event) => setModel(event.target.value)}>
              <option value="available">Available models</option>
              <option value="all">All configured models</option>
              {modelList.map((item) => (
                <option key={item.model_id} value={item.model_id} disabled={!item.available} title={item.message || ""}>
                  {item.model_name} - {item.available ? "ready" : "missing"}
                </option>
              ))}
            </select>
          </div>

          {kind === "image" && (
            <div className="control-group">
              <span className="label-row">
                <Table2 size={15} aria-hidden="true" />
                Image run
              </span>
              <div className="segmented mode-select">
                {[
                  ["standard", "Standard"],
                  ["pipelines", "Pipelines"]
                ].map(([value, label]) => (
                  <label key={value}>
                    <input type="radio" name="imageRunMode" value={value} checked={imageRunMode === value} onChange={() => setImageRunMode(value)} />
                    {label}
                  </label>
                ))}
              </div>
            </div>
          )}

          {kind === "video" && (
            <div className="video-controls">
              <div className="control-group">
                <span className="label-row">
                  <SlidersHorizontal size={15} aria-hidden="true" />
                  Video preset
                </span>
                <div className="segmented">
                  {["quick", "balanced", "thorough"].map((value) => (
                    <label key={value}>
                      <input type="radio" name="videoPreset" value={value} checked={videoPreset === value} onChange={() => setVideoPreset(value)} />
                      {value[0].toUpperCase() + value.slice(1)}
                    </label>
                  ))}
                </div>
              </div>

              <div className="control-group">
                <label htmlFor="videoFrames">Frame limit</label>
                <input id="videoFrames" className="input" type="number" min="1" step="1" placeholder="Preset" value={videoFrames} onChange={(event) => setVideoFrames(event.target.value)} />
              </div>
            </div>
          )}

          <label className="switch-row">
            <input type="checkbox" checked={includeDetails} onChange={(event) => setIncludeDetails(event.target.checked)} />
            <span />
            <strong>Model details</strong>
          </label>

          <div className="button-row">
            <button className="primary" type="submit" disabled={submitting}>
              <ScanLine size={18} aria-hidden="true" />
              <span>{submitting ? "Running" : "Analyze"}</span>
            </button>
            <button className="icon-button" type="button" onClick={resetForm} aria-label="Reset form" title="Reset">
              <RotateCcw size={18} aria-hidden="true" />
            </button>
          </div>

          <div className={`toast ${toastError ? "error" : ""}`} role="status" aria-live="polite">{toast}</div>
        </form>

        <section className="media-stage">
          <div className="stage-toolbar">
            <div>
              <span>Preview</span>
              {file && kind && <strong>{file.name}</strong>}
            </div>
            <span className={`media-chip ${kind || "empty"}`}>{kind || "empty"}</span>
          </div>

          <div className={`stage-canvas ${file && kind ? "" : "empty"}`}>
            {!file || !kind ? (
              <div className="empty-stage">
                <UploadCloud size={34} aria-hidden="true" />
                <span>Waiting for media</span>
              </div>
            ) : kind === "image" ? (
              <img src={previewUrl} alt={file.name} />
            ) : (
              <video src={previewUrl} controls preload="metadata" />
            )}
          </div>
        </section>

        <aside className="model-rail">
          <div className="rail-header">
            <Layers3 size={19} aria-hidden="true" />
            <h2>Models</h2>
          </div>
          <div className="model-list">
            {modelList.length ? modelList.map((item) => <ModelStatusRow key={item.model_id} item={item} />) : (
              <div className="empty-models">No model data</div>
            )}
          </div>
        </aside>
      </section>

      <section className={`verdict-board ${result ? "active" : ""}`}>
        <div className="verdict-main">
          <div className={`verdict-ring ${statusClass}`} style={{ "--score": `${fakeScore}%` }}>
            <span>{formatScore(summary.fake_score)}</span>
          </div>
          <div>
            <span className={`status-pill ${statusClass}`}>{summary.status || "waiting"}</span>
            <h2>{result ? "Analysis Result" : "Verdict Pending"}</h2>
          </div>
        </div>

        <div className="metric-grid">
          <Metric icon={Gauge} label="Fake score" value={formatScore(summary.fake_score)} />
          <Metric icon={ShieldCheck} label="Real score" value={formatScore(summary.real_score)} />
          <Metric icon={Clock3} label="Processing" value={formatSeconds(summary.processing_time)} />
          <Metric icon={CheckCircle2} label={isPipelineResult ? "Pipelines x models" : "Models run"} value={modelRunCount} />
        </div>

        {result && (
          <>
            {isPipelineResult ? (
              <PipelineMatrix pipelines={pipelineRows} models={pipelineModels} />
            ) : (
              <div className="result-list">
                {resultRows.length ? resultRows.map((item) => <ModelResult key={`${item.model_id}-${item.status}`} result={item} />) : (
                  <div className="no-score">
                    <AlertTriangle size={18} aria-hidden="true" />
                    <span>No model returned a score.</span>
                  </div>
                )}
              </div>
            )}
            <details className="raw-json">
              <summary>Raw JSON</summary>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </details>
          </>
        )}
      </section>
    </main>
  );
}
