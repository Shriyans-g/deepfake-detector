"use client";

import { useCallback, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Verdict = "likely_fake" | "uncertain" | "likely_real";

type Prediction = {
  verdict: Verdict;
  prob_fake: number;
  raw_prob_fake: number;
  face_box: { x: number; y: number; w: number; h: number; score: number };
  face_crop: string;
  heatmap: string;
  heatmap_explains: "fake" | "real";
  model: { arch: string; temperature: number };
  latency_ms: number;
};

const VERDICT_TEXT: Record<Verdict, { title: string; body: string }> = {
  likely_fake: {
    title: "Likely manipulated",
    body: "The model found patterns it associates with face-swapped video frames.",
  },
  uncertain: {
    title: "Uncertain",
    body: "The score is too close to the middle to call either way. Treat this as no answer.",
  },
  likely_real: {
    title: "Likely authentic",
    body: "The model did not find the patterns it associates with face swaps.",
  },
};

export default function Home() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  const analyze = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setImageSize(null);
    setImageUrl(URL.createObjectURL(file));
    setLoading(true);

    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) {
        setError(typeof data.detail === "string" ? data.detail : "Something went wrong.");
      } else {
        setResult(data);
      }
    } catch {
      setError(`Could not reach the API at ${API_URL}. Is the backend running?`);
    } finally {
      setLoading(false);
    }
  }, []);

  const onFiles = (files: FileList | null) => {
    if (files && files[0]) analyze(files[0]);
  };

  return (
    <main className="page">
      <header className="header">
        <div className="brand">
          <span className="brand-mark" aria-hidden />
          <span>Deepfake Detector</span>
        </div>
        <p className="tagline">
          Face-swap detection with calibrated confidence and a heatmap of what the model looked at.
        </p>
      </header>

      <section
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          onFiles(e.dataTransfer.files);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          hidden
          onChange={(e) => onFiles(e.target.files)}
        />
        <p className="dropzone-title">Drop an image with a face, or click to upload</p>
        <p className="dropzone-sub">JPEG, PNG or WebP, up to 10 MB. Only the largest face is analyzed.</p>
      </section>

      {error && <div className="error">{error}</div>}

      {imageUrl && (
        <section className="results">
          <div className="panel">
            <h2 className="panel-label">Input</h2>
            <div className="image-frame">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imageUrl}
                alt="Uploaded"
                onLoad={(e) =>
                  setImageSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
                }
              />
              {result && imageSize && (
                <div
                  className="face-box"
                  style={{
                    left: `${(result.face_box.x / imageSize.w) * 100}%`,
                    top: `${(result.face_box.y / imageSize.h) * 100}%`,
                    width: `${(result.face_box.w / imageSize.w) * 100}%`,
                    height: `${(result.face_box.h / imageSize.h) * 100}%`,
                  }}
                />
              )}
              {loading && <div className="scanning" />}
            </div>
          </div>

          <div className="panel">
            <h2 className="panel-label">Analysis</h2>
            {loading && <p className="muted">Analyzing…</p>}
            {result && <ResultView result={result} showHeatmap={showHeatmap} onToggle={setShowHeatmap} />}
          </div>
        </section>
      )}

      <footer className="footer">
        Trained on Celeb-DF v2 face swaps. It is a screening aid, not proof: it can miss other kinds of
        manipulation (fully AI-generated images, edits) and is less reliable on heavily compressed or tiny faces.
      </footer>
    </main>
  );
}

function ResultView({
  result,
  showHeatmap,
  onToggle,
}: {
  result: Prediction;
  showHeatmap: boolean;
  onToggle: (v: boolean) => void;
}) {
  const pct = Math.round(result.prob_fake * 100);
  const rawPct = Math.round(result.raw_prob_fake * 100);
  const text = VERDICT_TEXT[result.verdict];

  return (
    <div className="analysis">
      <div className={`verdict verdict-${result.verdict}`}>
        <div className="verdict-title">{text.title}</div>
        <div className="verdict-body">{text.body}</div>
      </div>

      <div className="meter-block">
        <div className="meter-head">
          <span>Calibrated probability of manipulation</span>
          <span className="mono big">{pct}%</span>
        </div>
        <div className="meter">
          <div className="meter-band" title="Uncertain zone" />
          <div className="meter-needle" style={{ left: `${pct}%` }} />
        </div>
        <div className="meter-scale mono">
          <span>real</span>
          <span>uncertain</span>
          <span>fake</span>
        </div>
        <p className="muted small">
          Raw model output was {rawPct}%. Temperature scaling (T = {result.model.temperature.toFixed(2)}), fit on
          held-out data, adjusts it so the stated confidence matches how often the model is actually right.
        </p>
      </div>

      <div className="crop-block">
        <div className="crop-head">
          <span>
            {showHeatmap
              ? `Grad-CAM: regions that pushed the model toward "${result.heatmap_explains}"`
              : "Face crop the model analyzed"}
          </span>
          <button className="toggle" onClick={() => onToggle(!showHeatmap)}>
            {showHeatmap ? "Show crop" : "Show heatmap"}
          </button>
        </div>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="crop" src={showHeatmap ? result.heatmap : result.face_crop} alt="Face analysis" />
      </div>

      <div className="meta mono">
        <span>{result.model.arch}</span>
        <span>face conf {Math.round(result.face_box.score * 100)}%</span>
        <span>{Math.round(result.latency_ms)} ms</span>
      </div>
    </div>
  );
}
