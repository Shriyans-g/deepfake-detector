"use client";

import { useCallback, useRef, useState } from "react";
import { DEPLOYED, EXPERIMENTS, GENERALIZATION, PER_METHOD, ROBUSTNESS, type Experiment } from "./benchmarks";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const REPO_URL = "https://github.com/Shriyans-g/deepfake-detector";

type Verdict = "likely_fake" | "uncertain" | "likely_real";
type Region = "eyes" | "nose" | "mouth" | "rest_of_face" | "outside_face";

type Prediction = {
  verdict: Verdict;
  prob_fake: number;
  raw_prob_fake: number;
  face_box: { x: number; y: number; w: number; h: number; score: number };
  face_crop: string;
  heatmap: string;
  heatmap_explains: "fake" | "real";
  attention: { region: Region; share: number }[];
  quality: { face_px: number; sharpness: number; detector_confidence: number; faces_in_image: number; notes: string[] };
  model: { arch: string; temperature: number };
  latency_ms: number;
};

const VERDICT_COPY: Record<Verdict, { title: string; body: string }> = {
  likely_fake: {
    title: "Likely manipulated",
    body: "The face shows patterns the model learned from face swaps and expression edits.",
  },
  uncertain: {
    title: "Can't tell",
    body: "The score sits in the middle band, where the model is wrong too often to make a call.",
  },
  likely_real: {
    title: "Likely authentic",
    body: "The model found none of the patterns it associates with manipulated faces.",
  },
};

const REGION_NAMES: Record<Region, string> = {
  eyes: "Eyes",
  nose: "Nose",
  mouth: "Mouth",
  rest_of_face: "Cheeks, forehead and jaw",
  outside_face: "Hair, neck and background",
};

const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

export default function Home() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
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
      if (!response.ok) setError(typeof data.detail === "string" ? data.detail : "The analysis failed. Try another image.");
      else setResult(data);
    } catch {
      setError(`Can't reach the analysis server at ${API_URL}. Check that it's running.`);
    } finally {
      setLoading(false);
    }
  }, []);

  const pick = () => inputRef.current?.click();
  const onFiles = (files: FileList | null) => files?.[0] && analyze(files[0]);
  const hasImage = imageUrl !== null;

  return (
    <div
      className={`shell ${dragging ? "is-dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        onFiles(e.dataTransfer.files);
      }}
    >
      <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => onFiles(e.target.files)} />

      <header className="topbar">
        <a className="wordmark" href="#top">
          <ScanMark />
          Deepfake Detector
        </a>
        <nav>
          <a href="#results">Results</a>
          <a href="#method" className="nav-optional">How it works</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            Source
          </a>
        </nav>
      </header>

      <main id="top">
        {!hasImage ? (
          <section className="intro">
            <h1>Is this face real?</h1>
            <p className="lede">
              Upload a photo with a face. A ResNet50 trained on 83,000 face crops from FaceForensics++ and Celeb-DF
              estimates whether it was manipulated, says how sure it is, and shows which parts of the face drove the
              decision.
            </p>
            <button className="dropzone" onClick={pick} type="button">
              <span className="dropzone-title">Choose a photo or drop it anywhere on this page</span>
              <span className="dropzone-sub">JPEG, PNG or WebP, up to 10 MB. Photos are analyzed and discarded, never stored.</span>
            </button>
            <dl className="headline-stats">
              <div>
                <dt>Accuracy per image</dt>
                <dd>{pct(DEPLOYED.frameAccuracy, 1)}</dd>
              </div>
              <div>
                <dt>AUC per video</dt>
                <dd>{DEPLOYED.videoAuc.toFixed(2)}</dd>
              </div>
              <div>
                <dt>On a dataset it never saw</dt>
                <dd>{GENERALIZATION.B.celebdf.toFixed(2)} AUC</dd>
              </div>
            </dl>
          </section>
        ) : (
          <section className="workspace" aria-live="polite">
            <div className="source">
              <div className="source-frame">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imageUrl}
                  alt="Your upload"
                  onLoad={(e) => setImageSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                />
                {result && imageSize && (
                  <div
                    className="face-box"
                    style={{
                      left: pct(result.face_box.x / imageSize.w, 2),
                      top: pct(result.face_box.y / imageSize.h, 2),
                      width: pct(result.face_box.w / imageSize.w, 2),
                      height: pct(result.face_box.h / imageSize.h, 2),
                    }}
                  />
                )}
                {loading && <div className="scanline" />}
              </div>
              <button className="secondary" onClick={pick} type="button">
                Analyze another photo
              </button>
            </div>

            <div className="report">
              {loading && <p className="status">Finding the face and running the model…</p>}
              {error && (
                <div className="error" role="alert">
                  <strong>No result.</strong> {error}
                </div>
              )}
              {result && <Report result={result} />}
            </div>
          </section>
        )}

        <ModelCard />
        <Method />
      </main>

      <footer className="footer">
        <p>
          A screening aid, not proof. Trained on academic datasets of face swaps and expression edits in video; it is
          not built for fully AI-generated images and is less reliable on tiny or heavily compressed faces.
        </p>
        <p>
          Built by Shriyans Gunda. <a href={REPO_URL}>Code, training notebook and full results are on GitHub.</a>
        </p>
      </footer>

      <div className="drop-overlay" aria-hidden>
        <span>Drop to analyze</span>
      </div>
    </div>
  );
}

function Report({ result }: { result: Prediction }) {
  const copy = VERDICT_COPY[result.verdict];
  const prob = result.prob_fake;
  const top = result.attention[0];
  const towards = result.heatmap_explains;
  const [view, setView] = useState<"heatmap" | "crop">("heatmap");

  return (
    <article className={`verdict-${result.verdict}`}>
      <header className="verdict">
        <p className="verdict-score">
          <span className="verdict-number">{pct(prob)}</span>
          <span className="verdict-unit">chance of manipulation</span>
        </p>
        <h2>{copy.title}</h2>
        <p className="verdict-body">{copy.body}</p>
      </header>

      <div className="scale" role="img" aria-label={`Score ${pct(prob)} on a scale from real to fake`}>
        <div className="scale-track">
          <span className="scale-zone zone-real" />
          <span className="scale-zone zone-unsure" />
          <span className="scale-zone zone-fake" />
          <span className="scale-needle" style={{ left: pct(prob, 1) }} />
        </div>
        <div className="scale-labels">
          <span>Authentic</span>
          <span>Can&apos;t tell</span>
          <span>Manipulated</span>
        </div>
        <p className="fine">
          Calibrated with temperature scaling (T = {result.model.temperature.toFixed(2)}): on held-out test data, the
          stated chance matches how often the model is right. The raw output was {pct(result.raw_prob_fake)}.
        </p>
      </div>

      <section className="evidence">
        <h3>What the model looked at</h3>
        <div className="evidence-grid">
          <figure className="evidence-image">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={view === "heatmap" ? result.heatmap : result.face_crop} alt="The face as the model saw it" />
            <figcaption>
              <button
                type="button"
                className={view === "heatmap" ? "seg active" : "seg"}
                onClick={() => setView("heatmap")}
                aria-pressed={view === "heatmap"}
              >
                Heatmap
              </button>
              <button
                type="button"
                className={view === "crop" ? "seg active" : "seg"}
                onClick={() => setView("crop")}
                aria-pressed={view === "crop"}
              >
                Face crop
              </button>
            </figcaption>
          </figure>

          <div>
            <p className="evidence-lead">
              {top.share > 0
                ? `Most of the evidence for “${towards}” came from the ${REGION_NAMES[top.region].toLowerCase()} (${pct(top.share)} of the heatmap).`
                : "The heatmap was empty for this image."}
            </p>
            <ul className="regions">
              {result.attention.map((r) => (
                <li key={r.region}>
                  <span className="region-name">{REGION_NAMES[r.region]}</span>
                  <span className="region-bar">
                    <span style={{ width: pct(r.share, 1) }} />
                  </span>
                  <span className="region-value">{pct(r.share)}</span>
                </li>
              ))}
            </ul>
            <p className="fine">
              Grad-CAM weights the network&apos;s last layer by how much each part pushed the score. Face swaps tend to
              leave traces at the blend line along the jaw and cheeks; expression edits concentrate around the mouth.
            </p>
          </div>
        </div>
      </section>

      <section className="trust">
        <h3>How much to trust this result</h3>
        {result.quality.notes.length > 0 ? (
          <ul className="notes">
            {result.quality.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        ) : (
          <p className="clean">Nothing about this photo is known to lower accuracy.</p>
        )}
        <dl className="facts">
          <div>
            <dt>Face size</dt>
            <dd>{result.quality.face_px}px</dd>
          </div>
          <div>
            <dt>Face detection</dt>
            <dd>{pct(result.quality.detector_confidence)}</dd>
          </div>
          <div>
            <dt>Faces found</dt>
            <dd>{result.quality.faces_in_image}</dd>
          </div>
          <div>
            <dt>Analysis time</dt>
            <dd>{Math.round(result.latency_ms)} ms</dd>
          </div>
        </dl>
      </section>
    </article>
  );
}

function ModelCard() {
  const rows: Experiment[] = ["A", "B", "C"];
  return (
    <section id="results" className="results">
      <div className="section-head">
        <h2>How well it works</h2>
        <p>
          Measured on the official test splits of both datasets, with people the model never trained on. The deployed
          model is right on {pct(DEPLOYED.frameAccuracy, 1)} of single images and {pct(DEPLOYED.videoAccuracy, 1)} of
          videos.
        </p>
      </div>

      <div className="results-grid">
        <figure className="matrix">
          <figcaption>
            <strong>Generalization.</strong> AUC when tested on each dataset (1.0 is perfect, 0.5 is guessing). Outlined
            cells are datasets the model never saw.
          </figcaption>
          <table>
            <thead>
              <tr>
                <th scope="col">Trained on</th>
                <th scope="col">Tested on FF++</th>
                <th scope="col">Tested on Celeb-DF</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((k) => {
                const g = GENERALIZATION[k];
                return (
                  <tr key={k} className={k === "C" ? "deployed" : undefined}>
                    <th scope="row">{EXPERIMENTS[k].trainedOn}</th>
                    <AucCell value={g.ffpp} unseen={!g.seen.ffpp} />
                    <AucCell value={g.celebdf} unseen={!g.seen.celebdf} />
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="finding">
            A model trained on one kind of fake (Celeb-DF) is close to guessing on FF++. Training on FF++&apos;s four
            different methods transfers far better. Variety in training fakes matters more than volume.
          </p>
        </figure>

        <figure className="methods">
          <figcaption>
            <strong>By manipulation method</strong> (FF++ test set, AUC).
          </figcaption>
          <ul>
            {PER_METHOD.map((m) => (
              <li key={m.method}>
                <div className="method-name">
                  {m.method}
                  <span>{m.kind}</span>
                </div>
                <Bar label="Deployed" value={m.C} strong />
                <Bar label="Celeb-DF only" value={m.A} />
              </li>
            ))}
          </ul>
        </figure>

        <figure className="robust">
          <figcaption>
            <strong>Under rough conditions</strong> (deployed model, AUC).
          </figcaption>
          <ul>
            {ROBUSTNESS.map((r) => (
              <li key={r.condition}>
                <Bar label={r.condition} value={r.auc} strong={r.condition === "Original"} />
              </li>
            ))}
          </ul>
          <p className="fine">
            Calibration error falls from {DEPLOYED.eceRaw.toFixed(3)} to {DEPLOYED.eceCalibrated.toFixed(3)} after
            temperature scaling.
          </p>
        </figure>
      </div>
    </section>
  );
}

function AucCell({ value, unseen }: { value: number; unseen: boolean }) {
  // Shade from 0.5 (guessing) to 1.0 (perfect).
  const strength = Math.max(0, Math.min(1, (value - 0.5) / 0.5));
  return (
    <td
      className={[unseen ? "unseen" : "", strength > 0.6 ? "dark" : ""].join(" ").trim() || undefined}
      style={{ ["--s" as string]: strength.toFixed(3) }}
    >
      <span>{value >= 0.9995 ? "1.00" : value.toFixed(3)}</span>
      {unseen && <small>never seen</small>}
    </td>
  );
}

function Bar({ label, value, strong = false }: { label: string; value: number; strong?: boolean }) {
  // Bars start at 0.5 (chance), so a guessing model shows an empty bar.
  const width = Math.max(0, (value - 0.5) / 0.5);
  return (
    <div className={`bar ${strong ? "bar-strong" : ""}`}>
      <span className="bar-label">{label}</span>
      <span className="bar-track">
        <span style={{ width: pct(width, 1) }} />
      </span>
      <span className="bar-value">{value.toFixed(3)}</span>
    </div>
  );
}

function Method() {
  const steps = [
    ["Find the face", "OpenCV's YuNet detector locates the largest face and its eyes, nose and mouth."],
    ["Crop it the way training did", "A square crop with 30% margin, resized to 256px, using the same code that built the training set."],
    ["Score it", "A ResNet50 fine-tuned on 83,000 face crops outputs one number: how fake the face looks."],
    ["Calibrate", "Temperature scaling turns that number into a probability you can take at face value."],
    ["Explain", "Grad-CAM maps the score back onto the face, then we measure how much falls on each region."],
  ];
  return (
    <section id="method" className="method">
      <div className="section-head">
        <h2>How it works</h2>
        <p>Each upload goes through the same five steps. Nothing is stored.</p>
      </div>
      <ol>
        {steps.map(([title, body]) => (
          <li key={title}>
            <h3>{title}</h3>
            <p>{body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function ScanMark() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" />
      <circle cx="12" cy="11" r="3.2" />
      <path d="M6.8 18.2c1.1-2.2 3-3.3 5.2-3.3s4.1 1.1 5.2 3.3" />
    </svg>
  );
}
