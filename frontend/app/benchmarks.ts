// Test-set results from reports/*/metrics.json (official test splits).
// Kept as static data so the page loads without asking the API.
// If you retrain, update these numbers from the new metrics.json files.

export type Experiment = "A" | "B" | "C";

export const EXPERIMENTS: Record<Experiment, { label: string; trainedOn: string }> = {
  A: { label: "A", trainedOn: "Celeb-DF only" },
  B: { label: "B", trainedOn: "FF++ only" },
  C: { label: "C", trainedOn: "Both (deployed)" },
};

// Video-level AUC. `seen: false` = the model never trained on that dataset.
export const GENERALIZATION: Record<Experiment, { ffpp: number; celebdf: number; seen: { ffpp: boolean; celebdf: boolean } }> = {
  A: { ffpp: 0.633, celebdf: 0.9995, seen: { ffpp: false, celebdf: true } },
  B: { ffpp: 0.987, celebdf: 0.852, seen: { ffpp: true, celebdf: false } },
  C: { ffpp: 0.983, celebdf: 0.9995, seen: { ffpp: true, celebdf: true } },
};

// Video-level AUC on the FF++ test split, per manipulation method.
export const PER_METHOD: { method: string; kind: string; A: number; B: number; C: number }[] = [
  { method: "Deepfakes", kind: "Face swap", A: 0.822, B: 0.996, C: 0.994 },
  { method: "FaceSwap", kind: "Face swap", A: 0.457, B: 0.994, C: 0.993 },
  { method: "Face2Face", kind: "Expression edit", A: 0.657, B: 0.99, C: 0.985 },
  { method: "NeuralTextures", kind: "Expression edit", A: 0.595, B: 0.97, C: 0.96 },
];

// Deployed model (C) on both test sets combined (12,180 frames / 1,218 videos).
export const DEPLOYED = {
  frameAccuracy: 0.938,
  videoAccuracy: 0.961,
  frameAuc: 0.979,
  videoAuc: 0.99,
  eceRaw: 0.039,
  eceCalibrated: 0.009,
  testFrames: 12180,
};

// Frame AUC of the deployed model when test images are degraded.
export const ROBUSTNESS: { condition: string; auc: number }[] = [
  { condition: "Original", auc: 0.979 },
  { condition: "JPEG quality 50", auc: 0.964 },
  { condition: "JPEG quality 30", auc: 0.946 },
  { condition: "Blurred", auc: 0.962 },
  { condition: "Shrunk to ¼ size", auc: 0.913 },
];
