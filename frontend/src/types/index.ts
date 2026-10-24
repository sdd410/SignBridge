/**
 * Type definitions for SignBridge Frontend.
 */

export interface ClassPrediction {
  label: string;
  confidence: number;
  index: number;
}

export interface PredictionEvent {
  type: string;
  label: string;
  raw_label: string;
  confidence: number;
  raw_confidence: number;
  is_accepted: boolean;
  is_stable: boolean;
  stability_progress: number;
  top_k: ClassPrediction[];
  latency_ms: number;
  fps: number;
  timestamp: number;
  accepted_char: string | null;
  transcript: string;
  roi_bbox?: [number, number, number, number];
}

export interface ModelStatus {
  status: 'LOADED' | 'NOT_FOUND' | 'ERROR' | 'INITIALIZING';
  model_type: string | null;
  num_classes: number;
  classes: string[];
  parameter_count: number;
  device: string;
  checkpoint_path: string;
  val_accuracy: number | null;
  message: string | null;
}

export interface ModelMetrics {
  top1_accuracy: number;
  top3_accuracy: number;
  macro_avg_f1: number;
  weighted_avg_f1: number;
  total_test_samples: number;
  class_count: number;
  classes: string[];
  per_class_metrics: Record<string, {
    precision: number;
    recall: number;
    f1_score: number;
    support: number;
  }>;
  confusion_matrix: number[][];
}

export interface AppSettings {
  confidence_threshold: number;
  smoothing_window_size: number;
  stability_duration_ms: number;
  repeat_lockout_ms: number;
  auto_insert_transcript: boolean;
  roi_mode: 'guide_roi' | 'skin_contour' | 'full_frame';
  target_fps: number;
  input_image_size: number;
}

export interface TranscriptState {
  text: string;
  character_count: number;
  word_count: number;
  last_updated: number;
  history: Array<{
    char: string;
    confidence: number;
    timestamp: number;
    source: string;
    current_length: number;
  }>;
}

export interface CameraDevice {
  deviceId: string;
  label: string;
}
