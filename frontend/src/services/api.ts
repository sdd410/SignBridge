import { AppSettings, ModelMetrics, ModelStatus, PredictionEvent, TranscriptState } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL || '';

export async function fetchHealth(): Promise<{ status: string; app: string; version: string; uptime_sec: number; model_ready: boolean }> {
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${BASE_URL}/api/model/status`);
  if (!res.ok) throw new Error(`Failed to fetch model status: ${res.statusText}`);
  return res.json();
}

export async function fetchModelMetrics(): Promise<ModelMetrics> {
  const res = await fetch(`${BASE_URL}/api/model/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch model metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchSettings(): Promise<AppSettings> {
  const res = await fetch(`${BASE_URL}/api/settings`);
  if (!res.ok) throw new Error(`Failed to fetch settings: ${res.statusText}`);
  return res.json();
}

export async function updateSettings(settings: Partial<AppSettings>): Promise<AppSettings> {
  const res = await fetch(`${BASE_URL}/api/settings`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw new Error(`Failed to update settings: ${res.statusText}`);
  return res.json();
}

export async function fetchTranscript(): Promise<TranscriptState> {
  const res = await fetch(`${BASE_URL}/api/transcripts`);
  if (!res.ok) throw new Error(`Failed to fetch transcript: ${res.statusText}`);
  return res.json();
}

export async function mutateTranscript(action: 'append' | 'space' | 'backspace' | 'clear' | 'set', text?: string): Promise<TranscriptState> {
  const res = await fetch(`${BASE_URL}/api/transcripts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, text }),
  });
  if (!res.ok) throw new Error(`Transcript action failed: ${res.statusText}`);
  return res.json();
}

export async function predictSingleFrame(base64Image: string): Promise<PredictionEvent> {
  const res = await fetch(`${BASE_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64Image }),
  });
  if (!res.ok) throw new Error(`Single frame prediction failed: ${res.statusText}`);
  return res.json();
}
