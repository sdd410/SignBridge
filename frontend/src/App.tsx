import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { CameraPanel } from './components/CameraPanel';
import { PredictionPanel } from './components/PredictionPanel';
import { TranscriptPanel } from './components/TranscriptPanel';
import { SettingsModal } from './components/SettingsModal';
import { ModelMetricsModal } from './components/ModelMetricsModal';
import { SignsGuideModal } from './components/SignsGuideModal';
import { DatasetStudioModal } from './components/DatasetStudioModal';
import { useCamera } from './hooks/useCamera';
import { SignBridgeWebSocket, ConnectionStatus } from './services/websocket';
import {
  fetchModelStatus,
  fetchModelMetrics,
  fetchSettings,
  updateSettings,
  fetchTranscript,
  mutateTranscript,
} from './services/api';
import { AppSettings, ModelMetrics, ModelStatus, PredictionEvent, TranscriptState } from './types';

export const App: React.FC = () => {
  // Application State
  const [wsStatus, setWsStatus] = useState<ConnectionStatus>('DISCONNECTED');
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [settings, setSettings] = useState<AppSettings>({
    confidence_threshold: 0.65,
    smoothing_window_size: 6,
    stability_duration_ms: 400,
    repeat_lockout_ms: 1200,
    auto_insert_transcript: true,
    roi_mode: 'guide_roi',
    target_fps: 15,
    input_image_size: 64,
  });

  const [prediction, setPrediction] = useState<PredictionEvent | null>(null);
  const [recentSigns, setRecentSigns] = useState<string[]>([]);
  const [transcript, setTranscript] = useState<string>('');
  const [transcriptState, setTranscriptState] = useState<TranscriptState | null>(null);

  // Modals
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isMetricsOpen, setIsMetricsOpen] = useState(false);
  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [isStudioOpen, setIsStudioOpen] = useState(false);

  // Camera hook
  const {
    videoRef,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    isStreaming,
    isPaused,
    isMirrored,
    error: cameraError,
    startCamera,
    stopCamera,
    pauseStream,
    resumeStream,
    toggleMirror,
    captureFrame,
  } = useCamera();

  const wsRef = useRef<SignBridgeWebSocket | null>(null);
  const frameIntervalRef = useRef<number | null>(null);

  // Initialize backend telemetry & settings
  useEffect(() => {
    const initData = async () => {
      try {
        const [mStatus, mMetrics, sData, tData] = await Promise.allSettled([
          fetchModelStatus(),
          fetchModelMetrics(),
          fetchSettings(),
          fetchTranscript(),
        ]);

        if (mStatus.status === 'fulfilled') setModelStatus(mStatus.value);
        if (mMetrics.status === 'fulfilled') setMetrics(mMetrics.value);
        if (sData.status === 'fulfilled') setSettings(sData.value);
        if (tData.status === 'fulfilled') {
          setTranscriptState(tData.value);
          setTranscript(tData.value.text);
        }
      } catch (err) {
        console.warn('Initial data load warning:', err);
      }
    };

    initData();
  }, []);

  // Handle incoming real-time predictions from WebSocket
  const handlePrediction = useCallback((pred: PredictionEvent) => {
    setPrediction(pred);

    if (pred.accepted_char) {
      setRecentSigns((prev) => [...prev.slice(-15), pred.accepted_char!]);
    }

    if (pred.transcript !== undefined && settings.auto_insert_transcript) {
      setTranscript(pred.transcript);
    }
  }, [settings.auto_insert_transcript]);

  // Setup WebSocket connection
  useEffect(() => {
    const ws = new SignBridgeWebSocket(handlePrediction, setWsStatus);
    ws.connect();
    wsRef.current = ws;

    return () => {
      ws.disconnect();
    };
  }, [handlePrediction]);

  // Frame streaming dispatch loop (throttled to settings.target_fps)
  useEffect(() => {
    if (isStreaming && !isPaused) {
      const intervalMs = Math.max(33, Math.floor(1000 / settings.target_fps));

      frameIntervalRef.current = window.setInterval(() => {
        if (!wsRef.current || isPaused) return;

        // Downscale frame for network efficiency (320x240)
        const frameData = captureFrame(320, 240, 0.75);
        if (frameData) {
          wsRef.current.sendFrame(frameData);
        }
      }, intervalMs);
    } else {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    }

    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    };
  }, [isStreaming, isPaused, settings.target_fps, captureFrame]);

  // Save updated settings
  const handleSaveSettings = async (newSettings: Partial<AppSettings>) => {
    const updated = await updateSettings(newSettings);
    setSettings(updated);
    if (wsRef.current) {
      wsRef.current.sendConfig({
        confidence_threshold: updated.confidence_threshold,
        smoothing_window_size: updated.smoothing_window_size,
        stability_duration_ms: updated.stability_duration_ms,
        repeat_lockout_ms: updated.repeat_lockout_ms,
        roi_mode: updated.roi_mode,
      });
    }
  };

  // Transcript mutations
  const handleAppendText = async (char: string) => {
    const res = await mutateTranscript('append', char);
    setTranscriptState(res);
    setTranscript(res.text);
  };

  const handleSpace = async () => {
    const res = await mutateTranscript('space');
    setTranscriptState(res);
    setTranscript(res.text);
  };

  const handleBackspace = async () => {
    const res = await mutateTranscript('backspace');
    setTranscriptState(res);
    setTranscript(res.text);
  };

  const handleClear = async () => {
    const res = await mutateTranscript('clear');
    setTranscriptState(res);
    setTranscript(res.text);
    setRecentSigns([]);
  };

  const handleSetText = async (newText: string) => {
    setTranscript(newText);
    const res = await mutateTranscript('set', newText);
    setTranscriptState(res);
  };

  const handleToggleAutoInsert = () => {
    setSettings((prev) => ({
      ...prev,
      auto_insert_transcript: !prev.auto_insert_transcript,
    }));
  };

  // Snapshot capture handler
  const handleTakeSnapshot = () => {
    const snapshot = captureFrame(640, 480, 0.95);
    if (snapshot) {
      const a = document.createElement('a');
      a.href = snapshot;
      a.download = `signbridge_snapshot_${Date.now()}.jpg`;
      a.click();
    }
  };

  return (
    <div className="min-h-screen bg-background text-slate-100 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-200">
      {/* Top Header */}
      <Header
        wsStatus={wsStatus}
        modelStatus={modelStatus}
        isStreaming={isStreaming}
        fps={prediction?.fps || 0}
        latencyMs={prediction?.latency_ms || 0}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenMetrics={() => setIsMetricsOpen(true)}
        onOpenGuide={() => setIsGuideOpen(true)}
        onOpenStudio={() => setIsStudioOpen(true)}
      />

      {/* Main Dashboard Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 flex flex-col gap-6">
        {/* Top Section: Video Stream & CNN Classification side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Main Video Viewport (7 cols on desktop) */}
          <div className="lg:col-span-7">
            <CameraPanel
              videoRef={videoRef}
              devices={devices}
              selectedDeviceId={selectedDeviceId}
              onSelectDevice={(id) => {
                setSelectedDeviceId(id);
                startCamera(id);
              }}
              isStreaming={isStreaming}
              isPaused={isPaused}
              isMirrored={isMirrored}
              error={cameraError}
              prediction={prediction}
              onStart={() => startCamera()}
              onStop={stopCamera}
              onPause={pauseStream}
              onResume={resumeStream}
              onToggleMirror={toggleMirror}
              onTakeSnapshot={handleTakeSnapshot}
            />
          </div>

          {/* Recognition & Stability Panel (5 cols on desktop) */}
          <div className="lg:col-span-5 h-full">
            <PredictionPanel
              prediction={prediction}
              recentHistory={recentSigns}
            />
          </div>
        </div>

        {/* Bottom Section: Transcript Editor Panel (Full width) */}
        <div>
          <TranscriptPanel
            transcript={transcript}
            transcriptState={transcriptState}
            autoInsert={settings.auto_insert_transcript}
            onToggleAutoInsert={handleToggleAutoInsert}
            onAppendText={handleAppendText}
            onSpace={handleSpace}
            onBackspace={handleBackspace}
            onClear={handleClear}
            onSetText={handleSetText}
          />
        </div>
      </main>

      {/* Modals */}
      <SettingsModal
        isOpen={isSettingsOpen}
        settings={settings}
        onClose={() => setIsSettingsOpen(false)}
        onSave={handleSaveSettings}
      />

      <ModelMetricsModal
        isOpen={isMetricsOpen}
        metrics={metrics}
        modelStatus={modelStatus}
        onClose={() => setIsMetricsOpen(false)}
      />

      <SignsGuideModal
        isOpen={isGuideOpen}
        onClose={() => setIsGuideOpen(false)}
      />

      <DatasetStudioModal
        isOpen={isStudioOpen}
        onClose={() => setIsStudioOpen(false)}
        captureFrame={captureFrame}
        isStreaming={isStreaming}
        onStartCamera={startCamera}
      />
    </div>
  );
};
