import React, { useRef, useEffect } from 'react';
import { Camera, CameraOff, FlipHorizontal, Pause, Play, Camera as SnapshotIcon, AlertTriangle } from 'lucide-react';
import { CameraDevice, PredictionEvent } from '../types';

interface CameraPanelProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  devices: CameraDevice[];
  selectedDeviceId: string;
  onSelectDevice: (deviceId: string) => void;
  isStreaming: boolean;
  isPaused: boolean;
  isMirrored: boolean;
  error: string | null;
  prediction: PredictionEvent | null;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  onToggleMirror: () => void;
  onTakeSnapshot: () => void;
}

export const CameraPanel: React.FC<CameraPanelProps> = ({
  videoRef,
  devices,
  selectedDeviceId,
  onSelectDevice,
  isStreaming,
  isPaused,
  isMirrored,
  error,
  prediction,
  onStart,
  onStop,
  onPause,
  onResume,
  onToggleMirror,
  onTakeSnapshot,
}) => {
  const canvasOverlayRef = useRef<HTMLCanvasElement | null>(null);

  // Render on-screen ROI guide and bounding box
  useEffect(() => {
    const canvas = canvasOverlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;

    const renderOverlay = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw ROI bounding box if available from backend prediction
        if (prediction?.roi_bbox && isStreaming && !isPaused) {
          const [rx, ry, rw, rh] = prediction.roi_bbox;
          
          // Calculate display coordinates (taking mirror mode into account)
          const displayX = isMirrored ? canvas.width - (rx + rw) : rx;

          ctx.save();
          // Glow effect
          ctx.strokeStyle = prediction.is_accepted ? '#10b981' : prediction.is_stable ? '#06b6d4' : '#64748b';
          ctx.lineWidth = 2.5;
          ctx.setLineDash([8, 6]);
          ctx.strokeRect(displayX, ry, rw, rh);

          // Corner brackets
          const cornerLen = 16;
          ctx.setLineDash([]);
          ctx.lineWidth = 4;
          ctx.strokeStyle = prediction.is_accepted ? '#10b981' : '#06b6d4';

          // Top-left
          ctx.beginPath();
          ctx.moveTo(displayX, ry + cornerLen);
          ctx.lineTo(displayX, ry);
          ctx.lineTo(displayX + cornerLen, ry);
          ctx.stroke();

          // Top-right
          ctx.beginPath();
          ctx.moveTo(displayX + rw - cornerLen, ry);
          ctx.lineTo(displayX + rw, ry);
          ctx.lineTo(displayX + rw, ry + cornerLen);
          ctx.stroke();

          // Bottom-left
          ctx.beginPath();
          ctx.moveTo(displayX, ry + rh - cornerLen);
          ctx.lineTo(displayX, ry + rh);
          ctx.lineTo(displayX + cornerLen, ry + rh);
          ctx.stroke();

          // Bottom-right
          ctx.beginPath();
          ctx.moveTo(displayX + rw - cornerLen, ry + rh);
          ctx.lineTo(displayX + rw, ry + rh);
          ctx.lineTo(displayX + rw, ry + rh - cornerLen);
          ctx.stroke();

          // Label above box
          ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
          ctx.fillRect(displayX, Math.max(0, ry - 24), 140, 22);
          ctx.fillStyle = '#38bdf8';
          ctx.font = 'bold 11px system-ui, sans-serif';
          ctx.fillText(`Hand ROI • ${(prediction.confidence * 100).toFixed(0)}%`, displayX + 6, Math.max(14, ry - 8));

          ctx.restore();
        } else if (isStreaming && !isPaused) {
          // Default standby guide box
          const boxSize = Math.floor(Math.min(canvas.width, canvas.height) * 0.58);
          const gx = Math.floor((canvas.width - boxSize) / 2);
          const gy = Math.max(10, Math.floor(canvas.height * 0.18));

          ctx.save();
          ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 6]);
          ctx.strokeRect(gx, gy, boxSize, boxSize);

          ctx.fillStyle = 'rgba(6, 182, 212, 0.6)';
          ctx.font = '11px system-ui, sans-serif';
          ctx.fillText('Place hand in frame', gx + 10, gy + 20);
          ctx.restore();
        }
      }

      animId = requestAnimationFrame(renderOverlay);
    };

    animId = requestAnimationFrame(renderOverlay);
    return () => cancelAnimationFrame(animId);
  }, [prediction, isStreaming, isPaused, isMirrored, videoRef]);

  return (
    <div className="bg-surface rounded-2xl border border-slate-800 p-4 flex flex-col shadow-xl">
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Live Video Stream</h2>
        </div>

        {/* Device selector */}
        {devices.length > 0 && (
          <select
            value={selectedDeviceId}
            onChange={(e) => onSelectDevice(e.target.value)}
            disabled={!isStreaming}
            className="text-xs bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-2.5 py-1 focus:ring-1 focus:ring-cyan-500 outline-none max-w-[180px] truncate"
          >
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Main Viewport Container */}
      <div className="relative w-full aspect-[4/3] bg-black/60 rounded-xl overflow-hidden border border-slate-800/80 flex items-center justify-center group">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`w-full h-full object-cover transition-transform ${isMirrored ? '-scale-x-100' : ''} ${!isStreaming ? 'hidden' : ''}`}
        />

        <canvas
          ref={canvasOverlayRef}
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        />

        {/* Active Prediction Overlay Banner */}
        {isStreaming && prediction && prediction.label && (
          <div className="absolute top-3 left-3 z-10 flex items-center gap-2">
            <div className={`px-3 py-1.5 rounded-lg backdrop-blur-md border shadow-lg flex items-center gap-2 transition-all ${
              prediction.is_accepted ? 'bg-emerald-950/80 border-emerald-500/80 text-emerald-300' :
              prediction.is_stable ? 'bg-cyan-950/80 border-cyan-500/80 text-cyan-300' :
              'bg-slate-900/80 border-slate-700/80 text-slate-300'
            }`}>
              <span className="text-base font-black tracking-wider uppercase font-mono">
                {prediction.label}
              </span>
              <span className="text-xs font-semibold opacity-90">
                {(prediction.confidence * 100).toFixed(0)}%
              </span>
              {prediction.is_accepted && (
                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                  Accepted
                </span>
              )}
            </div>
          </div>
        )}

        {/* Camera Inactive Placeholder */}
        {!isStreaming && (
          <div className="flex flex-col items-center justify-center p-6 text-center text-slate-400">
            <div className="w-16 h-16 rounded-full bg-slate-800/80 border border-slate-700/50 flex items-center justify-center mb-3">
              <CameraOff className="w-8 h-8 text-slate-500" />
            </div>
            <p className="text-sm font-medium text-slate-300 mb-1">Camera Feed Stopped</p>
            <p className="text-xs text-slate-500 max-w-xs mb-4">
              Click &quot;Start Camera&quot; to begin capturing hand gestures in real time.
            </p>
            <button
              onClick={onStart}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-medium text-xs shadow-lg shadow-cyan-500/25 transition-all flex items-center gap-2"
            >
              <Camera className="w-4 h-4" />
              <span>Start Camera</span>
            </button>
          </div>
        )}

        {/* Error overlay */}
        {error && (
          <div className="absolute inset-0 bg-slate-950/90 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center">
            <AlertTriangle className="w-10 h-10 text-rose-400 mb-2" />
            <p className="text-sm font-semibold text-rose-200 mb-1">Camera Access Error</p>
            <p className="text-xs text-rose-300/80 max-w-sm mb-4">{error}</p>
            <button
              onClick={onStart}
              className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-medium transition-all"
            >
              Retry Camera
            </button>
          </div>
        )}
      </div>

      {/* Camera Controls Bar */}
      <div className="flex items-center justify-between gap-2 mt-4 pt-3 border-t border-slate-800/80">
        <div className="flex items-center gap-2">
          {isStreaming ? (
            <button
              onClick={onStop}
              className="px-3 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs font-medium transition-all flex items-center gap-1.5"
            >
              <CameraOff className="w-3.5 h-3.5" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              onClick={onStart}
              className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium transition-all flex items-center gap-1.5 shadow-md shadow-cyan-600/20"
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Start</span>
            </button>
          )}

          {isStreaming && (
            <button
              onClick={isPaused ? onResume : onPause}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition-all flex items-center gap-1.5"
            >
              {isPaused ? <Play className="w-3.5 h-3.5 text-emerald-400" /> : <Pause className="w-3.5 h-3.5 text-amber-400" />}
              <span>{isPaused ? 'Resume' : 'Pause'}</span>
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          {isStreaming && (
            <button
              onClick={onTakeSnapshot}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-all"
              title="Capture Snapshot"
            >
              <SnapshotIcon className="w-4 h-4 text-cyan-400" />
            </button>
          )}

          <button
            onClick={onToggleMirror}
            className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-all flex items-center gap-1.5 ${
              isMirrored
                ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-400'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
            title="Mirror video preview for natural interaction"
          >
            <FlipHorizontal className="w-3.5 h-3.5" />
            <span>Mirror</span>
          </button>
        </div>
      </div>
    </div>
  );
};
