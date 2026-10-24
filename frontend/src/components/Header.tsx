import React from 'react';
import { Activity, Cpu, HelpCircle, Settings as SettingsIcon, BarChart2 } from 'lucide-react';
import { ModelStatus } from '../types';
import { ConnectionStatus } from '../services/websocket';

interface HeaderProps {
  wsStatus: ConnectionStatus;
  modelStatus: ModelStatus | null;
  isStreaming: boolean;
  fps: number;
  latencyMs: number;
  onOpenSettings: () => void;
  onOpenMetrics: () => void;
  onOpenGuide: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  wsStatus,
  modelStatus,
  isStreaming,
  fps,
  latencyMs,
  onOpenSettings,
  onOpenMetrics,
  onOpenGuide,
}) => {
  const isModelReady = modelStatus?.status === 'LOADED';

  return (
    <header className="border-b border-slate-800/80 bg-surface/80 backdrop-blur-md sticky top-0 z-40 px-4 lg:px-8 py-3 transition-colors">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-3">
        {/* Logo and title */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20 ring-1 ring-white/20">
            <span className="text-xl font-black text-white tracking-tighter">SB</span>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                SignBridge
                <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                  MVP v1.0
                </span>
              </h1>
            </div>
            <p className="text-xs text-slate-400">
              Real-Time ASL Gesture Recognition & Transcript Assembly
            </p>
          </div>
        </div>

        {/* Telemetry and Action Controls */}
        <div className="flex items-center flex-wrap gap-2 md:gap-3 text-xs">
          {/* WebSocket Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60">
            <span className={`w-2 h-2 rounded-full ${
              wsStatus === 'CONNECTED' ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]' :
              wsStatus === 'CONNECTING' ? 'bg-amber-400 animate-pulse' : 'bg-rose-500'
            }`} />
            <span className="text-slate-300 font-medium">WS: {wsStatus}</span>
          </div>

          {/* Model Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-300 font-medium">
              {isModelReady ? `CNN (${modelStatus?.num_classes} classes)` : 'Model Not Ready'}
            </span>
          </div>

          {/* FPS & Latency */}
          {isStreaming && (
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-300 font-mono">
                {fps > 0 ? `${fps.toFixed(0)} FPS` : 'Idle'}
              </span>
              <span className="text-slate-500">|</span>
              <span className="text-slate-300 font-mono">
                {latencyMs > 0 ? `${latencyMs.toFixed(0)}ms` : '-'}
              </span>
            </div>
          )}

          {/* Action Modals */}
          <div className="flex items-center gap-1.5 ml-auto md:ml-2">
            <button
              onClick={onOpenGuide}
              className="px-2.5 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 hover:border-slate-600 transition-all flex items-center gap-1.5"
              title="ASL Alphabet Reference Guide"
            >
              <HelpCircle className="w-3.5 h-3.5 text-cyan-400" />
              <span>Sign Guide</span>
            </button>

            <button
              onClick={onOpenMetrics}
              className="px-2.5 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 hover:border-slate-600 transition-all flex items-center gap-1.5"
              title="Model Evaluation & Confusion Matrix"
            >
              <BarChart2 className="w-3.5 h-3.5 text-blue-400" />
              <span>Metrics</span>
            </button>

            <button
              onClick={onOpenSettings}
              className="p-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 hover:border-slate-600 transition-all"
              title="Pipeline Settings"
            >
              <SettingsIcon className="w-4 h-4 text-slate-300" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
