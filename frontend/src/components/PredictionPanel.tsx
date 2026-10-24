import React from 'react';
import { Award, CheckCircle2, Clock, Zap } from 'lucide-react';
import { PredictionEvent } from '../types';

interface PredictionPanelProps {
  prediction: PredictionEvent | null;
  recentHistory: string[];
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({
  prediction,
  recentHistory,
}) => {
  const currentLabel = prediction?.label || '—';
  const confidence = prediction?.confidence || 0;
  const isStable = prediction?.is_stable || false;
  const isAccepted = prediction?.is_accepted || false;
  const progress = prediction?.stability_progress || 0;
  const topK = prediction?.top_k || [];

  return (
    <div className="bg-surface rounded-2xl border border-slate-800 p-4 flex flex-col shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            CNN Classification & Stability
          </h2>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Latency: {prediction?.latency_ms ? `${prediction.latency_ms.toFixed(0)}ms` : '—'}
        </span>
      </div>

      {/* Main Dominant Sign Display */}
      <div className="flex flex-col items-center justify-center p-6 bg-slate-900/60 rounded-xl border border-slate-800/80 mb-4 relative overflow-hidden">
        {/* Subtle radial glow background */}
        <div className={`absolute inset-0 transition-opacity duration-300 ${
          isAccepted ? 'bg-emerald-500/10' : isStable ? 'bg-cyan-500/10' : 'bg-transparent'
        }`} />

        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
          Active Recognized Sign
        </span>

        {/* Large Sign Character */}
        <div className={`w-24 h-24 rounded-2xl flex items-center justify-center text-5xl font-black font-mono transition-all duration-200 shadow-xl border ${
          isAccepted ? 'bg-emerald-950/80 border-emerald-500 text-emerald-300 scale-105 shadow-emerald-500/20' :
          isStable ? 'bg-cyan-950/80 border-cyan-500 text-cyan-300 scale-100 shadow-cyan-500/20' :
          'bg-slate-800/80 border-slate-700 text-slate-100'
        }`}>
          {currentLabel}
        </div>

        {/* Confidence Percentage */}
        <div className="mt-3 flex items-center gap-2">
          <div className="flex items-center gap-1 text-xs font-medium">
            <span className="text-slate-400">Confidence:</span>
            <span className={`font-mono font-bold ${
              confidence >= 0.7 ? 'text-emerald-400' : confidence >= 0.5 ? 'text-cyan-400' : 'text-amber-400'
            }`}>
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
          {isStable && (
            <span className="inline-flex items-center gap-1 text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              <CheckCircle2 className="w-3 h-3" />
              Stable
            </span>
          )}
        </div>

        {/* Temporal Stability Progress Bar */}
        <div className="w-full max-w-xs mt-3">
          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
            <span>Temporal Stability</span>
            <span>{(progress * 100).toFixed(0)}%</span>
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-100 rounded-full ${
                isAccepted ? 'bg-emerald-400' : isStable ? 'bg-cyan-400' : 'bg-blue-500'
              }`}
              style={{ width: `${Math.min(100, Math.max(0, progress * 100))}%` }}
            />
          </div>
        </div>
      </div>

      {/* Top 5 Probability Distribution */}
      <div className="mb-4">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
          <Award className="w-3.5 h-3.5 text-cyan-400" />
          <span>Top Predictions</span>
        </h3>
        <div className="space-y-1.5">
          {topK.length > 0 ? (
            topK.slice(0, 5).map((item, idx) => (
              <div key={item.label} className="flex items-center gap-2 text-xs">
                <span className="w-4 text-slate-500 font-mono text-[10px]">{idx + 1}.</span>
                <span className="w-12 font-mono font-bold text-slate-200 truncate">{item.label}</span>
                <div className="flex-1 h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-150 ${
                      idx === 0 ? (isStable ? 'bg-cyan-400' : 'bg-blue-500') : 'bg-slate-600'
                    }`}
                    style={{ width: `${item.confidence * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-[11px] text-slate-400">
                  {(item.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))
          ) : (
            <div className="text-xs text-slate-500 py-3 text-center">No active inference</div>
          )}
        </div>
      </div>

      {/* Recent Predictions Strip */}
      <div className="pt-3 border-t border-slate-800/80 mt-auto">
        <div className="flex items-center justify-between text-[11px] text-slate-400 mb-2">
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3 text-cyan-400" />
            <span>Recent Accepted Signs</span>
          </span>
          <span className="text-[10px] text-slate-500">Last 10</span>
        </div>
        <div className="flex items-center gap-1.5 overflow-x-auto py-1">
          {recentHistory.length > 0 ? (
            recentHistory.slice(-10).map((char, index) => (
              <div
                key={index}
                className="min-w-[28px] h-7 rounded-md bg-slate-900 border border-slate-700/80 flex items-center justify-center font-mono font-bold text-xs text-cyan-300 shadow-sm"
              >
                {char === ' ' ? '␣' : char}
              </div>
            ))
          ) : (
            <span className="text-xs text-slate-600 italic">Pending accepted gestures...</span>
          )}
        </div>
      </div>
    </div>
  );
};
