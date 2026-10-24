import React, { useState } from 'react';
import { X, BarChart2, Layers } from 'lucide-react';
import { ModelMetrics, ModelStatus } from '../types';

interface ModelMetricsModalProps {
  isOpen: boolean;
  metrics: ModelMetrics | null;
  modelStatus: ModelStatus | null;
  onClose: () => void;
}

export const ModelMetricsModal: React.FC<ModelMetricsModalProps> = ({
  isOpen,
  metrics,
  modelStatus,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'confusion' | 'curves'>('overview');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-surface border border-slate-800 rounded-2xl w-full max-w-3xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/40">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-blue-400" />
            <h3 className="text-base font-bold text-white">Model Architecture & Evaluation Metrics</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-2 px-6 pt-3 border-b border-slate-800/80 bg-slate-900/20 text-xs">
          <button
            onClick={() => setActiveTab('overview')}
            className={`pb-2.5 px-3 font-semibold border-b-2 transition-all ${
              activeTab === 'overview'
                ? 'border-cyan-400 text-cyan-300'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Performance Overview
          </button>
          <button
            onClick={() => setActiveTab('confusion')}
            className={`pb-2.5 px-3 font-semibold border-b-2 transition-all ${
              activeTab === 'confusion'
                ? 'border-cyan-400 text-cyan-300'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Confusion Matrix
          </button>
          <button
            onClick={() => setActiveTab('curves')}
            className={`pb-2.5 px-3 font-semibold border-b-2 transition-all ${
              activeTab === 'curves'
                ? 'border-cyan-400 text-cyan-300'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Training Curves
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-5">
          {activeTab === 'overview' && (
            <>
              {/* Stat cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400 uppercase font-semibold mb-1">Top-1 Accuracy</div>
                  <div className="text-2xl font-bold font-mono text-emerald-400">
                    {metrics ? `${(metrics.top1_accuracy * 100).toFixed(1)}%` : '—'}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Exact match test split</div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400 uppercase font-semibold mb-1">Top-3 Accuracy</div>
                  <div className="text-2xl font-bold font-mono text-cyan-400">
                    {metrics ? `${(metrics.top3_accuracy * 100).toFixed(1)}%` : '—'}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Ground truth in top 3</div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400 uppercase font-semibold mb-1">Macro F1 Score</div>
                  <div className="text-2xl font-bold font-mono text-blue-400">
                    {metrics ? metrics.macro_avg_f1.toFixed(3) : '—'}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Balanced class harmonic</div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400 uppercase font-semibold mb-1">Parameters</div>
                  <div className="text-2xl font-bold font-mono text-indigo-400">
                    {modelStatus?.parameter_count ? `${(modelStatus.parameter_count / 1e6).toFixed(2)}M` : '1.48M'}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Lightweight ConvNet</div>
                </div>
              </div>

              {/* Architecture Details */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-2">
                <div className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <span>Model Architecture: SignCNN</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  4-stage deep feature extractor featuring <span className="text-slate-200">Conv2d + BatchNorm2d + ReLU + MaxPool2d</span> with dropout regularization, adaptive pooling, and a multi-layer classification head. Trained on Apple Silicon with PyTorch MPS acceleration.
                </p>
                <div className="flex flex-wrap gap-2 pt-2">
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">Input: 3×64×64 RGB</span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">Classes: {modelStatus?.num_classes || 29} (A–Z, del, space, nothing)</span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">Chirality-Preserved Augmentations</span>
                </div>
              </div>

              {/* Note on educational MVP and limitations */}
              <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300/90 leading-relaxed">
                <span className="font-bold text-amber-200">Scope Notice: </span>
                SignBridge is an educational computer vision MVP recognizing individual static ASL alphabet gestures and assembling them into text. It is not an end-to-end continuous full-sentence sign language translator.
              </div>
            </>
          )}

          {activeTab === 'confusion' && (
            <div className="flex flex-col items-center justify-center space-y-3">
              <div className="text-xs text-slate-400 text-center max-w-md">
                Generated from actual predictions across the 261-sample test split. Diagonal represents correct gesture classifications.
              </div>
              <div className="w-full bg-slate-900/80 rounded-xl p-2 border border-slate-800 flex items-center justify-center overflow-hidden">
                <img
                  src="/static/plots/confusion_matrix.png"
                  alt="Confusion Matrix"
                  className="max-h-[420px] object-contain rounded-lg shadow-lg"
                  onError={(e) => {
                    (e.target as HTMLElement).style.display = 'none';
                  }}
                />
              </div>
            </div>
          )}

          {activeTab === 'curves' && (
            <div className="flex flex-col items-center justify-center space-y-3">
              <div className="text-xs text-slate-400 text-center max-w-md">
                Empirical training and validation loss & accuracy progression over epochs with Early Stopping.
              </div>
              <div className="w-full bg-slate-900/80 rounded-xl p-2 border border-slate-800 flex items-center justify-center overflow-hidden">
                <img
                  src="/static/plots/training_curves.png"
                  alt="Training and Validation Curves"
                  className="max-h-[400px] object-contain rounded-lg shadow-lg"
                  onError={(e) => {
                    (e.target as HTMLElement).style.display = 'none';
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
