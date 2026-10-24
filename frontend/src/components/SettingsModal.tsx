import React, { useState } from 'react';
import { X, Sliders, RotateCcw, Check } from 'lucide-react';
import { AppSettings } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  settings: AppSettings;
  onClose: () => void;
  onSave: (newSettings: Partial<AppSettings>) => Promise<void>;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  settings,
  onClose,
  onSave,
}) => {
  const [formData, setFormData] = useState<AppSettings>(settings);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  if (!isOpen) return null;

  const handleResetDefaults = () => {
    setFormData({
      confidence_threshold: 0.65,
      smoothing_window_size: 6,
      stability_duration_ms: 400,
      repeat_lockout_ms: 1200,
      auto_insert_transcript: true,
      roi_mode: 'guide_roi',
      target_fps: 15,
      input_image_size: 64,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    try {
      await onSave(formData);
      setSaveSuccess(true);
      setTimeout(() => {
        setSaveSuccess(false);
        onClose();
      }, 600);
    } catch (err) {
      console.error('Failed to save settings:', err);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-surface border border-slate-800 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/40">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-white">Pipeline & Recognition Settings</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5 overflow-y-auto max-h-[75vh]">
          {/* Target FPS */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-slate-300">
                Target Processing FPS: <span className="text-cyan-400 font-mono">{formData.target_fps} fps</span>
              </label>
              <span className="text-[11px] text-slate-500">Limits browser frame dispatch</span>
            </div>
            <input
              type="range"
              min="5"
              max="30"
              step="1"
              value={formData.target_fps}
              onChange={(e) => setFormData({ ...formData, target_fps: parseInt(e.target.value) })}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>

          {/* Confidence Threshold */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-slate-300">
                Confidence Threshold: <span className="text-cyan-400 font-mono">{(formData.confidence_threshold * 100).toFixed(0)}%</span>
              </label>
              <span className="text-[11px] text-slate-500">Minimum score to accept gesture</span>
            </div>
            <input
              type="range"
              min="0.40"
              max="0.95"
              step="0.05"
              value={formData.confidence_threshold}
              onChange={(e) => setFormData({ ...formData, confidence_threshold: parseFloat(e.target.value) })}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>

          {/* Smoothing Window Size */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-slate-300">
                Temporal Smoothing Window: <span className="text-cyan-400 font-mono">{formData.smoothing_window_size} frames</span>
              </label>
              <span className="text-[11px] text-slate-500">Sliding majority voting</span>
            </div>
            <input
              type="range"
              min="1"
              max="15"
              step="1"
              value={formData.smoothing_window_size}
              onChange={(e) => setFormData({ ...formData, smoothing_window_size: parseInt(e.target.value) })}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>

          {/* Stability Duration */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-slate-300">
                Required Stable Duration: <span className="text-cyan-400 font-mono">{formData.stability_duration_ms} ms</span>
              </label>
              <span className="text-[11px] text-slate-500">Hold time before sign acceptance</span>
            </div>
            <input
              type="range"
              min="150"
              max="1500"
              step="50"
              value={formData.stability_duration_ms}
              onChange={(e) => setFormData({ ...formData, stability_duration_ms: parseInt(e.target.value) })}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>

          {/* Repeat Lockout */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold text-slate-300">
                Repeat Lockout Cooldown: <span className="text-cyan-400 font-mono">{formData.repeat_lockout_ms} ms</span>
              </label>
              <span className="text-[11px] text-slate-500">Prevents repeated duplicate letters</span>
            </div>
            <input
              type="range"
              min="400"
              max="3000"
              step="100"
              value={formData.repeat_lockout_ms}
              onChange={(e) => setFormData({ ...formData, repeat_lockout_ms: parseInt(e.target.value) })}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>

          {/* Hand ROI Extraction Mode */}
          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-2">
              Hand ROI Extraction Mode
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: 'guide_roi', name: 'Guide Box', desc: 'Standard framing guide' },
                { id: 'skin_contour', name: 'Skin Contour', desc: 'Adaptive color tracker' },
                { id: 'full_frame', name: 'Full Frame', desc: 'Raw frame passthrough' },
              ].map((m) => (
                <button
                  type="button"
                  key={m.id}
                  onClick={() => setFormData({ ...formData, roi_mode: m.id as any })}
                  className={`p-2.5 rounded-xl border text-left transition-all ${
                    formData.roi_mode === m.id
                      ? 'bg-cyan-500/10 border-cyan-500 text-cyan-300'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  <div className="text-xs font-bold mb-0.5">{m.name}</div>
                  <div className="text-[10px] text-slate-500 leading-tight">{m.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Footer Actions */}
          <div className="flex items-center justify-between pt-4 border-t border-slate-800">
            <button
              type="button"
              onClick={handleResetDefaults}
              className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1.5 px-3 py-1.5 rounded-lg hover:bg-slate-800 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Defaults</span>
            </button>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSaving}
                className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium shadow-lg shadow-cyan-600/20 transition-all flex items-center gap-1.5"
              >
                {saveSuccess ? <Check className="w-4 h-4 text-emerald-300" /> : null}
                <span>{saveSuccess ? 'Saved!' : isSaving ? 'Saving...' : 'Apply Settings'}</span>
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
