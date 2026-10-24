import React, { useState, useEffect, useRef } from 'react';
import { X, FolderPlus, Play, Check, RefreshCw, Info, Database } from 'lucide-react';

interface DatasetStudioModalProps {
  isOpen: boolean;
  onClose: () => void;
  captureFrame: (w?: number, h?: number, q?: number) => string | null;
  isStreaming: boolean;
  onStartCamera: () => void;
}

const SUPPORTED_SIGNS = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
  'V', 'W', 'X', 'Y'
];

export const DatasetStudioModal: React.FC<DatasetStudioModalProps> = ({
  isOpen,
  onClose,
  captureFrame,
  isStreaming,
  onStartCamera,
}) => {
  const [selectedSign, setSelectedSign] = useState<string>('A');
  const [isRecording, setIsRecording] = useState(false);
  const [recordedCount, setRecordedCount] = useState<Record<string, number>>({});
  const [isTraining, setIsTraining] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const recordingIntervalRef = useRef<number | null>(null);

  // Fetch dataset summary from backend
  const fetchSummary = async () => {
    try {
      const res = await fetch('/api/dataset/summary');
      if (res.ok) {
        const data = await res.json();
        const counts: Record<string, number> = {};
        for (const [cls, val] of Object.entries<any>(data.class_breakdown || {})) {
          counts[cls] = (val.train_samples || 0) + (val.custom_samples || 0);
        }
        setRecordedCount(counts);
      }
    } catch (e) {
      console.warn('Could not fetch dataset summary:', e);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchSummary();
    }
  }, [isOpen]);

  // Capture a single frame for the selected sign
  const handleCaptureSingle = async () => {
    if (!isStreaming) {
      onStartCamera();
      return;
    }
    const frame = captureFrame(160, 160, 0.85);
    if (!frame) return;

    try {
      const res = await fetch('/api/dataset/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label: selectedSign,
          image: frame,
          roi: [0.22, 0.16, 0.56, 0.56],
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setRecordedCount((prev) => ({
          ...prev,
          [selectedSign]: data.class_total || (prev[selectedSign] || 0) + 1,
        }));
      }
    } catch (e) {
      console.error('Failed to capture sample:', e);
    }
  };

  // Toggle burst recording (30 frames at 6 fps)
  const handleToggleBurstRecord = () => {
    if (isRecording) {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
      setIsRecording(false);
      return;
    }

    if (!isStreaming) {
      onStartCamera();
      return;
    }

    setIsRecording(true);
    let captured = 0;
    const maxCaptures = 30;

    recordingIntervalRef.current = window.setInterval(async () => {
      if (captured >= maxCaptures) {
        if (recordingIntervalRef.current) {
          clearInterval(recordingIntervalRef.current);
          recordingIntervalRef.current = null;
        }
        setIsRecording(false);
        setStatusMsg(`Captured ${maxCaptures} real-hand frames for '${selectedSign}'!`);
        setTimeout(() => setStatusMsg(null), 3000);
        return;
      }

      const frame = captureFrame(160, 160, 0.85);
      if (frame) {
        try {
          const res = await fetch('/api/dataset/capture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              label: selectedSign,
              image: frame,
              roi: [0.22, 0.16, 0.56, 0.56],
            }),
          });
          if (res.ok) {
            const data = await res.json();
            setRecordedCount((prev) => ({
              ...prev,
              [selectedSign]: data.class_total || (prev[selectedSign] || 0) + 1,
            }));
          }
        } catch (e) {
          console.error(e);
        }
      }
      captured += 1;
    }, 150);
  };

  // Trigger retraining on real hands
  const handleTrain = async () => {
    setIsTraining(true);
    setStatusMsg('Retraining model on real hand images...');
    try {
      const res = await fetch('/api/dataset/train', { method: 'POST' });
      if (res.ok) {
        setStatusMsg('Model successfully retrained and reloaded!');
        setTimeout(() => setStatusMsg(null), 4000);
      } else {
        setStatusMsg('Retraining failed. Check terminal logs.');
      }
    } catch (e) {
      setStatusMsg('Retraining error.');
    } finally {
      setIsTraining(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-surface border border-slate-800 rounded-2xl w-full max-w-3xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/40">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-emerald-400" />
            <div>
              <h3 className="text-base font-bold text-white">Dataset Studio & Real Hand Collector</h3>
              <p className="text-xs text-slate-400">Capture your own real hand gestures to customize and train the CNN</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 overflow-y-auto space-y-5">
          {/* Status Alert Banner */}
          {statusMsg && (
            <div className="p-3 rounded-xl bg-cyan-950/80 border border-cyan-500/50 text-xs text-cyan-200 flex items-center gap-2">
              <Info className="w-4 h-4 text-cyan-400 shrink-0" />
              <span>{statusMsg}</span>
            </div>
          )}

          {/* Quick instructions */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300 leading-relaxed">
            <span className="font-semibold text-emerald-400">How to Collect Real Hand Data: </span>
            Select a sign below, hold your hand in front of your webcam inside the on-screen box, and click 
            <span className="font-semibold text-white"> &quot;Record 30 Frames&quot;</span> while varying your hand angle slightly. 
            Once you have samples, click <span className="font-semibold text-cyan-400">&quot;Train Model&quot;</span>.
          </div>

          {/* Sign Selection Grid */}
          <div>
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Select Sign to Record or Inspect
            </div>
            <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
              {SUPPORTED_SIGNS.map((sign) => {
                const count = recordedCount[sign] || 0;
                const isSelected = selectedSign === sign;
                return (
                  <button
                    key={sign}
                    onClick={() => setSelectedSign(sign)}
                    className={`p-2 rounded-xl border flex flex-col items-center justify-center transition-all ${
                      isSelected
                        ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300 ring-1 ring-emerald-500'
                        : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
                    }`}
                  >
                    <span className="text-base font-black font-mono">{sign}</span>
                    <span className="text-[10px] text-slate-400 font-mono mt-0.5">{count} imgs</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Active Sign Action Controls */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-emerald-950 border border-emerald-500 flex items-center justify-center text-2xl font-black font-mono text-emerald-300">
                {selectedSign}
              </div>
              <div>
                <div className="text-sm font-bold text-slate-200">Recording Sign &quot;{selectedSign}&quot;</div>
                <div className="text-xs text-slate-400 font-mono">
                  Current samples: {recordedCount[selectedSign] || 0} images
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 w-full sm:w-auto">
              <button
                onClick={handleCaptureSingle}
                className="flex-1 sm:flex-none px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-all flex items-center justify-center gap-1.5"
              >
                <FolderPlus className="w-4 h-4 text-slate-300" />
                <span>Capture Single</span>
              </button>

              <button
                onClick={handleToggleBurstRecord}
                className={`flex-1 sm:flex-none px-4 py-2 rounded-xl text-xs font-semibold transition-all flex items-center justify-center gap-2 shadow-lg ${
                  isRecording
                    ? 'bg-rose-600 hover:bg-rose-500 text-white animate-pulse'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/25'
                }`}
              >
                <Play className="w-4 h-4" />
                <span>{isRecording ? 'Recording (Stop)...' : 'Record 30 Frames'}</span>
              </button>
            </div>
          </div>

          {/* External Dataset Notice */}
          <div className="p-3.5 rounded-xl bg-slate-900/40 border border-slate-800 text-xs text-slate-400 space-y-1.5">
            <div className="font-semibold text-slate-300 flex items-center gap-1.5">
              <Info className="w-4 h-4 text-cyan-400" />
              <span>Providing Your Own External Dataset Folder</span>
            </div>
            <p>
              You can also directly drop any folder containing subfolders of hand images (<code className="text-slate-200">A/</code>, <code className="text-slate-200">B/</code>, etc.) into:
            </p>
            <div className="font-mono text-[11px] bg-slate-950 p-2 rounded border border-slate-800 text-cyan-300 select-all">
              /Users/sdd410/sign language detection/ml/dataset/custom/
            </div>
            <p>
              The system will automatically merge and train on all images placed there.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800 bg-slate-900/30">
          <button
            onClick={fetchSummary}
            className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh Counts</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              Close
            </button>

            <button
              onClick={handleTrain}
              disabled={isTraining}
              className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-cyan-600/20 transition-all flex items-center gap-1.5"
            >
              {isTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
              <span>{isTraining ? 'Training Model...' : 'Train Model Now'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
