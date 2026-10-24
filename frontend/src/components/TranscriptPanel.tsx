import React, { useState } from 'react';
import { 
  FileText, Copy, Download, Trash2, Volume2, Check, 
  Space, Delete, PauseCircle, PlayCircle 
} from 'lucide-react';
import { TranscriptState } from '../types';

interface TranscriptPanelProps {
  transcript: string;
  transcriptState: TranscriptState | null;
  autoInsert: boolean;
  onToggleAutoInsert: () => void;
  onAppendText: (char: string) => void;
  onSpace: () => void;
  onBackspace: () => void;
  onClear: () => void;
  onSetText: (newText: string) => void;
}

export const TranscriptPanel: React.FC<TranscriptPanelProps> = ({
  transcript,
  transcriptState,
  autoInsert,
  onToggleAutoInsert,
  onSpace,
  onBackspace,
  onClear,
  onSetText,
}) => {
  const [copied, setCopied] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Copy transcript to clipboard
  const handleCopy = async () => {
    if (!transcript) return;
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Failed to copy text:', e);
    }
  };

  // Download transcript as .txt file
  const handleDownload = () => {
    if (!transcript) return;
    const blob = new Blob([transcript], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `signbridge_transcript_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Speech synthesis of assembled text (explicitly labeled as speech synthesis of text, not full sign translation)
  const handleSpeak = () => {
    if (!transcript.trim()) return;
    if (!('speechSynthesis' in window)) {
      alert('Speech synthesis is not supported in this browser.');
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(transcript);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  const wordCount = transcriptState?.word_count ?? (transcript.trim() ? transcript.trim().split(/\s+/).length : 0);
  const charCount = transcript.length;

  return (
    <div className="bg-surface rounded-2xl border border-slate-800 p-4 flex flex-col shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Live Text Transcript
          </h2>
        </div>

        {/* Counts & Auto-Insert Indicator */}
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleAutoInsert}
            className={`px-2.5 py-1 rounded-md text-xs font-medium border transition-all flex items-center gap-1.5 ${
              autoInsert
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-slate-800 text-slate-400 border-slate-700'
            }`}
            title={autoInsert ? 'Pause automatic sign insertion' : 'Resume automatic sign insertion'}
          >
            {autoInsert ? <PlayCircle className="w-3.5 h-3.5 text-emerald-400" /> : <PauseCircle className="w-3.5 h-3.5 text-amber-400" />}
            <span>Auto: {autoInsert ? 'ON' : 'OFF'}</span>
          </button>

          <span className="text-[11px] font-mono text-slate-400 bg-slate-900 px-2 py-1 rounded border border-slate-800">
            {wordCount} words • {charCount} chars
          </span>
        </div>
      </div>

      {/* Editable Live Textarea */}
      <div className="relative mb-3 flex-1 min-h-[140px]">
        <textarea
          value={transcript}
          onChange={(e) => onSetText(e.target.value)}
          placeholder="Recognized characters and words will assemble here in real time... You can also type or edit directly."
          className="w-full h-full min-h-[140px] bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 font-sans text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-cyan-500/60 resize-none transition-all"
        />
      </div>

      {/* Manual Action Controls Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800/80">
        {/* Left: Editing shortcuts */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={onSpace}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all flex items-center gap-1.5 active:scale-95"
            title="Insert space"
          >
            <Space className="w-3.5 h-3.5 text-cyan-400" />
            <span>Space</span>
          </button>

          <button
            onClick={onBackspace}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all flex items-center gap-1.5 active:scale-95"
            title="Delete previous character"
          >
            <Delete className="w-3.5 h-3.5 text-amber-400" />
            <span>Backspace</span>
          </button>

          <button
            onClick={onClear}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-rose-950/40 text-slate-400 hover:text-rose-300 text-xs font-medium border border-slate-700 hover:border-rose-800/50 transition-all flex items-center gap-1.5"
            title="Clear entire transcript"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear</span>
          </button>
        </div>

        {/* Right: Export & TTS actions */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={handleSpeak}
            disabled={!transcript.trim() || isSpeaking}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 text-xs font-medium border border-slate-700 transition-all flex items-center gap-1.5"
            title="Speech synthesis of assembled text"
          >
            <Volume2 className={`w-3.5 h-3.5 ${isSpeaking ? 'text-emerald-400 animate-pulse' : 'text-blue-400'}`} />
            <span>{isSpeaking ? 'Speaking...' : 'Speak Text'}</span>
          </button>

          <button
            onClick={handleCopy}
            disabled={!transcript}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 text-xs font-medium border border-slate-700 transition-all flex items-center gap-1.5"
            title="Copy to clipboard"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-300" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>

          <button
            onClick={handleDownload}
            disabled={!transcript}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 text-xs font-medium border border-slate-700 transition-all flex items-center gap-1.5"
            title="Download transcript as .txt"
          >
            <Download className="w-3.5 h-3.5 text-slate-300" />
            <span>Export</span>
          </button>
        </div>
      </div>
    </div>
  );
};
