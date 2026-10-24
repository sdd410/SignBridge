import React, { useState } from 'react';
import { X, HelpCircle, Search } from 'lucide-react';

interface SignsGuideModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SignDescription {
  sign: string;
  category: string;
  handShape: string;
  description: string;
}

const ASL_GUIDE_DATA: SignDescription[] = [
  { sign: 'A', category: 'Letter', handShape: 'Fist', description: 'Make a fist with your thumb resting straight against the side of your index finger.' },
  { sign: 'B', category: 'Letter', handShape: 'Flat Hand', description: 'Hold four fingers straight up together, with thumb tucked across your palm.' },
  { sign: 'C', category: 'Letter', handShape: 'Curved Hand', description: 'Curve fingers and thumb into a "C" cup shape facing sideways.' },
  { sign: 'D', category: 'Letter', handShape: 'Index Pointer', description: 'Point your index finger straight up; curl middle, ring, pinky to touch your thumb.' },
  { sign: 'E', category: 'Letter', handShape: 'Curled Claw', description: 'Curl all fingertips tightly down towards the palm, with thumb folded underneath.' },
  { sign: 'F', category: 'Letter', handShape: 'OK Sign', description: 'Touch index fingertip to thumb tip in an "OK" circle; extend other 3 fingers.' },
  { sign: 'G', category: 'Letter', handShape: 'Horizontal Pinch', description: 'Point index finger horizontally forward; point thumb parallel to it.' },
  { sign: 'H', category: 'Letter', handShape: 'Two Fingers Horizontal', description: 'Extend index and middle fingers together horizontally; thumb folded.' },
  { sign: 'I', category: 'Letter', handShape: 'Pinky Up', description: 'Make a fist and raise only your little pinky finger straight up.' },
  { sign: 'J', category: 'Letter', handShape: 'Pinky Curve', description: 'Hold the "I" sign and trace a "J" curve in the air with your pinky tip.' },
  { sign: 'K', category: 'Letter', handShape: 'V with Thumb', description: 'Index finger straight up, middle finger angled forward, thumb tucked between them.' },
  { sign: 'L', category: 'Letter', handShape: 'L-Shape', description: 'Extend index finger straight up and thumb straight out sideways at 90 degrees.' },
  { sign: 'M', category: 'Letter', handShape: 'Fist over 3 Fingers', description: 'Fist with thumb tucked under your first three fingers (index, middle, ring).' },
  { sign: 'N', category: 'Letter', handShape: 'Fist over 2 Fingers', description: 'Fist with thumb tucked under your first two fingers (index, middle).' },
  { sign: 'O', category: 'Letter', handShape: 'O-Shape', description: 'Touch all four fingertips to the tip of your thumb, forming an "O".' },
  { sign: 'P', category: 'Letter', handShape: 'K pointing down', description: 'Same finger configuration as "K", but tilted downwards towards the floor.' },
  { sign: 'Q', category: 'Letter', handShape: 'G pointing down', description: 'Same finger configuration as "G", but tilted downwards towards the floor.' },
  { sign: 'R', category: 'Letter', handShape: 'Crossed Fingers', description: 'Cross your middle finger over your index finger (like wishing for luck).' },
  { sign: 'S', category: 'Letter', handShape: 'Fist with Thumb Across', description: 'Make a tight fist with your thumb resting across all fingers.' },
  { sign: 'T', category: 'Letter', handShape: 'Fist over 1 Finger', description: 'Fist with your thumb inserted between your index and middle fingers.' },
  { sign: 'U', category: 'Letter', handShape: 'Two Fingers Together', description: 'Index and middle fingers held straight up pressed together; thumb folded.' },
  { sign: 'V', category: 'Letter', handShape: 'Peace / V Sign', description: 'Index and middle fingers held straight up spread apart in a "V" shape.' },
  { sign: 'W', category: 'Letter', handShape: 'Three Fingers Up', description: 'Index, middle, and ring fingers held straight up spread apart like a "W".' },
  { sign: 'X', category: 'Letter', handShape: 'Bent Hook', description: 'Make a fist with your index finger bent into a hook shape.' },
  { sign: 'Y', category: 'Letter', handShape: 'Shaka / Phone', description: 'Extend only thumb and pinky outwards while holding middle three fingers in.' },
  { sign: 'Z', category: 'Letter', handShape: 'Index Z-Trace', description: 'Hold index finger up and draw a "Z" in the air.' },
  { sign: 'space', category: 'Control', handShape: 'Flat Horizontal', description: 'Hold open flat hand horizontally or gesture space bar action.' },
  { sign: 'del', category: 'Control', handShape: 'Backspace Wave', description: 'Gesture hand backspace / wave to remove the previous character.' },
  { sign: 'nothing', category: 'Neutral', handShape: 'Empty Frame', description: 'Rest hand or step away from the active detection ROI frame.' },
];

export const SignsGuideModal: React.FC<SignsGuideModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [search, setSearch] = useState('');

  if (!isOpen) return null;

  const filtered = ASL_GUIDE_DATA.filter(
    (item) =>
      item.sign.toLowerCase().includes(search.toLowerCase()) ||
      item.handShape.toLowerCase().includes(search.toLowerCase()) ||
      item.description.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-surface border border-slate-800 rounded-2xl w-full max-w-3xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/40">
          <div className="flex items-center gap-2">
            <HelpCircle className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-white">ASL Alphabet Reference Guide</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search */}
        <div className="px-6 py-3 border-b border-slate-800 bg-slate-900/20 flex items-center gap-2">
          <Search className="w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search sign, hand shape, or description..."
            className="w-full bg-transparent text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none"
          />
        </div>

        {/* Cards Grid */}
        <div className="p-6 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
          {filtered.map((item) => (
            <div
              key={item.sign}
              className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700 transition-colors flex flex-col"
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 font-mono font-black text-base flex items-center justify-center">
                  {item.sign}
                </span>
                <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  {item.category}
                </span>
              </div>
              <div className="text-xs font-bold text-slate-200 mb-1">{item.handShape}</div>
              <p className="text-[11px] text-slate-400 leading-relaxed mt-auto">
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
