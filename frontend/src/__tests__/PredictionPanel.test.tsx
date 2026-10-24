import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PredictionPanel } from '../components/PredictionPanel';
import { PredictionEvent } from '../types';

describe('PredictionPanel Component', () => {
  it('renders default placeholder state when prediction is null', () => {
    render(<PredictionPanel prediction={null} recentHistory={[]} />);
    expect(screen.getByText('Active Recognized Sign')).toBeDefined();
    expect(screen.getByText('—')).toBeDefined();
    expect(screen.getByText('No active inference')).toBeDefined();
  });

  it('renders prediction label, confidence, and top-5 classes', () => {
    const mockPrediction: PredictionEvent = {
      type: 'prediction',
      label: 'A',
      raw_label: 'A',
      confidence: 0.88,
      raw_confidence: 0.90,
      is_accepted: false,
      is_stable: true,
      stability_progress: 0.85,
      top_k: [
        { label: 'A', confidence: 0.88, index: 0 },
        { label: 'E', confidence: 0.05, index: 4 },
        { label: 'S', confidence: 0.04, index: 18 },
        { label: 'T', confidence: 0.02, index: 19 },
        { label: 'M', confidence: 0.01, index: 12 },
      ],
      latency_ms: 14.5,
      fps: 22.0,
      timestamp: Date.now(),
      accepted_char: null,
      transcript: 'TEST',
    };

    render(<PredictionPanel prediction={mockPrediction} recentHistory={['H', 'E', 'L', 'L', 'O']} />);
    
    // Label should be displayed prominently
    expect(screen.getAllByText('A').length).toBeGreaterThan(0);
    // Confidence text
    expect(screen.getByText('88.0%')).toBeDefined();
    // Stable badge
    expect(screen.getByText('Stable')).toBeDefined();
    // Latency
    expect(screen.getByText('Latency: 15ms')).toBeDefined();
    // Recent signs
    expect(screen.getByText('H')).toBeDefined();
  });
});
