import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TranscriptPanel } from '../components/TranscriptPanel';

describe('TranscriptPanel Component', () => {
  it('renders transcript text and statistics', () => {
    render(
      <TranscriptPanel
        transcript="HELLO WORLD"
        transcriptState={{
          text: 'HELLO WORLD',
          character_count: 11,
          word_count: 2,
          last_updated: Date.now(),
          history: [],
        }}
        autoInsert={true}
        onToggleAutoInsert={vi.fn()}
        onAppendText={vi.fn()}
        onSpace={vi.fn()}
        onBackspace={vi.fn()}
        onClear={vi.fn()}
        onSetText={vi.fn()}
      />
    );

    expect(screen.getByDisplayValue('HELLO WORLD')).toBeDefined();
    expect(screen.getByText('2 words • 11 chars')).toBeDefined();
    expect(screen.getByText('Auto: ON')).toBeDefined();
  });

  it('triggers editing callbacks for Space, Backspace, and Clear', () => {
    const handleSpace = vi.fn();
    const handleBackspace = vi.fn();
    const handleClear = vi.fn();

    render(
      <TranscriptPanel
        transcript="SAMPLE"
        transcriptState={null}
        autoInsert={false}
        onToggleAutoInsert={vi.fn()}
        onAppendText={vi.fn()}
        onSpace={handleSpace}
        onBackspace={handleBackspace}
        onClear={handleClear}
        onSetText={vi.fn()}
      />
    );

    fireEvent.click(screen.getByTitle('Insert space'));
    expect(handleSpace).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByTitle('Delete previous character'));
    expect(handleBackspace).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByTitle('Clear entire transcript'));
    expect(handleClear).toHaveBeenCalledTimes(1);
  });
});
