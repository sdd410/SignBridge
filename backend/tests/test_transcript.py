"""Tests for Transcript Manager."""

import pytest
from backend.app.transcript.manager import TranscriptManager


def test_transcript_operations():
    tm = TranscriptManager()
    tm.clear()

    # Append chars
    tm.append_char("H")
    tm.append_char("E")
    tm.append_char("L")
    tm.append_char("L")
    tm.append_char("O")
    assert tm.get_state()["text"] == "HELLO"
    assert tm.get_state()["word_count"] == 1

    # Space
    tm.space()
    tm.append_char("W")
    tm.append_char("O")
    tm.append_char("R")
    tm.append_char("L")
    tm.append_char("D")
    assert tm.get_state()["text"] == "HELLO WORLD"
    assert tm.get_state()["word_count"] == 2

    # Backspace
    tm.backspace()
    assert tm.get_state()["text"] == "HELLO WORL"

    # Backspace control character (\b)
    tm.append_char("\b")
    assert tm.get_state()["text"] == "HELLO WOR"

    # Set text
    tm.set_text("SIGN LANGUAGE")
    assert tm.get_state()["text"] == "SIGN LANGUAGE"

    # Clear
    tm.clear()
    assert tm.get_state()["text"] == ""
    assert tm.get_state()["word_count"] == 0
