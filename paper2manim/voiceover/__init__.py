"""Shared voiceover assembly layer.

This module is the single entry point for voiceover assembly used by both
Demo and Generation graphs. It owns the full TTS → align → concat → mux pipeline
so the graph nodes stay thin.
"""

from paper2manim.voiceover.assembly import (
    VoiceoverAssemblyResult,
    assemble_voiceover,
)

__all__ = ["VoiceoverAssemblyResult", "assemble_voiceover"]
