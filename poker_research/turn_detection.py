"""Recognize action labels locally, independent of button colour and table scale."""

from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ACTION_LABEL = re.compile(r'^(FOLD|CHECK|CALL|BET|RAISE)(?:\s+(?:TO\s+)?[$€£]?\d[\d.,]*(?:\s*BB)?)?$')


def action_controls(rows):
    """Return a row of live action labels; ignore seat badges and preselection text."""
    labels = []
    for row in rows:
        text = ' '.join(row['text'].upper().split())
        match = ACTION_LABEL.fullmatch(text)
        if match and row.get('confidence', 0) >= 0.35 and row['y'] >= 0.45:
            labels.append((match[1], row))
    for action, fold in labels:
        if action != 'FOLD':
            continue
        cy = fold['y'] + fold['height'] / 2
        aligned = [(name, box) for name, box in labels
                   if abs(box['y'] + box['height'] / 2 - cy) <= max(0.025, fold['height'])]
        if any(name in ('CALL', 'CHECK') for name, _ in aligned):
            return aligned
    return []


class LocalTurnDetector:
    """macOS Vision OCR. Images never leave the machine."""

    def __init__(self):
        self._directory = None
        self._binary = None
        self._initialization_error = None

    def _prepare(self):
        if self._binary is not None:
            return
        if self._initialization_error is not None:
            raise RuntimeError(self._initialization_error)
        if sys.platform != 'darwin':
            raise RuntimeError('Local action-text detection requires macOS. Use Read now for a single read.')
        self._directory = tempfile.TemporaryDirectory(prefix='poker-turn-')
        binary = Path(self._directory.name) / 'turn-text'
        try:
            subprocess.run(['/usr/bin/swiftc', '-module-cache-path', self._directory.name,
                            str(Path(__file__).with_name('turn_text.swift')), '-o', str(binary)],
                           check=True, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as error:
            self._initialization_error = 'Cannot initialize local action-text detection. Use Read now for a single read.'
            raise RuntimeError(self._initialization_error) from error
        self._binary = binary

    def inspect(self, image):
        self._prepare()
        encoded = BytesIO()
        image.save(encoded, format='PNG')
        try:
            result = subprocess.run([str(self._binary)], input=encoded.getvalue(),
                                    capture_output=True, check=True, timeout=5)
            controls = action_controls(json.loads(result.stdout))
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError('Local action-text detection failed. Use Read now for a single read.') from error
        # Label text (including call size) defines the decision; clocks do not.
        signature = tuple(text.encode('utf-8') for text in sorted(
            ' '.join(box['text'].upper().split()) for _, box in controls))
        return bool(controls), signature
