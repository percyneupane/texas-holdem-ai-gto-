"""Nonblocking selected-area autoplay for the desktop overlay.

All capture and input calls stay on the UI thread. Only model reading uses
the executor, so Stop or moving the capture box can cancel a pending submit.
"""

import math
import time

from .table_ui import AdaptiveScreenController


class SelectedAreaPlayer:
    def __init__(self, agent, executor, capture, context, mouse, clock=time.monotonic):
        self.agent, self.executor = agent, executor
        self.capture, self.context = capture, context
        self.mouse, self.clock = mouse, clock
        self.controller = None
        self.transaction = None

    @property
    def busy(self):
        return self.transaction is not None

    def cancel(self):
        if self.transaction is not None:
            future = self.transaction.get('future')
            if future is not None:
                future.cancel()
        self.transaction = None

    def resume(self):
        # Preserve the duplicate lock across an OFF/ON cycle, but allow time
        # for a fresh observation to establish whether the last action advanced.
        if self.controller is not None and self.controller.pending is not None:
            self.controller.sent_at = self.clock()

    def waiting(self):
        if self.controller is not None:
            self.controller.waiting()

    def start(self, decision, image, blinds=None):
        if self.busy:
            return False
        context = self.context()
        region = context[0]
        if self.controller is None or self.controller.region != region:
            self.controller = AdaptiveScreenController(region, self.mouse, clock=self.clock)
        current = self.capture()
        if context != self.context():
            raise ValueError('The selection changed before input.')
        prepared = self.controller.prepare(decision, image, current)
        if prepared is None:
            return False
        action, text = prepared
        if text is None:
            self.controller.submit(action)
            return True
        cooldown = self.agent.ai_reader.retry_in
        if not isinstance(cooldown, (int, float)) or not math.isfinite(cooldown):
            raise ValueError('Unknown vision cooldown; stopped before submitting.')
        if cooldown + .05 >= self.controller.ack_timeout:
            raise ValueError('Vision readback cannot finish before the action timeout.')
        self.transaction = dict(context=context, decision=decision, action=action,
                                text=text, blinds=dict(blinds or {}), future=None,
                                not_before=self.clock() + (cooldown + .05 if cooldown > 0 else 0))
        return True

    def poll(self):
        """Advance one UI tick; return True while a typed bet awaits readback."""
        try:
            if self.controller is not None:
                self.controller.check_timeout()
            transaction = self.transaction
            if transaction is None:
                return False
            if transaction['context'] != self.context():
                raise ValueError('The selection changed; the pending bet was cancelled.')
            if self.clock() < transaction['not_before']:
                return True
            if transaction['future'] is None:
                transaction['image'] = self.capture()
                transaction['future'] = self.executor.submit(
                    self.agent.decide, transaction['image'], transaction['blinds'])
                return True
            if not transaction['future'].done():
                return True
            readback = transaction['future'].result()
            if not self.controller.verify_entry(transaction['decision'], transaction['text'], readback):
                raise ValueError('The entered bet could not be verified; stopped before submitting.')
            self.controller.ensure_fresh(transaction['image'], self.capture(), transaction['action'])
            if transaction['context'] != self.context():
                raise ValueError('The selection changed; the pending bet was cancelled.')
            self.controller.submit(transaction['action'])
            self.transaction = None
            return False
        except Exception:
            self.cancel()
            raise
