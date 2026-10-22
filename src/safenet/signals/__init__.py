"""Stage 3: signals dictionary (config/signals.yaml) and evidence-based detector."""

from safenet.signals.detector import Evidence, SignalDetector, SignalHit, detect, fired_ids, get_detector

__all__ = ["Evidence", "SignalDetector", "SignalHit", "detect", "fired_ids", "get_detector"]
