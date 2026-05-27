"""memoscope.core — hook engine and mock model library."""
from memoscope.core.hooks import MemoryInspector, StepSnapshot
from memoscope.core.mock_models import MockTransformer, MockMamba, MockRNN, get_mock_model

__all__ = ["MemoryInspector", "StepSnapshot", "MockTransformer", "MockMamba", "MockRNN", "get_mock_model"]
