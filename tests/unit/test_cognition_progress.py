
import pytest
import torch
from models.consciousness_model import ConsciousnessModel

class TestCognitionProgress:
    @pytest.fixture
    def model(self):
        config = ConsciousnessModel.create_default_config()
        return ConsciousnessModel(**config)

    def test_cognition_progress_initial(self, model):
        assert len(model.cognition_progress_history) == 0

    def test_calculate_cognition_progress(self, model):
        metrics = {
            'phi': 0.7,
            'coherence': 0.8,
            'stability': 0.65,
            'adaptability': 0.75,
            'memory_retention': 0.85
        }
        progress = model.calculate_cognition_progress(metrics)
        expected = (0.3 * 0.7) + (0.2 * 0.8) + (0.15 * 0.65) + (0.15 * 0.75) + (0.2 * 0.85)
        assert progress == pytest.approx(expected * 100)
        assert len(model.cognition_progress_history) == 1

    def test_report_cognition_progress_no_data(self, model):
        report = model.report_cognition_progress()
        assert report == "No cognition progress data available."

    def test_report_cognition_progress_with_data(self, model):
        metrics = {
            'phi': 0.7,
            'coherence': 0.8,
            'stability': 0.65,
            'adaptability': 0.75,
            'memory_retention': 0.85
        }
        model.calculate_cognition_progress(metrics)
        report = model.report_cognition_progress()
        assert "Current Cognition Progress: 75.00%" in report
        assert "Areas Needing Improvement:" in report