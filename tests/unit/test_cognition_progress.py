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
        # Updated expected calculation based on new weights
        expected = (0.25 * 0.7) + (0.15 * 0.8) + (0.1 * 0.65) + (0.1 * 0.75) + (0.15 * 0.85) + (0.15 * 0) + (0.1 * 0)
        assert progress == pytest.approx(expected * 100)
        assert len(model.cognition_progress_history) == 1

    def test_calculate_cognition_progress_missing_metrics(self, model):
        metrics = {
            'phi': 0.7,
            'coherence': 0.8
            # Missing 'stability', 'adaptability', 'memory_retention', 'emotional_coherence', 'decision_making_efficiency'
        }
        progress = model.calculate_cognition_progress(metrics)
        expected = (0.25 * 0.7) + (0.15 * 0.8) + (0.1 * 0) + (0.1 * 0) + (0.15 * 0) + (0.15 * 0) + (0.1 * 0)
        assert progress == pytest.approx(expected * 100)
        assert len(model.cognition_progress_history) == 1

    def test_calculate_cognition_progress_extreme_values(self, model):
        metrics = {
            'phi': 1.0,
            'coherence': 1.0,
            'stability': 1.0,
            'adaptability': 1.0,
            'memory_retention': 1.0,
            'emotional_coherence': 1.0,
            'decision_making_efficiency': 1.0
        }
        progress = model.calculate_cognition_progress(metrics)
        expected = (0.25 * 1.0) + (0.15 * 1.0) + (0.1 * 1.0) + (0.1 * 1.0) + (0.15 * 1.0) + (0.15 * 1.0) + (0.1 * 1.0)
        assert progress == pytest.approx(expected * 100)
        assert len(model.cognition_progress_history) == 1

    def test_optimize_memory_usage(self, model):
        metrics = {
            'phi': 1.0,
            'coherence': 1.0,
            'stability': 1.0,
            'adaptability': 1.0,
            'memory_retention': 1.0,
            'emotional_coherence': 1.0,
            'decision_making_efficiency': 1.0
        }
        progress = model.calculate_cognition_progress(metrics)
        expected = (0.25 * 1.0) + (0.15 * 1.0) + (0.1 * 1.0) + (0.1 * 1.0) + (0.15 * 1.0) + (0.15 * 1.0) + (0.1 * 1.0)
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
        # Updated expected progress percentage
        assert "Current Cognition Progress: 56.25%" in report
        assert "Areas Needing Improvement:" in report

    def test_report_cognition_progress_with_new_metrics(self, model):
        metrics = {
            'phi': 0.7,
            'coherence': 0.8,
            'stability': 0.65,
            'adaptability': 0.75,
            'memory_retention': 0.85,
            'emotional_coherence': 0.55,  # Below threshold
            'decision_making_efficiency': 0.95
        }
        model.calculate_cognition_progress(metrics)
        report = model.report_cognition_progress()
        # Updated expected progress percentage
        assert "Current Cognition Progress: 74.00%" in report
        assert "Emotional Coherence: 55.00%" in report
        assert "Emotional Coherence" in report  # Area needing improvement

    def test_stress_condition_large_dataset(self, model):
        metrics = {
            'phi': 0.7,
            'coherence': 0.8,
            'stability': 0.65,
            'adaptability': 0.75,
            'memory_retention': 0.85,
            'emotional_coherence': 0.9,
            'decision_making_efficiency': 0.95
        }
        # Simulate large history
        for _ in range(1000):
            model.calculate_cognition_progress(metrics)
        assert len(model.cognition_progress_history) == 1000
        report = model.report_cognition_progress()
        # Updated expected progress percentage
        assert "Current Cognition Progress: 79.25%" in report
        # Populate cognition_progress_history and histories
        metrics = {
            'phi': 0.7,
            'coherence': 0.8,
            'stability': 0.65,
            'adaptability': 0.75,
            'memory_retention': 0.85,
            'emotional_coherence': 0.75,
            'decision_making_efficiency': 0.85
        }
        for _ in range(200):
            model.calculate_cognition_progress(metrics)
            state = torch.randn(1, model.hidden_dim)
            model.state_history.append(state)
            model.context_history.append(state)
        
        model.optimize_memory_usage()
        assert len(model.cognition_progress_history) == 100
        assert len(model.state_history) == 50
        assert len(model.context_history) == 50

    def test_report_cognition_progress_with_target(self, model):
        model.target_cognition_percentage = 80.0
        metrics = {
            'phi': 0.6,
            'coherence': 0.75,
            'stability': 0.65,
            'adaptability': 0.80,
            'memory_retention': 0.70,
            'emotional_coherence': 0.60,  # Below threshold
            'decision_making_efficiency': 0.85
        }
        model.calculate_cognition_progress(metrics)
        report = model.report_cognition_progress()
        assert "Current Cognition Progress: 68.75%" in report
        assert "Target Cognition Progress: 80.00%" in report
        assert "Emotional Coherence: 60.00%" in report
        assert "Areas Needing Improvement:" in report

    def test_report_cognition_progress_no_improvement_needed(self, model):
        model.target_cognition_percentage = 70.0
        metrics = {
            'phi': 0.8,
            'coherence': 0.75,
            'stability': 0.85,
            'adaptability': 0.80,
            'memory_retention': 0.90,
            'emotional_coherence': 0.75,  # Above threshold
            'decision_making_efficiency': 0.85
        }
        model.calculate_cognition_progress(metrics)
        report = model.report_cognition_progress()
        assert "Current Cognition Progress: 81.00%" in report
        assert "Target Cognition Progress: 70.00%" in report
        assert "Areas Needing Improvement:" in report
        # No areas should be listed as needing improvement
        assert not any(metric in report for metric in []), "No metrics should need improvement"