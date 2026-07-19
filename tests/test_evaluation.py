"""
Unit tests for Evaluation metrics
"""

import pytest
import numpy as np
import json
import tempfile

try:
    from src.evaluation import EvaluationMetrics
except ImportError:
    from evaluation import EvaluationMetrics


class TestEvaluationMetrics:
    """Test suite for EvaluationMetrics"""
    
    def test_accuracy_metrics(self):
        """Test accuracy computation"""
        evaluator = EvaluationMetrics()
        
        y_true = ["A", "B", "A", "C", "B"]
        y_pred = ["A", "B", "A", "B", "B"]
        y_pred_alternatives = [
            ["A", "B", "C"],
            ["B", "A", "C"],
            ["A", "C", "B"],
            ["C", "A", "B"],  # Wrong top-1, but C is in alternatives
            ["B", "C", "A"]
        ]
        
        metrics = evaluator.compute_accuracy_metrics(y_true, y_pred, y_pred_alternatives)
        
        assert "top1_accuracy" in metrics
        assert "top3_accuracy" in metrics
        assert metrics["top1_accuracy"] == 0.8  # 4 correct out of 5
        assert metrics["top3_accuracy"] == 1.0  # All correct in top-3
    
    def test_confidence_metrics(self):
        """Test confidence calibration"""
        evaluator = EvaluationMetrics()
        
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        correct = [True, True, False, False, False]
        
        metrics = evaluator.compute_confidence_metrics(confidences, correct)
        
        assert "expected_calibration_error" in metrics
        assert "average_confidence" in metrics
        assert metrics["average_confidence"] == 0.7
    
    def test_description_similarity(self):
        """Test description similarity computation"""
        evaluator = EvaluationMetrics()
        
        reference = ["A great product", "Very good quality"]
        generated = ["A great product", "excellent quality"]
        
        metrics = evaluator.compute_description_similarity(reference, generated)
        
        assert "average_similarity" in metrics
        assert metrics["average_similarity"] >= 0
        assert metrics["average_similarity"] <= 1
    
    def test_processing_time_metrics(self):
        """Test processing time statistics"""
        evaluator = EvaluationMetrics()
        
        times = [1.0, 1.5, 2.0, 1.2, 1.8]
        metrics = evaluator.compute_processing_time_metrics(times)
        
        assert "average_time_seconds" in metrics
        assert "median_time_seconds" in metrics
        assert "95th_percentile_seconds" in metrics
        assert metrics["average_time_seconds"] == 1.5
