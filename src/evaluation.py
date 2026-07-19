"""
Evaluation Module for Product Analyzer
Computes metrics to evaluate model performance
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Compute various evaluation metrics for the product analyzer"""
    
    def __init__(self):
        self.metrics = {}
    
    def compute_accuracy_metrics(self, y_true: List[str], y_pred: List[str], 
                                y_pred_alternatives: List[List[str]]) -> Dict:
        """
        Compute category prediction accuracy metrics
        
        Args:
            y_true: Ground truth categories
            y_pred: Predicted categories (top-1)
            y_pred_alternatives: Top-3 predictions for each sample
        
        Returns:
            Dictionary with accuracy metrics
        """
        # Top-1 Accuracy
        top1_accuracy = accuracy_score(y_true, y_pred)
        
        # Top-3 Accuracy
        top3_accuracy = sum(
            1 for true, alternatives in zip(y_true, y_pred_alternatives)
            if true in alternatives
        ) / len(y_true)
        
        # Confusion Matrix
        unique_categories = list(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=unique_categories)
        
        metrics = {
            "top1_accuracy": float(top1_accuracy),
            "top3_accuracy": float(top3_accuracy),
            "per_category_accuracy": {}
        }
        
        # Per-category accuracy
        for i, category in enumerate(unique_categories):
            category_mask = [t == category for t in y_true]
            if sum(category_mask) > 0:
                cat_accuracy = sum(
                    p == category for t, p in zip(y_true, y_pred) if t == category
                ) / sum(category_mask)
                metrics["per_category_accuracy"][category] = float(cat_accuracy)
        
        return metrics
    
    def compute_confidence_metrics(self, confidences: List[float], 
                                   predictions_correct: List[bool]) -> Dict:
        """
        Compute confidence calibration metrics
        
        Args:
            confidences: Confidence scores from model (0-1)
            predictions_correct: Whether each prediction was correct
        
        Returns:
            Dictionary with confidence metrics
        """
        confidences = np.array(confidences)
        predictions_correct = np.array(predictions_correct)
        
        # Expected Calibration Error (ECE)
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0
        
        for i in range(num_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(predictions_correct[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                ece += np.abs(bin_accuracy - bin_confidence) * np.sum(bin_mask) / len(confidences)
        
        metrics = {
            "expected_calibration_error": float(ece),
            "average_confidence": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences))
        }
        
        return metrics
    
    def compute_description_similarity(self, reference_descriptions: List[str],
                                      generated_descriptions: List[str]) -> Dict:
        """
        Compute semantic similarity between reference and generated descriptions
        Uses simple string overlap metrics (BLEU-like)
        
        Args:
            reference_descriptions: Human-written descriptions
            generated_descriptions: AI-generated descriptions
        
        Returns:
            Dictionary with similarity metrics
        """
        similarities = []
        
        for ref, gen in zip(reference_descriptions, generated_descriptions):
            # Simple word overlap metric
            ref_words = set(ref.lower().split())
            gen_words = set(gen.lower().split())
            
            if len(ref_words) == 0 or len(gen_words) == 0:
                similarity = 0.0
            else:
                intersection = len(ref_words & gen_words)
                union = len(ref_words | gen_words)
                similarity = intersection / union
            
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        metrics = {
            "average_similarity": float(np.mean(similarities)),
            "similarity_std": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "similarity_median": float(np.median(similarities))
        }
        
        return metrics
    
    def compute_processing_time_metrics(self, processing_times: List[float]) -> Dict:
        """
        Compute processing time statistics
        
        Args:
            processing_times: List of processing times in seconds
        
        Returns:
            Dictionary with timing metrics
        """
        times = np.array(processing_times)
        
        metrics = {
            "average_time_seconds": float(np.mean(times)),
            "std_time_seconds": float(np.std(times)),
            "min_time_seconds": float(np.min(times)),
            "max_time_seconds": float(np.max(times)),
            "median_time_seconds": float(np.median(times)),
            "95th_percentile_seconds": float(np.percentile(times, 95))
        }
        
        return metrics
    
    def compare_to_benchmarks(self, metrics: Dict, config: Dict) -> Dict:
        """
        Compare computed metrics to business benchmarks
        
        Args:
            metrics: Computed metrics
            config: Configuration with target metrics
        
        Returns:
            Comparison report
        """
        targets = config.get("business", {})
        benchmarks = config.get("benchmarks", {})
        
        comparison = {
            "target_performance": {},
            "vs_competitors": {}
        }
        
        # Compare to targets
        if "target_accuracy" in targets:
            target_acc = targets["target_accuracy"]
            actual_acc = metrics.get("accuracy_metrics", {}).get("top1_accuracy", 0)
            comparison["target_performance"]["accuracy"] = {
                "target": target_acc,
                "actual": actual_acc,
                "status": "✓ PASS" if actual_acc >= target_acc else "✗ FAIL"
            }
        
        if "target_processing_time" in targets:
            target_time = targets["target_processing_time"]
            actual_time = metrics.get("timing_metrics", {}).get("average_time_seconds", 0)
            comparison["target_performance"]["processing_time"] = {
                "target": target_time,
                "actual": actual_time,
                "status": "✓ PASS" if actual_time <= target_time else "✗ FAIL"
            }
        
        # Compare to competitors
        for company, company_benchmarks in benchmarks.items():
            comparison["vs_competitors"][company] = {
                "accuracy": {
                    "theirs": company_benchmarks.get("accuracy"),
                    "ours": metrics.get("accuracy_metrics", {}).get("top1_accuracy"),
                    "difference": metrics.get("accuracy_metrics", {}).get("top1_accuracy", 0) - company_benchmarks.get("accuracy", 0)
                },
                "processing_time": {
                    "theirs": company_benchmarks.get("processing_time"),
                    "ours": metrics.get("timing_metrics", {}).get("average_time_seconds"),
                    "difference": metrics.get("timing_metrics", {}).get("average_time_seconds", 0) - company_benchmarks.get("processing_time", 0)
                }
            }
        
        return comparison
    
    def generate_report(self, metrics: Dict, comparison: Dict, output_path: str = None) -> str:
        """
        Generate a text report of all metrics
        
        Args:
            metrics: Computed metrics
            comparison: Comparison to benchmarks
            output_path: Path to save report (optional)
        
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("MULTIMODAL PRODUCT ANALYZER - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Accuracy Metrics
        report.append("📊 ACCURACY METRICS")
        report.append("-" * 40)
        acc_metrics = metrics.get("accuracy_metrics", {})
        report.append(f"Top-1 Accuracy:     {acc_metrics.get('top1_accuracy', 0):.2%}")
        report.append(f"Top-3 Accuracy:     {acc_metrics.get('top3_accuracy', 0):.2%}")
        report.append("")
        
        # Confidence Metrics
        report.append("🎯 CONFIDENCE CALIBRATION")
        report.append("-" * 40)
        conf_metrics = metrics.get("confidence_metrics", {})
        report.append(f"Expected Calibration Error: {conf_metrics.get('expected_calibration_error', 0):.4f}")
        report.append(f"Average Confidence:         {conf_metrics.get('average_confidence', 0):.2%}")
        report.append("")
        
        # Description Similarity
        report.append("📝 DESCRIPTION QUALITY")
        report.append("-" * 40)
        sim_metrics = metrics.get("description_similarity", {})
        report.append(f"Average Similarity Score:   {sim_metrics.get('average_similarity', 0):.2%}")
        report.append(f"Median Similarity Score:    {sim_metrics.get('similarity_median', 0):.2%}")
        report.append("")
        
        # Processing Time
        report.append("⏱️  PERFORMANCE")
        report.append("-" * 40)
        timing = metrics.get("timing_metrics", {})
        report.append(f"Average Time:       {timing.get('average_time_seconds', 0):.3f}s")
        report.append(f"95th Percentile:    {timing.get('95th_percentile_seconds', 0):.3f}s")
        report.append("")
        
        # Target Comparison
        report.append("🎯 TARGET PERFORMANCE")
        report.append("-" * 40)
        target_perf = comparison.get("target_performance", {})
        for metric, details in target_perf.items():
            report.append(f"{metric.upper()}:")
            report.append(f"  Target: {details.get('target')}")
            report.append(f"  Actual: {details.get('actual')}")
            report.append(f"  Status: {details.get('status')}")
        report.append("")
        
        # Competitive Comparison
        report.append("🏆 COMPETITIVE COMPARISON")
        report.append("-" * 40)
        competitors = comparison.get("vs_competitors", {})
        for company, metrics_comp in competitors.items():
            report.append(f"{company}:")
            acc_comp = metrics_comp.get("accuracy", {})
            report.append(f"  Accuracy: Ours {acc_comp.get('ours'):.2%} vs {acc_comp.get('theirs'):.2%} (Δ {acc_comp.get('difference'):+.2%})")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text


class EvaluationDataset:
    """Load and manage evaluation dataset"""
    
    @staticmethod
    def load_from_csv(csv_path: str) -> Dict:
        """
        Load evaluation dataset from CSV
        Expected columns: image_path, category, description
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Dictionary with dataset information
        """
        data = {
            "image_paths": [],
            "categories": [],
            "descriptions": []
        }
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["image_paths"].append(row["image_path"])
                data["categories"].append(row["category"])
                data["descriptions"].append(row["description"])
        
        logger.info(f"Loaded {len(data['image_paths'])} samples from {csv_path}")
        return data
    
    @staticmethod
    def save_results(results: List[Dict], output_path: str):
        """Save analysis results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = EvaluationMetrics()
    
    # Example data
    y_true = ["Electronics", "Fashion", "Home & Garden", "Electronics"]
    y_pred = ["Electronics", "Fashion", "Home & Garden", "Home & Garden"]
    y_pred_alternatives = [
        ["Electronics", "Toys", "Books"],
        ["Fashion", "Electronics", "Toys"],
        ["Home & Garden", "Fashion", "Electronics"],
        ["Electronics", "Home & Garden", "Fashion"]
    ]
    
    metrics = evaluator.compute_accuracy_metrics(y_true, y_pred, y_pred_alternatives)
    print(json.dumps(metrics, indent=2))
