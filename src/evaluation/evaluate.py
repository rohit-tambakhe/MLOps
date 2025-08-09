"""Model evaluation script for DVC pipeline."""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torchvision import datasets, transforms

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.onnx_predictor import ONNXPredictor
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def evaluate_onnx_model(model_path: str, data_dir: str = "./data"):
    """Evaluate ONNX model on test dataset."""
    logger.info(f"Evaluating ONNX model: {model_path}")
    
    # Initialize predictor
    predictor = ONNXPredictor(model_path)
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )
    
    # Get predictions
    logger.info("Running inference on test dataset...")
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    for i, (image, label) in enumerate(test_dataset):
        if i % 1000 == 0:
            logger.info(f"Processed {i}/{len(test_dataset)} samples")
        
        # Convert tensor to PIL Image for predictor
        image_pil = transforms.ToPILImage()(image)
        result = predictor.predict(image_pil)
        
        predicted_class = result["predicted_class"]
        predicted_idx = predictor.classes.index(predicted_class)
        
        all_predictions.append(predicted_idx)
        all_labels.append(label)
        all_probabilities.append(list(result["class_probabilities"].values()))
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Per-class metrics
    class_report = classification_report(
        all_labels, all_predictions,
        target_names=predictor.classes,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "num_samples": len(all_labels),
        "per_class_metrics": class_report
    }
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return metrics, cm, all_probabilities


def save_metrics(metrics: dict, output_path: str = "metrics/evaluation_metrics.json"):
    """Save evaluation metrics to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str = "plots/confusion_matrix.png"):
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {output_path}")


def plot_classification_report(metrics: dict, output_path: str = "plots/classification_report.png"):
    """Plot and save classification report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract per-class metrics
    classes = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for class_name, class_metrics in metrics["per_class_metrics"].items():
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            classes.append(class_name)
            precision_scores.append(class_metrics["precision"])
            recall_scores.append(class_metrics["recall"])
            f1_scores.append(class_metrics["f1-score"])
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
    ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Classification report plot saved to: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ONNX model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for metrics and plots"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics, cm, probabilities = evaluate_onnx_model(args.model_path, args.data_dir)
    
    # Save results
    metrics_path = os.path.join(args.output_dir, "metrics", "evaluation_metrics.json")
    cm_path = os.path.join(args.output_dir, "plots", "confusion_matrix.png")
    report_path = os.path.join(args.output_dir, "plots", "classification_report.png")
    
    save_metrics(metrics, metrics_path)
    
    # Get class names from metrics
    class_names = [name for name in metrics["per_class_metrics"].keys() 
                   if name not in ["accuracy", "macro avg", "weighted avg"]]
    
    plot_confusion_matrix(cm, class_names, cm_path)
    plot_classification_report(metrics, report_path)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
