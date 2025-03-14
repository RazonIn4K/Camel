"""
Example script demonstrating the Advanced Analytics and Visualization capabilities.

This script shows how to use the advanced analytics utilities to analyze and visualize
results from AI red-teaming exercises.
"""

import os
import sys
import json
import random
from typing import Dict, Any, List
from datetime import datetime

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from cybersec_agents.grayswan.utils.advanced_analytics_utils import (
    extract_features_from_results,
    create_correlation_matrix,
    create_feature_distribution_plots,
    create_pairplot,
    create_advanced_clustering,
    create_advanced_model_comparison,
    create_advanced_success_prediction_model,
    create_advanced_analytics_report
)
from cybersec_agents.grayswan.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging("advanced_analytics_example")


def generate_sample_data(n_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Generate sample data for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        List of result dictionaries
    """
    models = ["GPT-4", "GPT-3.5", "Claude-2", "Llama-2", "Mistral-7B"]
    prompt_types = ["Direct", "Indirect", "Chain", "Role-play", "Technical"]
    attack_vectors = ["Jailbreak", "Prompt Injection", "Context Manipulation", "Misalignment", "Boundary Testing"]
    
    results = []
    
    for i in range(n_samples):
        model = random.choice(models)
        prompt_type = random.choice(prompt_types)
        attack_vector = random.choice(attack_vectors)
        
        # Generate prompt
        prompt_length = random.randint(50, 500)
        prompt = f"This is a sample prompt for {model} using {prompt_type} approach targeting {attack_vector}. "
        prompt += "Lorem ipsum " * (prompt_length // 12)
        
        # Generate response
        response_length = random.randint(20, 1000)
        response = f"This is a sample response from {model}. "
        response += "Lorem ipsum " * (response_length // 12)
        
        # Determine success based on model and attack vector
        # Different models have different vulnerabilities
        base_success_rate = {
            "GPT-4": 0.2,
            "GPT-3.5": 0.4,
            "Claude-2": 0.3,
            "Llama-2": 0.5,
            "Mistral-7B": 0.6
        }.get(model, 0.3)
        
        # Different attack vectors have different effectiveness
        attack_modifier = {
            "Jailbreak": 0.3,
            "Prompt Injection": 0.2,
            "Context Manipulation": 0.1,
            "Misalignment": 0.4,
            "Boundary Testing": 0.2
        }.get(attack_vector, 0.2)
        
        # Different prompt types have different effectiveness
        prompt_modifier = {
            "Direct": 0.1,
            "Indirect": 0.2,
            "Chain": 0.3,
            "Role-play": 0.4,
            "Technical": 0.2
        }.get(prompt_type, 0.2)
        
        # Calculate success probability
        success_prob = base_success_rate + attack_modifier + prompt_modifier
        success_prob = min(max(success_prob, 0.1), 0.9)  # Clamp between 0.1 and 0.9
        
        # Determine success
        success = random.random() < success_prob
        
        # Generate response time (successful attempts tend to take longer)
        response_time = random.uniform(0.5, 5.0)
        if success:
            response_time *= 1.5  # Successful attempts take longer
        
        # Create result dictionary
        result = {
            "model_name": model,
            "prompt_type": prompt_type,
            "attack_vector": attack_vector,
            "prompt": prompt,
            "response": response,
            "success": success,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        
        results.append(result)
    
    return results


def basic_feature_analysis_example(results: List[Dict[str, Any]], output_dir: str):
    """
    Demonstrate basic feature analysis.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
    """
    print("\n=== Basic Feature Analysis Example ===\n")
    
    # Extract features
    features_df = extract_features_from_results(results)
    
    print(f"Extracted {len(features_df)} samples with {len(features_df.columns)} features")
    print("\nFeature columns:")
    for col in features_df.columns:
        print(f"- {col}")
    
    # Create correlation matrix
    correlation_file = create_correlation_matrix(features_df, output_dir)
    if correlation_file:
        print(f"\nCreated correlation matrix: {correlation_file}")
    
    # Create feature distribution plots
    distribution_files = create_feature_distribution_plots(features_df, output_dir)
    if distribution_files:
        print(f"\nCreated {len(distribution_files)} feature distribution plots")
        for file in distribution_files:
            print(f"- {file}")
    
    # Create pairplot
    pairplot_file = create_pairplot(features_df, output_dir)
    if pairplot_file:
        print(f"\nCreated pairplot: {pairplot_file}")


def clustering_analysis_example(results: List[Dict[str, Any]], output_dir: str):
    """
    Demonstrate clustering analysis.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
    """
    print("\n=== Clustering Analysis Example ===\n")
    
    # Create KMeans clustering
    kmeans_results = create_advanced_clustering(
        results, output_dir, n_clusters=4, algorithm="kmeans"
    )
    
    if kmeans_results:
        print(f"\nCreated KMeans clustering visualization: {kmeans_results.get('clustering_file', '')}")
        print(f"Silhouette score: {kmeans_results.get('silhouette_score', 'N/A')}")
        
        # Load and print cluster analysis
        analysis_file = kmeans_results.get("analysis_file", "")
        if analysis_file and os.path.exists(analysis_file):
            with open(analysis_file, "r") as f:
                analysis = json.load(f)
            
            print(f"\nFound {analysis.get('n_clusters', 0)} clusters:")
            for cluster in analysis.get("clusters", []):
                print(f"- Cluster {cluster.get('cluster_id', '?')}: {cluster.get('size', 0)} samples, "
                      f"Success rate: {cluster.get('success_rate', 0):.2f}")
                
                # Print top models in cluster
                common_models = cluster.get("common_models", {})
                if common_models:
                    print("  Top models:")
                    for model, count in sorted(common_models.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"  - {model}: {count} samples")
    
    # Create DBSCAN clustering
    dbscan_results = create_advanced_clustering(
        results, output_dir, algorithm="dbscan"
    )
    
    if dbscan_results:
        print(f"\nCreated DBSCAN clustering visualization: {dbscan_results.get('clustering_file', '')}")
        print(f"Found {dbscan_results.get('n_clusters', 0)} clusters")
        print(f"Noise points: {dbscan_results.get('noise_points', 0)}")


def model_comparison_example(results: List[Dict[str, Any]], output_dir: str):
    """
    Demonstrate model comparison.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
    """
    print("\n=== Model Comparison Example ===\n")
    
    # Create advanced model comparison
    model_comparison_file = create_advanced_model_comparison(results, output_dir)
    
    if model_comparison_file:
        print(f"\nCreated advanced model comparison: {model_comparison_file}")
        
        # Check if model comparison table was created
        table_file = os.path.join(output_dir, "model_comparison_table.csv")
        if os.path.exists(table_file):
            print(f"Created model comparison table: {table_file}")
            
            # Load and print table
            import pandas as pd
            table = pd.read_csv(table_file)
            print("\nModel comparison table:")
            print(table)


def prediction_model_example(results: List[Dict[str, Any]], output_dir: str):
    """
    Demonstrate prediction model.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
    """
    print("\n=== Prediction Model Example ===\n")
    
    # Create Random Forest prediction model
    rf_results = create_advanced_success_prediction_model(
        results, output_dir, model_type="random_forest"
    )
    
    if rf_results:
        print(f"\nCreated Random Forest prediction model:")
        print(f"- Feature importance: {rf_results.get('importance_file', '')}")
        print(f"- ROC curve: {rf_results.get('roc_file', '')}")
        print(f"- Confusion matrix: {rf_results.get('cm_file', '')}")
        
        # Print metrics
        metrics = rf_results.get("metrics", {})
        if metrics:
            print("\nRandom Forest model metrics:")
            print(f"- Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"- Precision: {metrics.get('precision', 0):.3f}")
            print(f"- Recall: {metrics.get('recall', 0):.3f}")
            print(f"- F1 score: {metrics.get('f1', 0):.3f}")
            print(f"- ROC AUC: {metrics.get('roc_auc', 0):.3f}")
            
            # Print top features
            feature_importance = metrics.get("feature_importance", {})
            if feature_importance:
                print("\nTop 5 features by importance:")
                for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"- {feature}: {importance:.3f}")
    
    # Create Gradient Boosting prediction model
    gb_results = create_advanced_success_prediction_model(
        results, output_dir, model_type="gradient_boosting"
    )
    
    if gb_results:
        print(f"\nCreated Gradient Boosting prediction model:")
        print(f"- Feature importance: {gb_results.get('importance_file', '')}")
        print(f"- ROC curve: {gb_results.get('roc_file', '')}")
        print(f"- Confusion matrix: {gb_results.get('cm_file', '')}")
        
        # Print metrics
        metrics = gb_results.get("metrics", {})
        if metrics:
            print("\nGradient Boosting model metrics:")
            print(f"- Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"- Precision: {metrics.get('precision', 0):.3f}")
            print(f"- Recall: {metrics.get('recall', 0):.3f}")
            print(f"- F1 score: {metrics.get('f1', 0):.3f}")
            print(f"- ROC AUC: {metrics.get('roc_auc', 0):.3f}")


def comprehensive_report_example(results: List[Dict[str, Any]], output_dir: str):
    """
    Demonstrate comprehensive analytics report.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
    """
    print("\n=== Comprehensive Analytics Report Example ===\n")
    
    # Create comprehensive analytics report
    report_files = create_advanced_analytics_report(
        results, output_dir,
        include_clustering=True,
        include_prediction=True,
        include_model_comparison=True
    )
    
    if report_files:
        print(f"\nCreated comprehensive analytics report with {len(report_files)} files:")
        for name, file in report_files.items():
            print(f"- {name}: {file}")


def main():
    """Main function to run all examples."""
    print("Advanced Analytics and Visualization Examples")
    print("============================================")
    
    # Create output directory
    output_dir = os.path.join("data", "advanced_analytics_examples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    print("\nGenerating sample data...")
    results = generate_sample_data(n_samples=100)
    print(f"Generated {len(results)} sample results")
    
    # Save sample data
    sample_data_file = os.path.join(output_dir, "sample_data.json")
    with open(sample_data_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sample data to {sample_data_file}")
    
    # Run examples
    basic_feature_analysis_example(results, output_dir)
    clustering_analysis_example(results, output_dir)
    model_comparison_example(results, output_dir)
    prediction_model_example(results, output_dir)
    comprehensive_report_example(results, output_dir)
    
    print("\nAll examples completed!")
    print(f"Output files are in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()