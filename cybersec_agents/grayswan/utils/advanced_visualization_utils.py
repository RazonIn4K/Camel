"""Advanced visualization utilities for Gray Swan Arena."""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .logging_utils import setup_logging
from .visualization_utils import ensure_output_dir

# Set up logger
logger = setup_logging("AdvancedVisualizationUtils")


def create_attack_pattern_visualization(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Attack Pattern Clustering",
    n_clusters: int = 4,
    random_state: int = 42,
) -> str:
    """
    Create visualization showing attack patterns and effectiveness clusters.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the chart
        n_clusters: Number of clusters to identify
        random_state: Random state for reproducibility

    Returns:
        Path to the saved chart file
    """
    # Convert results to a format suitable for machine learning
    data: dict[str, Any] = []
    for result in results:
        # Extract features from the result
        features: dict[str, Any] = {
            "model": result.get("model_name", "Unknown"),
            "prompt_type": result.get("prompt_type", "Unknown"),
            "attack_vector": result.get("attack_vector", "Unknown"),
            "prompt_length": len(result.get("prompt", "")),
            "response_time": result.get("response_time", 0),
            "success": 1 if result.get("success", False) else 0,
        }
        data.append(features)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure we have enough data points
    if len(df) < 10:
        logger.warning("Not enough data points for attack pattern visualization")
        return ""

    # Ensure output directory exists
    ensure_output_dir(output_dir)

    try:
        # Select numerical columns
        numerical_cols: list[Any] = ["prompt_length", "response_time", "success"]

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_cols])

        # Apply dimensionality reduction
        if len(df) > 50:
            # Use t-SNE for larger datasets
            tsne = TSNE(n_components=2, random_state=random_state)
            reduced_data = tsne.fit_transform(scaled_data)
            reduction_method: str = "t-SNE"
        else:
            # Use PCA for smaller datasets
            pca = PCA(n_components=2, random_state=random_state)
            reduced_data = pca.fit_transform(scaled_data)
            reduction_method: str = "PCA"

        # Add reduced dimensions to DataFrame
        df["x"] = reduced_data[:, 0]
        df["y"] = reduced_data[:, 1]

        # Apply clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(df)), random_state=random_state)
        df["cluster"] = kmeans.fit_predict(scaled_data)

        # Create visualization
        plt.figure(figsize=(12, 10))

        # Create a scatter plot with multiple dimensions
        scatter = plt.scatter(
            df["x"],
            df["y"],
            c=df["cluster"],
            s=df["success"] * 100 + 50,  # Size based on success
            alpha=0.7,
            cmap="viridis",
            edgecolors="w",
            linewidths=0.5,
        )

        # Add labels for each point
        for i, row in df.iterrows():
            plt.annotate(
                f"{row['model'][:10]}",
                (row["x"], row["y"]),
                fontsize=8,
                alpha=0.7,
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )

        # Add cluster centers
        centers = kmeans.cluster_centers_
        if reduction_method == "PCA":
            # For PCA, we can directly transform the centers
            centers_2d = pca.transform(centers)
        else:
            # For t-SNE, we need to approximate the centers in 2D space
            centers_2d = np.array(
                [
                    df[df["cluster"] == i][["x", "y"]].mean()
                    for i in range(min(n_clusters, len(df)))
                ]
            )

        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            s=200,
            marker="X",
            c=range(len(centers_2d)),
            cmap="viridis",
            edgecolors="k",
            linewidths=2,
            alpha=0.8,
        )

        # Add legend for clusters
        legend_elements: list[Any] = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cm.viridis(i / n_clusters),
                markersize=10,
                label=f"Cluster {i+1}",
            )
            for i in range(min(n_clusters, len(df)))
        ]

        # Add legend for success (size)
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=5,
                label="Failure",
            )
        )
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=10,
                label="Success",
            )
        )

        plt.legend(handles=legend_elements, loc="best", title="Clusters & Outcomes")

        # Customize plot
        plt.title(f"{title} ({reduction_method})", fontsize=16)
        plt.xlabel(f"{reduction_method} Dimension 1", fontsize=12)
        plt.ylabel(f"{reduction_method} Dimension 2", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save plot
        output_file = os.path.join(output_dir, "attack_pattern_visualization.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created attack pattern visualization: {output_file}")

        # Create cluster analysis
        cluster_analysis = analyze_clusters(df, kmeans, numerical_cols)

        # Save cluster analysis
        analysis_file = os.path.join(output_dir, "cluster_analysis.json")
        with open(analysis_file, "w") as f:
            json.dump(cluster_analysis, f, indent=2)

        logger.info(f"Created cluster analysis: {analysis_file}")

        return output_file

    except Exception as e:
        logger.error(f"Error creating attack pattern visualization: {e}")
        return ""


def analyze_clusters(
    df: pd.DataFrame, kmeans: KMeans, feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Analyze clusters to identify patterns and characteristics.

    Args:
        df: DataFrame with cluster assignments
        kmeans: Fitted KMeans model
        feature_cols: Columns used for clustering

    Returns:
        Dictionary with cluster analysis
    """
    analysis = {"n_clusters": kmeans.n_clusters, "clusters": []}

    for i in range(kmeans.n_clusters):
        cluster_df = df[df["cluster"] == i]

        # Skip empty clusters
        if len(cluster_df) == 0:
            continue

        # Calculate cluster statistics
        cluster_info: dict[str, Any] = {
            "cluster_id": i,
            "size": len(cluster_df),
            "success_rate": cluster_df["success"].mean(),
            "avg_prompt_length": cluster_df["prompt_length"].mean(),
            "avg_response_time": cluster_df["response_time"].mean(),
            "common_models": cluster_df["model"].value_counts().to_dict(),
            "common_prompt_types": cluster_df["prompt_type"].value_counts().to_dict(),
            "common_attack_vectors": cluster_df["attack_vector"]
            .value_counts()
            .to_dict(),
        }

        # Add feature importance (distance from cluster mean to global mean)
        global_means = df[feature_cols].mean()
        cluster_means = cluster_df[feature_cols].mean()

        # Calculate normalized distances
        distances: dict[str, Any] = {}
        for col in feature_cols:
            global_std = df[col].std()
            if global_std > 0:
                distances[col] = (
                    abs(cluster_means[col] - global_means[col]) / global_std
                )
            else:
                distances[col] = 0

        cluster_info["feature_importance"] = distances

        analysis["clusters"].append(cluster_info)

    return analysis


def create_prompt_similarity_network(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Prompt Similarity Network",
    threshold: float = 0.5,
    random_state: int = 42,
) -> str:
    """
    Create a network visualization showing similarities between prompts.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the chart
        threshold: Similarity threshold for connecting nodes
        random_state: Random state for reproducibility

    Returns:
        Path to the saved chart file
    """
    try:
        # Extract prompts and success information
        prompts: list[Any] = []
        success: list[Any] = []
        models: dict[str, Any] = []

        for result in results:
            prompt = result.get("prompt", "")
            if prompt:
                prompts.append(prompt)
                success.append(1 if result.get("success", False) else 0)
                models.append(result.get("model_name", "Unknown"))

        if len(prompts) < 3:
            logger.warning("Not enough prompts for similarity network visualization")
            return ""

        # Ensure output directory exists
        ensure_output_dir(output_dir)

        # Try to import networkx
        try:
            import networkx as nx
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.error("Required packages not available: networkx, scikit-learn")
            return ""

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(prompts)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Create graph
        G = nx.Graph()

        # Add nodes
        for i, (prompt, is_success, model) in enumerate(zip(prompts, success, models)):
            # Truncate prompt for display
            short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt

            G.add_node(i, prompt=short_prompt, success=is_success, model=model)

        # Add edges based on similarity threshold
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    G.add_edge(i, j, weight=similarity)

        # Create plot
        plt.figure(figsize=(14, 12))

        # Use spring layout for node positioning
        pos = nx.spring_layout(G, seed=random_state)

        # Draw nodes
        node_colors: list[Any] = ["green" if s else "red" for s in success]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)

        # Draw edges with width based on similarity
        edge_weights: list[Any] = [G[u][v]["weight"] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color="gray")

        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            labels={
                i: f"{G.nodes[i]['model']}\n{G.nodes[i]['prompt']}" for i in G.nodes()
            },
            font_size=8,
            font_color="black",
            verticalalignment="bottom",
        )

        # Customize plot
        plt.title(title, fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        # Save plot
        output_file = os.path.join(output_dir, "prompt_similarity_network.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created prompt similarity network: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error creating prompt similarity network: {e}")
        return ""


def create_success_prediction_model(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Success Prediction Model",
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[str, Dict[str, Any]]:
    """
    Create a model to predict success based on prompt features and visualize feature importance.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the chart
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        Tuple of (path to the saved chart file, model metrics)
    """
    try:
        # Extract features and target
        data: dict[str, Any] = []
        for result in results:
            prompt = result.get("prompt", "")
            if not prompt:
                continue

            # Extract basic features
            features: dict[str, Any] = {
                "prompt_length": len(prompt),
                "word_count": len(prompt.split()),
                "question_marks": prompt.count("?"),
                "exclamation_marks": prompt.count("!"),
                "has_code_markers": 1 if "```" in prompt else 0,
                "uppercase_ratio": sum(1 for c in prompt if c.isupper()) / len(prompt)
                if prompt
                else 0,
                "model": result.get("model_name", "Unknown"),
                "prompt_type": result.get("prompt_type", "Unknown"),
                "attack_vector": result.get("attack_vector", "Unknown"),
                "success": 1 if result.get("success", False) else 0,
            }
            data.append(features)

        if len(data) < 10:
            logger.warning("Not enough data for success prediction model")
            return "", {}

        # Create DataFrame
        df = pd.DataFrame(data)

        # Ensure output directory exists
        ensure_output_dir(output_dir)

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            logger.error("Required packages not available: scikit-learn")
            return "", {}

        # Encode categorical features
        categorical_cols: list[Any] = ["model", "prompt_type", "attack_vector"]
        encoders: dict[str, Any] = {}

        for col in categorical_cols:
            if col in df.columns:
                encoder = LabelEncoder()
                df[f"{col}_encoded"] = encoder.fit_transform(df[col])
                encoders[col] = encoder

        # Prepare features and target
        feature_cols: list[Any] = [
            "prompt_length",
            "word_count",
            "question_marks",
            "exclamation_marks",
            "has_code_markers",
            "uppercase_ratio",
        ] + [f"{col}_encoded" for col in categorical_cols if col in df.columns]

        X = df[feature_cols]
        y = df["success"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)

        metrics: dict[str, Any] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
        }

        # Create feature importance visualization
        plt.figure(figsize=(12, 8))

        # Sort features by importance
        importance_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": model.feature_importances_}
        )
        importance_df = importance_df.sort_values("Importance", ascending=False)

        # Create bar chart
        ax = sns.barplot(x="Importance", y="Feature", data=importance_df)

        # Customize plot
        plt.title(f"{title} - Feature Importance", fontsize=16)
        plt.xlabel("Importance", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.tight_layout()

        # Save plot
        output_file = os.path.join(
            output_dir, "success_prediction_feature_importance.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Save model metrics
        metrics_file = os.path.join(output_dir, "success_prediction_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Created success prediction model: {output_file}")
        logger.info(f"Model metrics: {metrics_file}")

        return output_file, metrics

    except Exception as e:
        logger.error(f"Error creating success prediction model: {e}")
        return "", {}


def create_interactive_dashboard(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Gray Swan Arena Interactive Dashboard",
) -> str:
    """
    Create an interactive HTML dashboard with multiple visualizations.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the dashboard

    Returns:
        Path to the saved dashboard file
    """
    try:
        # Ensure output directory exists
        ensure_output_dir(output_dir)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Calculate summary statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get("success", False))
        success_rate = successful_tests / total_tests if total_tests > 0 else 0

        # Count unique models and prompt types
        models: dict[str, Any] = {r.get("model_name", "Unknown") for r in results}
        prompt_types: dict[str, Any] = {
            r.get("prompt_type", "Unknown") for r in results
        }
        attack_vectors: dict[str, Any] = {
            r.get("attack_vector", "Unknown") for r in results
        }

        # Create basic charts
        from .visualization_utils import (
            create_prompt_type_effectiveness_chart,
            create_response_time_chart,
            create_success_rate_chart,
            create_vulnerability_heatmap,
        )

        chart_files: dict[str, Any] = {
            "success_rate": create_success_rate_chart(results, output_dir),
            "response_time": create_response_time_chart(results, output_dir),
            "prompt_effectiveness": create_prompt_type_effectiveness_chart(
                results, output_dir
            ),
            "vulnerability_heatmap": create_vulnerability_heatmap(results, output_dir),
        }

        # Create advanced charts
        attack_pattern_file = create_attack_pattern_visualization(results, output_dir)
        if attack_pattern_file:
            chart_files["attack_pattern"] = attack_pattern_file

        similarity_network_file = create_prompt_similarity_network(results, output_dir)
        if similarity_network_file:
            chart_files["similarity_network"] = similarity_network_file

        prediction_file, prediction_metrics = create_success_prediction_model(
            results, output_dir
        )
        if prediction_file:
            chart_files["prediction_model"] = prediction_file

        # Generate HTML with interactive elements
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    background-color: #f9f9f9;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    border-radius: 8px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }}
                .stats-container {{
                    display: flex;
                    flex-wrap: wrap;
                    margin-bottom: 20px;
                }}
                .stat-box {{
                    flex: 1;
                    min-width: 200px;
                    padding: 15px;
                    margin: 10px;
                    background: #f5f5f5;
                    border-left: 4px solid #3498db;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    border-radius: 4px;
                }}
                .chart-container {{
                    margin-bottom: 30px;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .chart-row {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .chart-col {{
                    flex: 1;
                    min-width: 300px;
                }}
                .chart {{
                    max-width: 100%;
                    height: auto;
                    margin-top: 10px;
                    border-radius: 4px;
                }}
                .tabs {{
                    display: flex;
                    margin-bottom: 20px;
                }}
                .tab {{
                    padding: 10px 20px;
                    background: #f5f5f5;
                    border: none;
                    cursor: pointer;
                    border-radius: 4px 4px 0 0;
                    margin-right: 5px;
                }}
                .tab.active {{
                    background: #3498db;
                    color: white;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px;
                    background: white;
                    border-radius: 0 4px 4px 4px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .tab-content.active {{
                    display: block;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    text-align: left;
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f5f5f5;
                }}
                .success {{
                    color: green;
                }}
                .failure {{
                    color: red;
                }}
                .filters {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin-bottom: 20px;
                    padding: 15px;
                    background: #f5f5f5;
                    border-radius: 4px;
                }}
                .filter-group {{
                    display: flex;
                    flex-direction: column;
                }}
                .filter-group label {{
                    margin-bottom: 5px;
                    font-weight: bold;
                }}
                select, input {{
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .apply-filters {{
                    padding: 8px 15px;
                    background: #3498db;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    align-self: flex-end;
                }}
                .apply-filters:hover {{
                    background: #2980b9;
                }}
                .metric-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .metric-title {{
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-change {{
                    font-size: 14px;
                    margin-top: 5px;
                }}
                .metric-positive {{
                    color: green;
                }}
                .metric-negative {{
                    color: red;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="stats-container">
                    <div class="stat-box">
                        <h3>Total Tests</h3>
                        <p>{total_tests}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Success Rate</h3>
                        <p>{success_rate:.2%}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Models Tested</h3>
                        <p>{len(models)}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Prompt Types</h3>
                        <p>{len(prompt_types)}</p>
                    </div>
                </div>
                
                <div class="tabs">
                    <button class="tab active" onclick="openTab(event, 'overview')">Overview</button>
                    <button class="tab" onclick="openTab(event, 'advanced')">Advanced Analysis</button>
                    <button class="tab" onclick="openTab(event, 'details')">Detailed Results</button>
                </div>
                
                <div id="overview" class="tab-content active">
                    <div class="chart-row">
                        <div class="chart-col">
                            <div class="chart-container">
                                <h2>Success Rate by Model and Prompt Type</h2>
                                <img class="chart" src="{os.path.basename(chart_files.get('success_rate', ''))}" alt="Success Rate Chart">
                            </div>
                        </div>
                        <div class="chart-col">
                            <div class="chart-container">
                                <h2>Response Time by Model</h2>
                                <img class="chart" src="{os.path.basename(chart_files.get('response_time', ''))}" alt="Response Time Chart">
                            </div>
                        </div>
                    </div>
                    
                    <div class="chart-row">
                        <div class="chart-col">
                            <div class="chart-container">
                                <h2>Prompt Type Effectiveness</h2>
                                <img class="chart" src="{os.path.basename(chart_files.get('prompt_effectiveness', ''))}" alt="Prompt Effectiveness Chart">
                            </div>
                        </div>
                        <div class="chart-col">
                            <div class="chart-container">
                                <h2>Vulnerability Heatmap</h2>
                                <img class="chart" src="{os.path.basename(chart_files.get('vulnerability_heatmap', ''))}" alt="Vulnerability Heatmap">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div id="advanced" class="tab-content">
                    <div class="chart-row">
        """

        # Add attack pattern visualization if available
        if "attack_pattern" in chart_files:
            html += f"""
                        <div class="chart-col">
                            <div class="chart-container">
                                <h2>Attack Pattern Clustering</h2>
                                <img class="chart" src="{os.path.basename(chart_files['attack_pattern'])}" alt="Attack Pattern Visualization">
                            </div>
                        </div>
            """

        # Add similarity network if available
        if "similarity_network" in chart_files:
            html += f"""
                        <div class="chart-col">
                            <div class="chart-container">
                                <h2>Prompt Similarity Network</h2>
                                <img class="chart" src="{os.path.basename(chart_files['similarity_network'])}" alt="Prompt Similarity Network">
                            </div>
                        </div>
            """

        html += """
                    </div>
                    
                    <div class="chart-row">
        """

        # Add prediction model if available
        if "prediction_model" in chart_files:
            html += f"""
                        <div class="chart-col">
                            <div class="chart-container">
                                <h2>Success Prediction Model - Feature Importance</h2>
                                <img class="chart" src="{os.path.basename(chart_files['prediction_model'])}" alt="Success Prediction Model">
                            </div>
                        </div>
            """

            # Add prediction metrics if available
            if prediction_metrics:
                html += f"""
                        <div class="chart-col">
                            <div class="chart-container">
                                <h2>Model Performance Metrics</h2>
                                <div class="metric-card">
                                    <div class="metric-title">Accuracy</div>
                                    <div class="metric-value">{prediction_metrics.get('accuracy', 0):.2f}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">Precision</div>
                                    <div class="metric-value">{prediction_metrics.get('precision', 0):.2f}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">Recall</div>
                                    <div class="metric-value">{prediction_metrics.get('recall', 0):.2f}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">F1 Score</div>
                                    <div class="metric-value">{prediction_metrics.get('f1', 0):.2f}</div>
                                </div>
                            </div>
                        </div>
                """

        # Add detailed results tab
        html += """
                    </div>
                </div>
                
                <div id="details" class="tab-content">
                    <div class="filters">
                        <div class="filter-group">
                            <label for="model-filter">Model</label>
                            <select id="model-filter">
                                <option value="all">All Models</option>
        """

        # Add model options
        for model in models:
            html += f'<option value="{model}">{model}</option>\n'

        html += """
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="prompt-type-filter">Prompt Type</label>
                            <select id="prompt-type-filter">
                                <option value="all">All Prompt Types</option>
        """

        # Add prompt type options
        for prompt_type in prompt_types:
            html += f'<option value="{prompt_type}">{prompt_type}</option>\n'

        html += """
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="attack-vector-filter">Attack Vector</label>
                            <select id="attack-vector-filter">
                                <option value="all">All Attack Vectors</option>
        """

        # Add attack vector options
        for attack_vector in attack_vectors:
            html += f'<option value="{attack_vector}">{attack_vector}</option>\n'

        html += """
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="success-filter">Success</label>
                            <select id="success-filter">
                                <option value="all">All</option>
                                <option value="success">Success</option>
                                <option value="failure">Failure</option>
                            </select>
                        </div>
                        <button class="apply-filters" onclick="applyFilters()">Apply Filters</button>
                    </div>
                    
                    <table id="results-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Prompt Type</th>
                                <th>Attack Vector</th>
                                <th>Success</th>
                                <th>Response Time (s)</th>
                                <th>Prompt</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        # Add table rows
        for result in results:
            model = result.get("model_name", "Unknown")
            prompt_type = result.get("prompt_type", "Unknown")
            attack_vector = result.get("attack_vector", "Unknown")
            success = result.get("success", False)
            response_time = result.get("response_time", 0)
            prompt = result.get("prompt", "")

            # Truncate prompt for display
            short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt

            success_class: str = "success" if success else "failure"
            success_text: str = "Yes" if success else "No"

            html += f"""
                            <tr data-model="{model}" data-prompt-type="{prompt_type}" data-attack-vector="{attack_vector}" data-success="{success}">
                                <td>{model}</td>
                                <td>{prompt_type}</td>
                                <td>{attack_vector}</td>
                                <td class="{success_class}">{success_text}</td>
                                <td>{response_time:.2f}</td>
                                <td title="{prompt}">{short_prompt}</td>
                            </tr>
            """

        # Close table and add JavaScript
        html += """
                        </tbody>
                    </table>
                </div>
                
                <script>
                    function openTab(evt, tabName) {
                        var i, tabcontent, tablinks;
                        tabcontent = document.getElementsByClassName("tab-content");
                        for (i = 0; i < tabcontent.length; i++) {
                            tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                        }
                        tablinks = document.getElementsByClassName("tab");
                        for (i = 0; i < tablinks.length; i++) {
                            tablinks[i].className = tablinks[i].className.replace(" active", "");
                        }
                        document.getElementById(tabName).className += " active";
                        evt.currentTarget.className += " active";
                    }
                    
                    function applyFilters() {
                        var modelFilter = document.getElementById("model-filter").value;
                        var promptTypeFilter = document.getElementById("prompt-type-filter").value;
                        var attackVectorFilter = document.getElementById("attack-vector-filter").value;
                        var successFilter = document.getElementById("success-filter").value;
                        
                        var rows = document.getElementById("results-table").getElementsByTagName("tbody")[0].rows;
                        
                        for (var i = 0; i < rows.length; i++) {
                            var row = rows[i];
                            var model = row.getAttribute("data-model");
                            var promptType = row.getAttribute("data-prompt-type");
                            var attackVector = row.getAttribute("data-attack-vector");
                            var success = row.getAttribute("data-success") === "true";
                            
                            var showRow = true;
                            
                            if (modelFilter !== "all" && model !== modelFilter) {
                                showRow = false;
                            }
                            
                            if (promptTypeFilter !== "all" && promptType !== promptTypeFilter) {
                                showRow = false;
                            }
                            
                            if (attackVectorFilter !== "all" && attackVector !== attackVectorFilter) {
                                showRow = false;
                            }
                            
                            if (successFilter === "success" && !success) {
                                showRow = false;
                            } else if (successFilter === "failure" && success) {
                                showRow = false;
                            }
                            
                            row.style.display = showRow ? "" : "none";
                        }
                    }
                </script>
            </div>
        </body>
        </html>
        """

        # Save HTML
        output_file = os.path.join(output_dir, "interactive_dashboard.html")
        with open(output_file, "w") as f:
            f.write(html)

        logger.info(f"Created interactive dashboard: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}")
        return ""


def create_advanced_evaluation_report(
    results: List[Dict[str, Any]], output_dir: str, include_interactive: bool = True
) -> Dict[str, str]:
    """
    Create a comprehensive evaluation report with advanced visualizations.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        include_interactive: Whether to include interactive dashboard

    Returns:
        Dictionary mapping chart names to file paths
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Create basic visualizations using the original utility
    from .visualization_utils import create_evaluation_report

    basic_report_files = create_evaluation_report(results, output_dir)

    # Create advanced visualizations
    advanced_report_files: dict[str, Any] = {}

    # Attack pattern visualization
    attack_pattern_file = create_attack_pattern_visualization(results, output_dir)
    if attack_pattern_file:
        advanced_report_files["attack_pattern"] = attack_pattern_file

    # Prompt similarity network
    similarity_network_file = create_prompt_similarity_network(results, output_dir)
    if similarity_network_file:
        advanced_report_files["similarity_network"] = similarity_network_file

    # Success prediction model
    prediction_file, _ = create_success_prediction_model(results, output_dir)
    if prediction_file:
        advanced_report_files["prediction_model"] = prediction_file

    # Interactive dashboard
    if include_interactive:
        dashboard_file = create_interactive_dashboard(results, output_dir)
        if dashboard_file:
            advanced_report_files["interactive_dashboard"] = dashboard_file

    # Combine all report files
    report_files: dict[str, Any] = {**basic_report_files, **advanced_report_files}

    return report_files
