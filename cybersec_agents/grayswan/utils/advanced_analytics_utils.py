"""
Advanced analytics utilities for Gray Swan Arena.

This module provides advanced analytics capabilities for analyzing and visualizing
results from AI red-teaming exercises.
"""

import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .advanced_visualization_utils import (
    create_attack_pattern_visualization,
    create_interactive_dashboard,
    create_prompt_similarity_network,
    create_success_prediction_model,
)
from .logging_utils import setup_logging
from .visualization_utils import ensure_output_dir

# Set up logger
logger = setup_logging("AdvancedAnalyticsUtils")


def extract_features_from_results(
    results: List[Dict[str, Any]],
    include_text_features: bool = True,
    include_nlp_features: bool = True,
) -> pd.DataFrame:
    """
    Extract features from results for analysis.

    Args:
        results: List of result dictionaries
        include_text_features: Whether to include text-based features
        include_nlp_features: Whether to include NLP-based features

    Returns:
        DataFrame with extracted features
    """
    # Basic features
    data: dict[str, Any] = []
    for result in results:
        prompt = result.get("prompt", "")
        if not prompt:
            continue

        # Extract basic features
        features: dict[str, Any] = {
            "model": result.get("model_name", "Unknown"),
            "prompt_type": result.get("prompt_type", "Unknown"),
            "attack_vector": result.get("attack_vector", "Unknown"),
            "success": 1 if result.get("success", False) else 0,
            "response_time": result.get("response_time", 0),
        }

        # Add text-based features
        if include_text_features:
            features.update(
                {
                    "prompt_length": len(prompt),
                    "word_count": len(prompt.split()),
                    "question_marks": prompt.count("?"),
                    "exclamation_marks": prompt.count("!"),
                    "has_code_markers": 1 if "```" in prompt else 0,
                    "uppercase_ratio": sum(1 for c in prompt if c.isupper())
                    / len(prompt)
                    if prompt
                    else 0,
                    "digit_ratio": sum(1 for c in prompt if c.isdigit()) / len(prompt)
                    if prompt
                    else 0,
                    "special_char_ratio": sum(
                        1 for c in prompt if not c.isalnum() and not c.isspace()
                    )
                    / len(prompt)
                    if prompt
                    else 0,
                    "avg_word_length": sum(len(word) for word in prompt.split())
                    / len(prompt.split())
                    if prompt.split()
                    else 0,
                    "line_count": prompt.count("\n") + 1,
                }
            )

        # Add NLP-based features if requested
        if include_nlp_features:
            try:
                # Import NLTK utilities
                from cybersec_agents.utils.nltk_utils import analyze_sentiment

                # Analyze sentiment
                sentiment_scores = analyze_sentiment(prompt)

                # Add sentiment features if analysis was successful
                if sentiment_scores:
                    features.update(
                        {
                            "sentiment_negative": sentiment_scores["neg"],
                            "sentiment_neutral": sentiment_scores["neu"],
                            "sentiment_positive": sentiment_scores["pos"],
                            "sentiment_compound": sentiment_scores["compound"],
                        }
                    )
                else:
                    logger.warning("Sentiment analysis failed for prompt")
            except ImportError:
                logger.warning("NLTK utilities not available for NLP features")

        # Add response features if available
        response = result.get("response", "")
        if response and isinstance(response, str):
            features.update(
                {
                    "response_length": len(response),
                    "response_word_count": len(response.split()),
                }
            )

        data.append(features)

    # Create DataFrame
    df = pd.DataFrame(data)

    return df


def create_correlation_matrix(
    df: pd.DataFrame,
    output_dir: str,
    title: str = "Feature Correlation Matrix",
    figsize: Tuple[int, int] = (14, 12),
) -> str:
    """
    Create a correlation matrix visualization.

    Args:
        df: DataFrame with features
        output_dir: Directory where visualization files will be saved
        title: Title of the chart
        figsize: Figure size (width, height) in inches

    Returns:
        Path to the saved chart file
    """
    try:
        # Ensure output directory exists
        ensure_output_dir(output_dir)

        # Select numerical columns
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()

        # Create plot
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.5},
        )

        # Customize plot
        plt.title(title, fontsize=16)
        plt.tight_layout()

        # Save plot
        output_file = os.path.join(output_dir, "correlation_matrix.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created correlation matrix: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}")
        return ""


def create_feature_distribution_plots(
    df: pd.DataFrame,
    output_dir: str,
    features: Optional[List[str]] = None,
    hue: str = "success",
    figsize: Tuple[int, int] = (16, 12),
) -> List[str]:
    """
    Create distribution plots for features.

    Args:
        df: DataFrame with features
        output_dir: Directory where visualization files will be saved
        features: List of features to plot (if None, uses all numerical features)
        hue: Column to use for coloring (typically "success")
        figsize: Figure size (width, height) in inches

    Returns:
        List of paths to the saved chart files
    """
    try:
        # Ensure output directory exists
        ensure_output_dir(output_dir)

        # Select features to plot
        if features is None:
            features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            # Remove the hue column if it's in the features list
            if hue in features:
                features.remove(hue)

        # Create a subdirectory for distribution plots
        dist_dir = os.path.join(output_dir, "feature_distributions")
        ensure_output_dir(dist_dir)

        output_files: list[Any] = []

        # Create plots in batches of 4
        for i in range(0, len(features), 4):
            batch_features = features[i : i + 4]
            n_features = len(batch_features)

            # Calculate grid dimensions
            n_cols = min(2, n_features)
            n_rows = math.ceil(n_features / n_cols)

            # Create figure
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            # Create plots
            for j, feature in enumerate(batch_features):
                if feature in df.columns:
                    ax = axes[j]

                    # Check if feature is binary
                    if df[feature].nunique() <= 2:
                        # Create countplot for binary features
                        sns.countplot(x=feature, hue=hue, data=df, ax=ax)
                    else:
                        # Create histogram for continuous features
                        sns.histplot(x=feature, hue=hue, data=df, kde=True, ax=ax)

                    # Customize plot
                    ax.set_title(f"Distribution of {feature}")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Count")

                    # Add legend
                    if hue:
                        ax.legend(title=hue.replace("_", " ").title())

            # Hide unused subplots
            for j in range(n_features, len(axes)):
                axes[j].set_visible(False)

            # Add overall title
            plt.suptitle(f"Feature Distributions (Batch {i//4 + 1})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # Save plot
            output_file = os.path.join(
                dist_dir, f"feature_distributions_batch_{i//4 + 1}.png"
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            output_files.append(output_file)

        logger.info(f"Created {len(output_files)} feature distribution plots")
        return output_files

    except Exception as e:
        logger.error(f"Error creating feature distribution plots: {e}")
        return []


def create_pairplot(
    df: pd.DataFrame,
    output_dir: str,
    features: Optional[List[str]] = None,
    hue: str = "success",
    title: str = "Feature Pairplot",
    max_features: int = 5,
) -> str:
    """
    Create a pairplot visualization.

    Args:
        df: DataFrame with features
        output_dir: Directory where visualization files will be saved
        features: List of features to plot (if None, selects top correlated features)
        hue: Column to use for coloring (typically "success")
        title: Title of the chart
        max_features: Maximum number of features to include

    Returns:
        Path to the saved chart file
    """
    try:
        # Ensure output directory exists
        ensure_output_dir(output_dir)

        # Select features to plot
        if features is None:
            # Select features most correlated with success
            if hue in df.columns and df[hue].nunique() <= 5:
                corr_with_target = df.corr()[hue].abs().sort_values(ascending=False)
                features = corr_with_target.index[
                    1 : max_features + 1
                ].tolist()  # Skip the target itself
            else:
                # Just select a few numerical features
                numerical_cols = df.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()
                features = numerical_cols[:max_features]
        else:
            # Limit to max_features
            features = features[:max_features]

        # Add the hue column if it's not in the features list
        plot_cols = features.copy()
        if hue not in plot_cols:
            plot_cols.append(hue)

        # Create pairplot
        g = sns.pairplot(
            df[plot_cols], hue=hue, diag_kind="kde", plot_kws={"alpha": 0.6}, height=2.5
        )

        # Customize plot
        g.fig.suptitle(title, fontsize=16, y=1.02)

        # Save plot
        output_file = os.path.join(output_dir, "feature_pairplot.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created pairplot: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error creating pairplot: {e}")
        return ""


def create_advanced_clustering(
    results: List[Dict[str, Any]],
    output_dir: str,
    n_clusters: int = 4,
    algorithm: str = "kmeans",
    perplexity: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Create advanced clustering visualization with multiple algorithms.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        n_clusters: Number of clusters for KMeans
        algorithm: Clustering algorithm to use ('kmeans' or 'dbscan')
        perplexity: Perplexity parameter for t-SNE
        random_state: Random state for reproducibility

    Returns:
        Dictionary with paths to saved files and clustering metrics
    """
    try:
        # Extract features
        df = extract_features_from_results(results)

        # Ensure output directory exists
        ensure_output_dir(output_dir)

        # Select numerical columns for clustering
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Remove target variable if present
        if "success" in numerical_cols:
            numerical_cols.remove("success")

        # Ensure we have enough data points
        if len(df) < 10 or len(numerical_cols) < 2:
            logger.warning("Not enough data points for advanced clustering")
            return {}

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_cols])

        # Apply dimensionality reduction
        if len(df) > 50:
            # Use t-SNE for larger datasets
            tsne = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(df) - 1),
                random_state=random_state,
            )
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
        if algorithm.lower() == "kmeans":
            # KMeans clustering
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(df)), random_state=random_state
            )
            df["cluster"] = kmeans.fit_predict(scaled_data)

            # Calculate silhouette score if possible
            try:
                from sklearn.metrics import silhouette_score

                silhouette_avg = silhouette_score(scaled_data, df["cluster"])
                logger.info(f"Silhouette Score: {silhouette_avg:.3f}")
            except:
                silhouette_avg: Optional[Any] = None

            # Create visualization
            plt.figure(figsize=(14, 12))

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
                    markerfacecolor=plt.cm.viridis(i / n_clusters),
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
            plt.title(f"Advanced Clustering ({reduction_method} + KMeans)", fontsize=16)
            plt.xlabel(f"{reduction_method} Dimension 1", fontsize=12)
            plt.ylabel(f"{reduction_method} Dimension 2", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save plot
            output_file = os.path.join(output_dir, "advanced_clustering_kmeans.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            # Create cluster analysis
            cluster_analysis = analyze_clusters(df, kmeans, numerical_cols)

            # Save cluster analysis
            analysis_file = os.path.join(output_dir, "advanced_cluster_analysis.json")
            with open(analysis_file, "w") as f:
                json.dump(cluster_analysis, f, indent=2)

            logger.info(f"Created advanced KMeans clustering: {output_file}")

            return {
                "clustering_file": output_file,
                "analysis_file": analysis_file,
                "silhouette_score": silhouette_avg,
                "algorithm": "kmeans",
                "n_clusters": min(n_clusters, len(df)),
                "reduction_method": reduction_method,
            }

        elif algorithm.lower() == "dbscan":
            # DBSCAN clustering
            from sklearn.cluster import DBSCAN

            # Determine epsilon based on nearest neighbors
            from sklearn.neighbors import NearestNeighbors

            neighbors = NearestNeighbors(n_neighbors=min(5, len(df) - 1))
            neighbors_fit = neighbors.fit(scaled_data)
            distances, indices = neighbors_fit.kneighbors(scaled_data)
            distances = np.sort(distances[:, -1])

            # Estimate epsilon as the "elbow" in the k-distance graph
            # For simplicity, we'll use a heuristic approach
            epsilon = np.percentile(distances, 90) * 0.5

            # Apply DBSCAN
            dbscan = DBSCAN(eps=epsilon, min_samples=3)
            df["cluster"] = dbscan.fit_predict(scaled_data)

            # Count clusters (excluding noise points labeled as -1)
            n_clusters_found = len(set(df["cluster"])) - (
                1 if -1 in df["cluster"] else 0
            )

            # Calculate silhouette score if possible and if we have more than one cluster
            silhouette_avg: Optional[Any] = None
            if n_clusters_found > 1:
                try:
                    from sklearn.metrics import silhouette_score

                    # Filter out noise points for silhouette calculation
                    non_noise_mask = df["cluster"] != -1
                    if sum(non_noise_mask) > 1:
                        silhouette_avg = silhouette_score(
                            scaled_data[non_noise_mask], df["cluster"][non_noise_mask]
                        )
                        logger.info(f"Silhouette Score: {silhouette_avg:.3f}")
                except Exception as e:
                    logger.warning(f"Could not calculate silhouette score: {e}")

            # Create visualization
            plt.figure(figsize=(14, 12))

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

            # Add legend for clusters
            unique_clusters = sorted(df["cluster"].unique())
            legend_elements: list[Any] = []

            for i, cluster_id in enumerate(unique_clusters):
                if cluster_id == -1:
                    # Noise points
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor="gray",
                            markersize=10,
                            label="Noise",
                        )
                    )
                else:
                    # Regular clusters
                    color_idx: tuple[Any, ...] = (cluster_id % 10) / 10  # Cycle through colors
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=plt.cm.viridis(color_idx),
                            markersize=10,
                            label=f"Cluster {cluster_id+1}",
                        )
                    )

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
            plt.title(f"Advanced Clustering ({reduction_method} + DBSCAN)", fontsize=16)
            plt.xlabel(f"{reduction_method} Dimension 1", fontsize=12)
            plt.ylabel(f"{reduction_method} Dimension 2", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save plot
            output_file = os.path.join(output_dir, "advanced_clustering_dbscan.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            # Create cluster analysis
            cluster_analysis: dict[str, Any] = {
                "n_clusters": n_clusters_found,
                "epsilon": epsilon,
                "noise_points": int(sum(df["cluster"] == -1)),
                "clusters": [],
            }

            for cluster_id in sorted(df["cluster"].unique()):
                cluster_df = df[df["cluster"] == cluster_id]

                # Skip empty clusters
                if len(cluster_df) == 0:
                    continue

                # Calculate cluster statistics
                cluster_info: dict[str, Any] = {
                    "cluster_id": int(cluster_id),
                    "size": len(cluster_df),
                    "success_rate": float(cluster_df["success"].mean()),
                    "is_noise": cluster_id == -1,
                }

                # Add model and prompt type distributions
                for col in ["model", "prompt_type", "attack_vector"]:
                    if col in cluster_df.columns:
                        cluster_info[f"common_{col}s"] = (
                            cluster_df[col].value_counts().to_dict()
                        )

                cluster_analysis["clusters"].append(cluster_info)

            # Save cluster analysis
            analysis_file = os.path.join(
                output_dir, "advanced_cluster_analysis_dbscan.json"
            )
            with open(analysis_file, "w") as f:
                json.dump(cluster_analysis, f, indent=2)

            logger.info(f"Created advanced DBSCAN clustering: {output_file}")

            return {
                "clustering_file": output_file,
                "analysis_file": analysis_file,
                "silhouette_score": silhouette_avg,
                "algorithm": "dbscan",
                "n_clusters": n_clusters_found,
                "epsilon": epsilon,
                "noise_points": int(sum(df["cluster"] == -1)),
                "reduction_method": reduction_method,
            }

        else:
            logger.error(f"Unknown clustering algorithm: {algorithm}")
            return {}

    except Exception as e:
        logger.error(f"Error creating advanced clustering: {e}")
        return {}


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
            "success_rate": float(cluster_df["success"].mean())
            if "success" in cluster_df.columns
            else None,
        }

        # Add average values for numerical features
        for col in feature_cols:
            if col in cluster_df.columns:
                cluster_info[f"avg_{col}"] = float(cluster_df[col].mean())

        # Add model and prompt type distributions
        for col in ["model", "prompt_type", "attack_vector"]:
            if col in cluster_df.columns:
                cluster_info[f"common_{col}s"] = (
                    cluster_df[col].value_counts().to_dict()
                )

        # Add feature importance (distance from cluster mean to global mean)
        global_means = df[feature_cols].mean()
        cluster_means = cluster_df[feature_cols].mean()

        # Calculate normalized distances
        distances: dict[str, Any] = {}
        for col in feature_cols:
            global_std = df[col].std()
            if global_std > 0:
                distances[col] = float(
                    abs(cluster_means[col] - global_means[col]) / global_std
                )
            else:
                distances[col] = 0.0

        cluster_info["feature_importance"] = distances

        analysis["clusters"].append(cluster_info)

    return analysis


def create_advanced_model_comparison(
    results: List[Dict[str, Any]],
    output_dir: str,
    title: str = "Advanced Model Comparison",
) -> str:
    """
    Create an advanced model comparison visualization.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        title: Title of the chart

    Returns:
        Path to the saved chart file
    """
    try:
        # Process results
        data: dict[str, Any] = []
        for result in results:
            model = result.get("model_name", "Unknown")
            prompt_type = result.get("prompt_type", "Unknown")
            attack_vector = result.get("attack_vector", "Unknown")
            success = result.get("success", False)
            response_time = result.get("response_time", 0)

            data.append(
                {
                    "Model": model,
                    "Prompt Type": prompt_type,
                    "Attack Vector": attack_vector,
                    "Success": 1 if success else 0,
                    "Response Time": response_time,
                }
            )

        if not data:
            logger.warning("No data available for advanced model comparison")
            return ""

        # Create DataFrame
        df = pd.DataFrame(data)

        # Ensure output directory exists
        ensure_output_dir(output_dir)

        # Calculate metrics by model
        model_metrics: tuple[Any, ...] = (
            df.groupby("Model")
            .agg(
                {
                    "Success": ["mean", "count"],
                    "Response Time": ["mean", "median", "std"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        model_metrics.columns = [
            "Model" if col[0] == "Model" else f"{col[0]}_{col[1]}"
            for col in model_metrics.columns
        ]

        # Calculate success rate by prompt type for each model
        prompt_success = df.pivot_table(
            index="Model", columns="Prompt Type", values="Success", aggfunc="mean"
        ).reset_index()

        # Merge metrics
        model_metrics = model_metrics.merge(prompt_success, on="Model")

        # Create radar chart
        plt.figure(figsize=(14, 10))

        # Get unique models
        models: dict[str, Any] = model_metrics["Model"].unique()
        n_models = len(models)

        # Select metrics for radar chart
        radar_metrics: list[Any] = ["Success_mean", "Response Time_mean"]

        # Add prompt type success rates
        prompt_types: list[Any] = [
            col
            for col in model_metrics.columns
            if col
            not in [
                "Model",
                "Success_mean",
                "Success_count",
                "Response Time_mean",
                "Response Time_median",
                "Response Time_std",
            ]
        ]
        radar_metrics.extend(prompt_types)

        # Number of variables
        N = len(radar_metrics)

        # Create angles for radar chart
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Create subplot with polar projection
        ax = plt.subplot(111, polar=True)

        # Add variable labels
        plt.xticks(angles[:-1], radar_metrics, size=12)

        # Add y-axis labels
        ax.set_rlabel_position(0)
        plt.yticks(
            [0.25, 0.5, 0.75, 1.0],
            ["0.25", "0.5", "0.75", "1.0"],
            color="grey",
            size=10,
        )
        plt.ylim(0, 1)

        # Plot each model
        for i, model in enumerate(models):
            # Get model data
            model_data = model_metrics[model_metrics["Model"] == model]

            # Extract values for radar chart
            values: list[Any] = []
            for metric in radar_metrics:
                if metric == "Response Time_mean":
                    # Normalize response time (lower is better)
                    max_time = model_metrics["Response Time_mean"].max()
                    if max_time > 0:
                        # Invert so that lower response time = higher score
                        value = 1 - (
                            model_data["Response Time_mean"].values[0] / max_time
                        )
                    else:
                        value: float = 1.0
                else:
                    # For success rates, higher is better
                    value = model_data[metric].values[0]

                values.append(value)

            # Close the loop
            values += values[:1]

            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle="solid", label=model)
            ax.fill(angles, values, alpha=0.1)

        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        # Add title
        plt.title(title, size=16, y=1.1)

        # Save plot
        output_file = os.path.join(output_dir, "advanced_model_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created advanced model comparison: {output_file}")

        # Create detailed model comparison table
        model_table = model_metrics.copy()

        # Rename columns for better readability
        model_table = model_table.rename(
            columns={
                "Success_mean": "Success Rate",
                "Success_count": "Test Count",
                "Response Time_mean": "Avg Response Time",
                "Response Time_median": "Median Response Time",
                "Response Time_std": "Response Time Std Dev",
            }
        )

        # Format numeric columns
        for col in model_table.columns:
            if col != "Model" and model_table[col].dtype in [np.float64, np.float32]:
                if "Time" in col:
                    # Format time in seconds
                    model_table[col] = model_table[col].round(2).astype(str) + " s"
                else:
                    # Format percentages
                    model_table[col] = (model_table[col] * 100).round(1).astype(
                        str
                    ) + "%"

        # Save table as CSV
        table_file = os.path.join(output_dir, "model_comparison_table.csv")
        model_table.to_csv(table_file, index=False)

        logger.info(f"Created model comparison table: {table_file}")

        return output_file

    except Exception as e:
        logger.error(f"Error creating advanced model comparison: {e}")
        return ""


def create_advanced_success_prediction_model(
    results: List[Dict[str, Any]],
    output_dir: str,
    model_type: str = "random_forest",
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Create an advanced model to predict success based on prompt features.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        model_type: Type of model to use ('random_forest' or 'gradient_boosting')
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        Dictionary with paths to saved files and model metrics
    """
    try:
        # Extract features
        df = extract_features_from_results(results)

        if len(df) < 20:
            logger.warning("Not enough data for advanced success prediction model")
            return {}

        # Ensure output directory exists
        ensure_output_dir(output_dir)

        # Prepare features and target
        feature_cols: list[Any] = [
            col
            for col in df.columns
            if col not in ["success", "model", "prompt_type", "attack_vector"]
        ]

        # Handle categorical features
        categorical_cols: list[Any] = []
        for col in ["model", "prompt_type", "attack_vector"]:
            if col in df.columns:
                # One-hot encode categorical features
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                categorical_cols.extend(dummies.columns.tolist())

        # Combine all feature columns
        all_feature_cols = feature_cols + categorical_cols

        # Check if we have enough features
        if len(all_feature_cols) < 2:
            logger.warning("Not enough features for advanced success prediction model")
            return {}

        X = df[all_feature_cols]
        y = df["success"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train model
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=random_state,
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {}

        # Fit model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create feature importance visualization
        plt.figure(figsize=(12, 8))

        # Sort features by importance
        importance_df = pd.DataFrame(
            {"Feature": all_feature_cols, "Importance": model.feature_importances_}
        )
        importance_df = importance_df.sort_values("Importance", ascending=False)

        # Create bar chart
        ax = sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))

        # Customize plot
        plt.title(
            f"Advanced Success Prediction - Feature Importance ({model_type.replace('_', ' ').title()})",
            fontsize=16,
        )
        plt.xlabel("Importance", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.tight_layout()

        # Save plot
        importance_file = os.path.join(
            output_dir, f"advanced_success_prediction_importance_{model_type}.png"
        )
        plt.savefig(importance_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Create ROC curve visualization
        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title(f"Receiver Operating Characteristic (ROC) Curve", fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save plot
        roc_file = os.path.join(
            output_dir, f"advanced_success_prediction_roc_{model_type}.png"
        )
        plt.savefig(roc_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Create confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16)
        plt.tight_layout()

        # Save plot
        cm_file = os.path.join(
            output_dir, f"advanced_success_prediction_cm_{model_type}.png"
        )
        plt.savefig(cm_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Save model metrics
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist(),
            "feature_importance": dict(
                zip(
                    importance_df["Feature"].tolist(),
                    importance_df["Importance"].tolist(),
                )
            ),
            "model_type": model_type,
            "test_size": test_size,
            "n_samples": len(df),
            "n_features": len(all_feature_cols),
        }

        # Save metrics as JSON
        metrics_file = os.path.join(
            output_dir, f"advanced_success_prediction_metrics_{model_type}.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Created advanced success prediction model: {importance_file}")

        return {
            "importance_file": importance_file,
            "roc_file": roc_file,
            "cm_file": cm_file,
            "metrics_file": metrics_file,
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"Error creating advanced success prediction model: {e}")
        return {}


def create_advanced_analytics_report(
    results: List[Dict[str, Any]],
    output_dir: str,
    include_clustering: bool = True,
    include_prediction: bool = True,
    include_model_comparison: bool = True,
) -> Dict[str, str]:
    """
    Create a comprehensive analytics report with advanced visualizations.

    Args:
        results: List of result dictionaries
        output_dir: Directory where visualization files will be saved
        include_clustering: Whether to include clustering analysis
        include_prediction: Whether to include prediction models
        include_model_comparison: Whether to include model comparison

    Returns:
        Dictionary mapping chart names to file paths
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Create advanced analytics directory
    analytics_dir = os.path.join(output_dir, "advanced_analytics")
    ensure_output_dir(analytics_dir)

    # Extract features
    features_df = extract_features_from_results(results)

    # Initialize report files dictionary
    report_files: dict[str, Any] = {}

    # Create correlation matrix
    correlation_file = create_correlation_matrix(features_df, analytics_dir)
    if correlation_file:
        report_files["correlation_matrix"] = correlation_file

    # Create feature distribution plots
    distribution_files = create_feature_distribution_plots(features_df, analytics_dir)
    if distribution_files:
        report_files["feature_distributions"] = distribution_files

    # Create pairplot
    pairplot_file = create_pairplot(features_df, analytics_dir)
    if pairplot_file:
        report_files["pairplot"] = pairplot_file

    # Create advanced clustering if requested
    if include_clustering:
        clustering_results = create_advanced_clustering(results, analytics_dir)
        if clustering_results:
            report_files["advanced_clustering"] = clustering_results.get(
                "clustering_file", ""
            )
            report_files["cluster_analysis"] = clustering_results.get(
                "analysis_file", ""
            )

    # Create advanced model comparison if requested
    if include_model_comparison:
        model_comparison_file = create_advanced_model_comparison(results, analytics_dir)
        if model_comparison_file:
            report_files["model_comparison"] = model_comparison_file

    # Create advanced success prediction model if requested
    if include_prediction:
        # Random Forest
        rf_results = create_advanced_success_prediction_model(
            results, analytics_dir, model_type="random_forest"
        )
        if rf_results:
            report_files["rf_importance"] = rf_results.get("importance_file", "")
            report_files["rf_roc"] = rf_results.get("roc_file", "")
            report_files["rf_cm"] = rf_results.get("cm_file", "")
            report_files["rf_metrics"] = rf_results.get("metrics_file", "")

        # Gradient Boosting
        gb_results = create_advanced_success_prediction_model(
            results, analytics_dir, model_type="gradient_boosting"
        )
        if gb_results:
            report_files["gb_importance"] = gb_results.get("importance_file", "")
            report_files["gb_roc"] = gb_results.get("roc_file", "")
            report_files["gb_cm"] = gb_results.get("cm_file", "")
            report_files["gb_metrics"] = gb_results.get("metrics_file", "")

    return report_files
