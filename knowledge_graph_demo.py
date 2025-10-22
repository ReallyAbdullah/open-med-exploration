"""
knowledge_graph_demo.py
=======================

This script demonstrates how to extract simple disease–drug association
statistics from a CSV file of de‑identified patient notes using the
OpenMed models.  It is a companion to `analyze_patients.py` but focuses
on building a rudimentary knowledge graph by counting co‑occurrences of
disease and pharmaceutical entities within the same patient note.

Each note is analyzed with both the disease detection and the
pharmaceutical detection models.  For each disease mention, the
script records which drugs appear in the same note.  At the end of
processing it prints a table of the most frequently co‑occurring
disease–drug pairs.

Usage::

    python3 knowledge_graph_demo.py --file patient_notes.csv --sample-size 50

Requirements:
    - pandas
    - openmed (installed via pip)
"""

import argparse
import csv
import logging
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Suppress noisy logs from downstream libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    from openmed import analyze_text
except ImportError as e:
    raise ImportError(
        "OpenMed must be installed to run this script. Install it via `pip install openmed`."
    ) from e


def get_device() -> torch.device:
    """Return the best available torch device (MPS, CUDA or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda")
    return torch.device("cpu")


def analyze_note(
    text: str,
    device: torch.device,
) -> Tuple[List, List]:
    """
    Analyze a single note using OpenMed's disease and pharmaceutical models.

    Returns a tuple (disease_entities, pharma_entities), where each is a list
    of entity objects.  Entities include attributes `text`, `label`, `start`,
    `end` and `confidence`.
    """
    # Disease model
    disease_result = analyze_text(
        text,
        model_name="disease_detection_superclinical",
        group_entities=True,
        include_confidence=True,
        device=device,
    )
    # Pharmaceutical model
    pharma_result = analyze_text(
        text,
        model_name="pharma_detection_superclinical",
        group_entities=True,
        include_confidence=True,
        device=device,
    )
    return list(disease_result.entities), list(pharma_result.entities)


def build_association_graph(
    records: pd.DataFrame,
    *,
    sample_size: int,
    device: torch.device,
) -> Dict[str, Counter]:
    """
    Build disease–drug association counts from a subset of records.

    Parameters
    ----------
    records:
        A DataFrame containing at least a `pn_history` (text) column or a
        `description` column.
    sample_size:
        Number of records to sample for analysis.  A smaller subset can
        significantly reduce runtime when working with large datasets.
    device:
        Torch device on which to perform model inference.

    Returns
    -------
    Dict[str, Counter]
        A mapping from disease text to a Counter of drug texts and the
        number of times the pair appeared together.
    """
    # Sample the dataset
    df = records.sample(n=min(sample_size, len(records)), random_state=42)

    associations: Dict[str, Counter] = defaultdict(Counter)

    for idx, row in df.iterrows():
        text = None
        # Determine which column contains the note text
        for col in ["pn_history", "description", "text"]:
            if col in row and isinstance(row[col], str):
                text = row[col]
                break
        if not text:
            continue

        disease_entities, pharma_entities = analyze_note(text, device=device)

        # Record associations: for each disease mention, record all drugs in the note
        for d_entity in disease_entities:
            disease_name = d_entity.text
            for p_entity in pharma_entities:
                drug_name = p_entity.text
                associations[disease_name][drug_name] += 1

    return associations


def create_knowledge_graph(associations: Dict[str, Counter], min_weight: int = 1) -> nx.Graph:
    """Create a NetworkX graph from disease-drug associations."""
    G = nx.Graph()
    
    # Add nodes with different colors for diseases and drugs
    diseases = set(associations.keys())
    drugs = set(drug for counter in associations.values() for drug in counter.keys())
    
    for disease in diseases:
        G.add_node(disease, node_type='disease')
    for drug in drugs:
        G.add_node(drug, node_type='drug')
    
    # Add edges with weights
    for disease, counter in associations.items():
        for drug, count in counter.items():
            if count >= min_weight:
                G.add_edge(disease, drug, weight=count)
    
    return G

def visualize_and_save_graph(G: nx.Graph, output_path: str, title: str = "Disease-Drug Associations") -> None:
    """Visualize and save the knowledge graph with improved layout and readability."""
    plt.figure(figsize=(20, 16))
    
    # Create layout with more space and better separation
    pos = nx.kamada_kawai_layout(G)
    
    # Prepare node lists by type
    diseases = [node for node in G.nodes() if G.nodes[node].get('node_type') == 'disease']
    drugs = [node for node in G.nodes() if G.nodes[node].get('node_type') == 'drug']
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(G)
    disease_sizes = [3000 * centrality[node] + 2000 for node in diseases]
    drug_sizes = [2000 * centrality[node] + 1500 for node in drugs]
    
    # Get edge weights and normalize them
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    edge_widths = [1.5 * w/max_weight for w in edge_weights]
    
    # Set color scheme
    disease_color = '#3498db'  # Blue
    drug_color = '#2ecc71'     # Green
    edge_color = '#bdc3c7'     # Light gray
    
    # Draw edges with curved lines
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, 
                          edge_color=edge_color, style='solid')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=diseases, node_size=disease_sizes,
                          node_color=disease_color, alpha=0.7, 
                          node_shape='o', label='Diseases')
    nx.draw_networkx_nodes(G, pos, nodelist=drugs, node_size=drug_sizes,
                          node_color=drug_color, alpha=0.7, 
                          node_shape='h', label='Drugs')
    
    # Add labels with better formatting
    labels = {node: node if len(node) < 20 else node[:17] + '...' 
             for node in G.nodes()}
    
    # Draw labels with white background for better readability
    for node, (x, y) in pos.items():
        plt.text(x, y, labels[node],
                fontsize=8,
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', 
                         edgecolor='none',
                         alpha=0.7,
                         pad=2.0))
    
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    
    # Add legend with more detail
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Diseases',
                  markerfacecolor=disease_color, markersize=15, alpha=0.7),
        plt.Line2D([0], [0], marker='h', color='w', label='Drugs',
                  markerfacecolor=drug_color, markersize=15, alpha=0.7),
        plt.Line2D([0], [0], color=edge_color, label='Association',
                  linewidth=2, alpha=0.5)
    ]
    plt.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.15, 1), fontsize=12)
    
    # Add information about node sizes
    plt.figtext(0.99, 0.02, 'Node size indicates connection strength\nLarger nodes have more connections',
                ha='right', va='bottom', fontsize=10, alpha=0.7)
    
    # Save the graph with high quality
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def print_top_associations(associations: Dict[str, Counter], top_n: int = 5) -> None:
    """Print the top N drugs associated with each disease."""
    for disease, counter in associations.items():
        print(f"\nDisease: {disease}")
        for drug, count in counter.most_common(top_n):
            print(f"  - {drug}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple disease–drug association graph from patient notes.")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to a CSV file containing de‑identified patient notes. Must have a `pn_history` or `description` column.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of notes to sample for analysis.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top associated drugs to display per disease.",
    )
    parser.add_argument(
        "--min-weight",
        type=int,
        default=1,
        help="Minimum number of co-occurrences required to include an edge in the graph visualization.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="knowledge_graph.png",
        help="Path to save the visualization image.",
    )
    args = parser.parse_args()

    # Load the CSV.  We rely on pandas to handle quoting and delimiters.
    df = pd.read_csv(args.file)
    device = get_device()
    associations = build_association_graph(df, sample_size=args.sample_size, device=device)
    
    # Print associations
    print_top_associations(associations, top_n=args.top_n)
    
    # Create and save visualization
    G = create_knowledge_graph(associations, min_weight=args.min_weight)
    output_path = args.output
    title = f"Disease-Drug Associations (min. weight: {args.min_weight})"
    visualize_and_save_graph(G, output_path, title)
    print(f"\nGraph visualization saved to: {output_path}")


if __name__ == "__main__":
    main()