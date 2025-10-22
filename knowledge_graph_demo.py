"""
knowledge_graph_demo.py
=======================

This script demonstrates how to extract simple disease–drug association
statistics from a CSV file of de‑identified patient notes using the
OpenMed models.  It is a companion to `analyze_patients.py` but focuses
on building a knowledge graph by counting co‑occurrences of disease and
pharmaceutical entities within the same patient note. The script now
produces both a high‑fidelity static export and an optional interactive
HTML dashboard to accelerate drug repurposing, safety and treatment
pathway exploration.

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
import json
import logging
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch

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
    
    # Annotate nodes with metadata that is useful for downstream visualisations
    weighted_degrees = _calculate_weighted_degree(G)
    nx.set_node_attributes(G, weighted_degrees, name="weighted_degree")

    top_partners = _calculate_top_partners(G)
    nx.set_node_attributes(G, top_partners, name="top_partners")

    return G


def _calculate_weighted_degree(graph: nx.Graph) -> Dict[str, float]:
    """Return the weighted degree for every node in the graph."""

    return {
        node: sum(graph[node][neighbor]["weight"] for neighbor in graph.neighbors(node))
        for node in graph.nodes
    }


def _calculate_top_partners(graph: nx.Graph, limit: int = 5) -> Dict[str, List[Tuple[str, int]]]:
    """Return the most strongly connected neighbours for each node."""

    result: Dict[str, List[Tuple[str, int]]] = {}
    for node in graph.nodes:
        neighbors = [
            (neighbor, graph[node][neighbor]["weight"])
            for neighbor in graph.neighbors(node)
        ]
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        result[node] = neighbors[:limit]
    return result

def visualize_and_save_graph(
    G: nx.Graph,
    output_path: str,
    title: str = "Disease-Drug Associations",
) -> None:
    """Visualize and save the knowledge graph with improved layout and readability."""

    if G.number_of_nodes() == 0:
        logging.warning("The graph is empty. Skipping static export.")
        return

    fig, ax = plt.subplots(figsize=(22, 16))

    # Layout that separates diseases and drugs while remaining force-directed
    pos = nx.spring_layout(G, k=1.2, seed=42, weight="weight")
    _spread_bipartite_layers(G, pos)

    diseases = [node for node in G.nodes if G.nodes[node].get("node_type") == "disease"]
    drugs = [node for node in G.nodes if G.nodes[node].get("node_type") == "drug"]

    weighted_degree = nx.get_node_attributes(G, "weighted_degree")
    disease_sizes = _scale_values([weighted_degree.get(node, 0) for node in diseases], base=1600, spread=2600)
    drug_sizes = _scale_values([weighted_degree.get(node, 0) for node in drugs], base=1200, spread=2200)

    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [1.0 + 5.0 * (weight / max_weight) for weight in edge_weights]
        edge_colors = [weight / max_weight for weight in edge_weights]
    else:
        max_weight = 1
        edge_widths = []
        edge_colors = []

    cmap = plt.cm.get_cmap("YlGnBu")

    if edge_widths:
        nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=cmap,
            alpha=0.55,
            ax=ax,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=diseases,
        node_size=disease_sizes,
        node_color="#1f77b4",
        alpha=0.85,
        node_shape="o",
        label="Diseases",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=drugs,
        node_size=drug_sizes,
        node_color="#2ecc71",
        alpha=0.85,
        node_shape="h",
        label="Drugs",
        ax=ax,
    )

    labels = _format_labels(G.nodes, max_length=28)
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            labels[node],
            fontsize=9,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2.4),
        )

    ax.set_title(title, fontsize=20, pad=24)
    ax.axis("off")

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Diseases",
            markerfacecolor="#1f77b4",
            markersize=16,
            alpha=0.85,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="h",
            color="w",
            label="Drugs",
            markerfacecolor="#2ecc71",
            markersize=16,
            alpha=0.85,
        ),
        plt.Line2D(
            [0],
            [0],
            color=cmap(0.8),
            label="Stronger associations",
            linewidth=3,
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.2, 1.0), fontsize=12)

    fig.text(
        0.01,
        0.01,
        "Node size ∝ weighted connections. Highlight dense subgraphs for repurposing, safety and pathway insights.",
        ha="left",
        va="bottom",
        fontsize=11,
        alpha=0.75,
    )

    if edge_colors:
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
        cbar.set_label("Relative association strength", rotation=270, labelpad=18)

    fig.tight_layout()
    fig.savefig(output_path, dpi=350, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def _spread_bipartite_layers(G: nx.Graph, pos: Dict[str, List[float]]) -> None:
    """Nudge diseases and drugs to opposite sides for clarity."""

    diseases = [node for node in G.nodes if G.nodes[node].get("node_type") == "disease"]
    drugs = [node for node in G.nodes if G.nodes[node].get("node_type") == "drug"]

    for node in diseases:
        pos[node][0] -= 0.25
    for node in drugs:
        pos[node][0] += 0.25


def _scale_values(values: Iterable[float], *, base: float, spread: float) -> List[float]:
    if not values:
        return []
    max_value = max(values)
    if max_value == 0:
        return [base for _ in values]
    return [base + (value / max_value) * spread for value in values]


def _format_labels(nodes: Iterable[str], max_length: int) -> Dict[str, str]:
    formatted = {}
    for node in nodes:
        node_str = str(node)
        if len(node_str) <= max_length:
            formatted[node] = node_str
        else:
            formatted[node] = node_str[: max_length - 1] + "…"
    return formatted


LONG_TERM_GOALS = [
    "Drug repurposing",
    "Adverse drug reactions (ADR) & contraindications",
    "Comorbidity & phenotype modules",
    "Early warning & surveillance",
    "Treatment pathway recommendation",
]


def export_interactive_graph(
    G: nx.Graph,
    html_path: str,
    title: str,
    long_term_goals: Iterable[str] = LONG_TERM_GOALS,
) -> None:
    """Generate an interactive PyVis export highlighting rich metadata."""

    if G.number_of_nodes() == 0:
        logging.warning("The graph is empty. Skipping interactive export.")
        return

    try:
        from pyvis.network import Network
    except ImportError as exc:  # pragma: no cover - dependency optional during tests
        raise ImportError(
            "PyVis is required for interactive exports. Install it via `pip install pyvis`."
        ) from exc

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50",
        cdn_resources="remote",
    )
    net.heading = title
    net.barnes_hut()
    net.show_buttons(filter_=["physics", "interaction"])

    weighted_degree = nx.get_node_attributes(G, "weighted_degree")
    top_partners = nx.get_node_attributes(G, "top_partners")

    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "entity")
        degree_weight = weighted_degree.get(node, 0)
        partner_list = top_partners.get(node, [])
        partner_html = "".join(
            f"<li><strong>{partner}</strong>: {weight} co-occurrences</li>" for partner, weight in partner_list
        )
        tooltip = (
            f"<h4>{node}</h4>"
            f"<p><b>Type:</b> {node_type.title()}<br/>"
            f"<b>Total associations:</b> {int(degree_weight)}</p>"
            f"<p><b>Top partners</b></p><ul>{partner_html or '<li>No partners</li>'}</ul>"
        )

        shape = "ellipse" if node_type == "disease" else "hexagon"
        color = "#1f77b4" if node_type == "disease" else "#2ecc71"
        net.add_node(
            node,
            label=node,
            title=tooltip,
            shape=shape,
            color=color,
            value=1 + degree_weight,
        )

    for source, target, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(
            source,
            target,
            value=weight,
            title=f"{weight} shared mentions",
            color="#7f8c8d",
        )

    net.set_options(
        json.dumps(
            {
                "nodes": {
                    "font": {"size": 16},
                    "shadow": True,
                },
                "edges": {
                    "smooth": {"type": "dynamic"},
                    "color": {"inherit": False},
                },
                "interaction": {
                    "hover": True,
                    "multiselect": True,
                    "tooltipDelay": 120,
                },
                "physics": {
                    "barnesHut": {
                        "springLength": 180,
                        "avoidOverlap": 0.25,
                    }
                },
            }
        )
    )

    html_path = Path(html_path)
    net.save_graph(str(html_path))

    if long_term_goals:
        html = html_path.read_text(encoding="utf-8")
        goals_list = "".join(f"<li>{goal}</li>" for goal in long_term_goals)
        annotation = f"""
<section style='padding: 16px 24px 32px; font-family: "Helvetica Neue", Arial, sans-serif;'>
  <h2 style='margin-bottom: 8px;'>Strategic Knowledge Graph Outcomes</h2>
  <p style='max-width: 720px;'>The interactive network supports downstream discovery workflows by surfacing
  candidate disease–drug relationships. Use it to prioritise hypotheses across:</p>
  <ul style='columns: 2; margin-top: 8px;'>{goals_list}</ul>
</section>
"""
        html = html.replace("</body>", f"{annotation}</body>")
        html_path.write_text(html, encoding="utf-8")

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
    parser.add_argument(
        "--html-output",
        type=str,
        default=None,
        help="Optional path to export an interactive HTML knowledge graph (requires pyvis).",
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

    if args.html_output:
        export_interactive_graph(G, args.html_output, title)
        print(f"Interactive graph exported to: {args.html_output}")


if __name__ == "__main__":
    main()
