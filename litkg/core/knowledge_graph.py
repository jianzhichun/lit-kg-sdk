"""Knowledge Graph class for LitKG SDK."""

import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import networkx as nx
from datetime import datetime


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    label: str
    type: str
    properties: Dict[str, Any]
    confidence: float
    source: Optional[str] = None


@dataclass
class Relation:
    """Represents a relation in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    confidence: float
    source: Optional[str] = None


class KnowledgeGraph:
    """Main Knowledge Graph class with advanced features."""

    def __init__(self, session=None):
        """Initialize knowledge graph."""
        self.session = session
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "total_entities": 0,
            "total_relations": 0,
            "confidence_stats": {},
            "processing_history": []
        }
        self._communities = None
        self._temporal_data = None

    def add_entity(self, entity: Entity) -> None:
        """Add entity to the knowledge graph."""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            label=entity.label,
            entity_type=entity.type,  # Changed from 'type' to 'entity_type' to avoid NetworkX conflict
            confidence=entity.confidence,
            **entity.properties
        )
        self._update_metadata()

    def add_relation(self, relation: Relation) -> None:
        """Add relation to the knowledge graph."""
        self.relations[relation.id] = relation
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation.id,
            relation_type=relation.type,  # Changed from 'type' to 'relation_type'
            confidence=relation.confidence,
            **relation.properties
        )
        self._update_metadata()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        return self.relations.get(relation_id)

    def filter_by_confidence(self, threshold: float) -> "KnowledgeGraph":
        """Filter graph by confidence threshold."""
        filtered_kg = KnowledgeGraph(self.session)

        # Filter entities
        for entity in self.entities.values():
            if entity.confidence >= threshold:
                filtered_kg.add_entity(entity)

        # Filter relations
        for relation in self.relations.values():
            if (relation.confidence >= threshold and
                relation.source_id in filtered_kg.entities and
                relation.target_id in filtered_kg.entities):
                filtered_kg.add_relation(relation)

        return filtered_kg

    def merge_with(self, other_kg: "KnowledgeGraph") -> None:
        """Merge with another knowledge graph."""
        # Merge entities (handle duplicates)
        for entity in other_kg.entities.values():
            if entity.id not in self.entities:
                self.add_entity(entity)
            else:
                # Update confidence if higher
                existing = self.entities[entity.id]
                if entity.confidence > existing.confidence:
                    self.add_entity(entity)

        # Merge relations
        for relation in other_kg.relations.values():
            if relation.id not in self.relations:
                if (relation.source_id in self.entities and
                    relation.target_id in self.entities):
                    self.add_relation(relation)

    def get_neighbors(self, entity_id: str, depth: int = 1) -> List[str]:
        """Get neighboring entities up to specified depth."""
        if entity_id not in self.graph:
            return []

        neighbors = set()
        current_level = {entity_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
                next_level.update(self.graph.predecessors(node))
            neighbors.update(next_level)
            current_level = next_level - neighbors

        neighbors.discard(entity_id)
        return list(neighbors)

    def get_subgraph(self, entity_ids: List[str]) -> "KnowledgeGraph":
        """Extract subgraph containing specified entities."""
        subgraph_kg = KnowledgeGraph(self.session)

        # Add entities
        for entity_id in entity_ids:
            if entity_id in self.entities:
                subgraph_kg.add_entity(self.entities[entity_id])

        # Add relations between these entities
        for relation in self.relations.values():
            if (relation.source_id in entity_ids and
                relation.target_id in entity_ids):
                subgraph_kg.add_relation(relation)

        return subgraph_kg

    def analyze_communities(self) -> Dict[str, Any]:
        """Detect and analyze communities in the graph."""
        if not self._communities:
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(self.graph.to_undirected())

                communities = {}
                for node, comm_id in partition.items():
                    if comm_id not in communities:
                        communities[comm_id] = []
                    communities[comm_id].append(node)

                self._communities = {
                    "partition": partition,
                    "communities": communities,
                    "modularity": community_louvain.modularity(partition, self.graph.to_undirected()),
                    "num_communities": len(communities)
                }
            except ImportError:
                # Fallback to basic connected components
                communities = list(nx.connected_components(self.graph.to_undirected()))
                self._communities = {
                    "communities": {i: list(comm) for i, comm in enumerate(communities)},
                    "num_communities": len(communities),
                    "modularity": None
                }

        return self._communities

    def track_knowledge_evolution(self) -> Dict[str, Any]:
        """Track temporal evolution of knowledge."""
        if not self._temporal_data:
            # Basic temporal analysis based on entity sources
            evolution = {}
            for entity in self.entities.values():
                source = entity.source or "unknown"
                if source not in evolution:
                    evolution[source] = {"entities": 0, "types": set()}
                evolution[source]["entities"] += 1
                evolution[source]["types"].add(entity.type)

            # Convert sets to lists for serialization
            for source_data in evolution.values():
                source_data["types"] = list(source_data["types"])

            self._temporal_data = evolution

        return self._temporal_data

    def collaborate_interactively(self) -> None:
        """Launch interactive collaboration interface."""
        try:
            from ..human_loop.jupyter_widgets import JupyterInterface
            interface = JupyterInterface(self)
            interface.launch()
        except ImportError:
            print("Jupyter widgets not available. Install with: pip install lit-kg-sdk[jupyter]")
            self._fallback_interactive_mode()

    def _fallback_interactive_mode(self) -> None:
        """Fallback interactive mode for command line."""
        print(f"\n📊 Knowledge Graph Summary:")
        print(f"  Entities: {len(self.entities)}")
        print(f"  Relations: {len(self.relations)}")

        print(f"\n🎯 Entity Types:")
        entity_types = {}
        for entity in self.entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        for etype, count in sorted(entity_types.items()):
            print(f"  {etype}: {count}")

        print(f"\n🔗 Relation Types:")
        relation_types = {}
        for relation in self.relations.values():
            relation_types[relation.type] = relation_types.get(relation.type, 0) + 1
        for rtype, count in sorted(relation_types.items()):
            print(f"  {rtype}: {count}")

        # Basic validation prompts
        while True:
            action = input("\n🤖 Actions: (v)iew entities, (f)ilter by confidence, (s)ave, (q)uit: ").lower()
            if action == 'q':
                break
            elif action == 'v':
                self._display_entities(limit=10)
            elif action == 'f':
                threshold = float(input("Enter confidence threshold (0-1): "))
                filtered = self.filter_by_confidence(threshold)
                print(f"Filtered to {len(filtered.entities)} entities and {len(filtered.relations)} relations")
            elif action == 's':
                filename = input("Enter filename (with extension): ")
                self.export(filename)
                print(f"Saved to {filename}")

    def _display_entities(self, limit: int = 10) -> None:
        """Display top entities."""
        print(f"\n📋 Top {limit} Entities:")
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda x: x.confidence,
            reverse=True
        )[:limit]

        for i, entity in enumerate(sorted_entities, 1):
            print(f"  {i}. {entity.label} ({entity.type}) - {entity.confidence:.2f}")

    def visualize(self, layout: str = "spring", figsize=(12, 8), **kwargs) -> None:
        """Visualize the knowledge graph."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Check if graph is empty
            if not self.entities or not self.graph.nodes():
                print("⚠️ Knowledge graph is empty. Add some entities first.")
                return

            plt.figure(figsize=figsize)

            # Extract layout-specific kwargs and figsize
            layout_kwargs = {k: v for k, v in kwargs.items() if k != 'figsize'}

            if layout == "spring":
                pos = nx.spring_layout(self.graph, **layout_kwargs)
            elif layout == "circular":
                pos = nx.circular_layout(self.graph)
            elif layout == "random":
                pos = nx.random_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph)

            # Draw nodes with better visibility
            node_colors = [self.entities[node].confidence for node in self.graph.nodes()]
            node_sizes = [300 + (self.entities[node].confidence * 200) for node in self.graph.nodes()]

            # Create the plot
            nodes = nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                                         cmap=plt.cm.viridis, node_size=node_sizes, alpha=0.8)

            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, alpha=0.6, width=2, edge_color='gray')

            # Draw labels
            labels = {node: self.entities[node].label[:15] + "..." if len(self.entities[node].label) > 15 else self.entities[node].label
                     for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=10, font_weight='bold')

            plt.title(f"Knowledge Graph ({len(self.entities)} entities, {len(self.relations)} relations)")
            plt.axis('off')

            # Add colorbar for confidence scores (only if we have nodes)
            if len(node_colors) > 0:
                plt.colorbar(nodes, label='Confidence Score', shrink=0.8)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")

    def export(self, filepath: str, format: Optional[str] = None) -> None:
        """Export knowledge graph to various formats."""
        path = Path(filepath)

        if format is None:
            format = path.suffix.lower().lstrip('.')

        if format in ['json']:
            self._export_json(path)
        elif format in ['graphml']:
            self._export_graphml(path)
        elif format in ['neo4j', 'cypher']:
            self._export_neo4j(path)
        elif format in ['csv']:
            self._export_csv(path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, path: Path) -> None:
        """Export to JSON format."""
        data = {
            "metadata": self.metadata,
            "entities": [
                {
                    "id": e.id,
                    "label": e.label,
                    "type": e.type,
                    "properties": e.properties,
                    "confidence": e.confidence,
                    "source": e.source
                } for e in self.entities.values()
            ],
            "relations": [
                {
                    "id": r.id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.type,
                    "properties": r.properties,
                    "confidence": r.confidence,
                    "source": r.source
                } for r in self.relations.values()
            ]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_graphml(self, path: Path) -> None:
        """Export to GraphML format."""
        nx.write_graphml(self.graph, path)

    def _export_neo4j(self, path: Path) -> None:
        """Export to Neo4j Cypher format."""
        with open(path, 'w', encoding='utf-8') as f:
            # Create entities
            for entity in self.entities.values():
                props = ", ".join([f"{k}: '{v}'" for k, v in entity.properties.items()])
                f.write(f"CREATE (:{entity.type} {{id: '{entity.id}', "
                       f"label: '{entity.label}', confidence: {entity.confidence}")
                if props:
                    f.write(f", {props}")
                f.write("});\n")

            # Create relations
            for relation in self.relations.values():
                props = ", ".join([f"{k}: '{v}'" for k, v in relation.properties.items()])
                f.write(f"MATCH (a {{id: '{relation.source_id}'}}), "
                       f"(b {{id: '{relation.target_id}'}}) "
                       f"CREATE (a)-[:{relation.type} {{confidence: {relation.confidence}")
                if props:
                    f.write(f", {props}")
                f.write("}]->(b);\n")

    def _export_csv(self, path: Path) -> None:
        """Export to CSV format (entities and relations)."""
        import pandas as pd

        # Export entities
        entities_data = []
        for entity in self.entities.values():
            row = {
                "id": entity.id,
                "label": entity.label,
                "type": entity.type,
                "confidence": entity.confidence,
                "source": entity.source
            }
            row.update(entity.properties)
            entities_data.append(row)

        entities_df = pd.DataFrame(entities_data)
        entities_df.to_csv(path.with_suffix('.entities.csv'), index=False)

        # Export relations
        relations_data = []
        for relation in self.relations.values():
            row = {
                "id": relation.id,
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "type": relation.type,
                "confidence": relation.confidence,
                "source": relation.source
            }
            row.update(relation.properties)
            relations_data.append(row)

        relations_df = pd.DataFrame(relations_data)
        relations_df.to_csv(path.with_suffix('.relations.csv'), index=False)

    def _update_metadata(self) -> None:
        """Update graph metadata."""
        self.metadata["total_entities"] = len(self.entities)
        self.metadata["total_relations"] = len(self.relations)
        self.metadata["last_updated"] = datetime.now().isoformat()

        # Update confidence statistics
        if self.entities:
            confidences = [e.confidence for e in self.entities.values()]
            self.metadata["confidence_stats"] = {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences)
            }

    def __len__(self) -> int:
        """Return total number of entities and relations."""
        return len(self.entities) + len(self.relations)

    def __str__(self) -> str:
        """String representation of the knowledge graph."""
        return (f"KnowledgeGraph(entities={len(self.entities)}, "
                f"relations={len(self.relations)})")

    def __repr__(self) -> str:
        """Detailed representation of the knowledge graph."""
        return self.__str__()