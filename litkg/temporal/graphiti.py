"""Graphiti integration for temporal knowledge graphs."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class TemporalEdge:
    """Temporal edge with time awareness."""
    source: str
    target: str
    relation_type: str
    timestamp: datetime
    properties: Dict[str, Any]
    confidence: float
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None


@dataclass
class TemporalNode:
    """Temporal node with evolution tracking."""
    id: str
    label: str
    node_type: str
    timestamp: datetime
    properties: Dict[str, Any]
    confidence: float
    version: int = 1
    previous_version: Optional[str] = None


class TemporalKG:
    """
    Temporal Knowledge Graph with Graphiti integration.

    Features:
    - Time-aware relationship tracking
    - Point-in-time queries
    - Knowledge evolution analysis
    - Episodic processing
    """

    def __init__(self, config=None):
        """Initialize temporal knowledge graph."""
        self.config = config or {}
        self.graphiti_client = None
        self.temporal_nodes: Dict[str, List[TemporalNode]] = {}
        self.temporal_edges: List[TemporalEdge] = []
        self.episodes: List[Dict[str, Any]] = []
        self._initialize_graphiti()

    def _initialize_graphiti(self):
        """Initialize Graphiti client."""
        try:
            # Note: This is a placeholder for actual Graphiti integration
            # The real implementation would use the actual Graphiti library
            from graphiti_core import Graphiti
            self.graphiti_client = Graphiti(
                uri=self.config.get("graphiti_uri", "memory://"),
                api_key=self.config.get("graphiti_api_key")
            )
            logger.info("Graphiti client initialized")
        except ImportError:
            logger.warning("Graphiti not available. Using local temporal implementation.")
            self.graphiti_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            self.graphiti_client = None

    def is_available(self) -> bool:
        """Check if temporal features are available."""
        return True  # Local implementation always available

    async def add_temporal_node(self,
                              node_id: str,
                              label: str,
                              node_type: str,
                              properties: Dict[str, Any],
                              confidence: float,
                              timestamp: Optional[datetime] = None) -> None:
        """Add temporal node with version tracking."""

        if timestamp is None:
            timestamp = datetime.now()

        # Check if node already exists
        existing_versions = self.temporal_nodes.get(node_id, [])
        version = len(existing_versions) + 1
        previous_version = existing_versions[-1].id if existing_versions else None

        temporal_node = TemporalNode(
            id=f"{node_id}_v{version}",
            label=label,
            node_type=node_type,
            timestamp=timestamp,
            properties=properties,
            confidence=confidence,
            version=version,
            previous_version=previous_version
        )

        # Store locally
        if node_id not in self.temporal_nodes:
            self.temporal_nodes[node_id] = []
        self.temporal_nodes[node_id].append(temporal_node)

        # Store in Graphiti if available
        if self.graphiti_client:
            try:
                await self.graphiti_client.add_node(
                    node_id=temporal_node.id,
                    properties={
                        "label": label,
                        "type": node_type,
                        "timestamp": timestamp.isoformat(),
                        "confidence": confidence,
                        "version": version,
                        **properties
                    }
                )
            except Exception as e:
                logger.error(f"Error adding node to Graphiti: {e}")

    async def add_temporal_edge(self,
                              source_id: str,
                              target_id: str,
                              relation_type: str,
                              properties: Dict[str, Any],
                              confidence: float,
                              timestamp: Optional[datetime] = None,
                              valid_from: Optional[datetime] = None,
                              valid_to: Optional[datetime] = None) -> None:
        """Add temporal edge with validity period."""

        if timestamp is None:
            timestamp = datetime.now()

        temporal_edge = TemporalEdge(
            source=source_id,
            target=target_id,
            relation_type=relation_type,
            timestamp=timestamp,
            properties=properties,
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to
        )

        # Store locally
        self.temporal_edges.append(temporal_edge)

        # Store in Graphiti if available
        if self.graphiti_client:
            try:
                await self.graphiti_client.add_edge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties={
                        "timestamp": timestamp.isoformat(),
                        "confidence": confidence,
                        "valid_from": valid_from.isoformat() if valid_from else None,
                        "valid_to": valid_to.isoformat() if valid_to else None,
                        **properties
                    }
                )
            except Exception as e:
                logger.error(f"Error adding edge to Graphiti: {e}")

    async def query_point_in_time(self,
                                query_time: datetime,
                                entity_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query knowledge graph state at specific point in time."""

        # Get nodes valid at query time
        valid_nodes = []
        for node_id, versions in self.temporal_nodes.items():
            if entity_ids and node_id not in entity_ids:
                continue

            # Find the version valid at query time
            valid_version = None
            for version in sorted(versions, key=lambda x: x.timestamp):
                if version.timestamp <= query_time:
                    valid_version = version
                else:
                    break

            if valid_version:
                valid_nodes.append(valid_version)

        # Get edges valid at query time
        valid_edges = []
        for edge in self.temporal_edges:
            if edge.timestamp <= query_time:
                # Check validity period
                if edge.valid_from and edge.valid_from > query_time:
                    continue
                if edge.valid_to and edge.valid_to < query_time:
                    continue

                # Check if both nodes exist at query time
                source_exists = any(n.id.startswith(edge.source) for n in valid_nodes)
                target_exists = any(n.id.startswith(edge.target) for n in valid_nodes)

                if source_exists and target_exists:
                    valid_edges.append(edge)

        return {
            "query_time": query_time.isoformat(),
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.node_type,
                    "properties": node.properties,
                    "confidence": node.confidence,
                    "version": node.version
                } for node in valid_nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.relation_type,
                    "properties": edge.properties,
                    "confidence": edge.confidence
                } for edge in valid_edges
            ]
        }

    async def track_entity_evolution(self, entity_id: str) -> Dict[str, Any]:
        """Track how an entity evolved over time."""

        if entity_id not in self.temporal_nodes:
            return {"entity_id": entity_id, "evolution": [], "total_versions": 0}

        versions = sorted(self.temporal_nodes[entity_id], key=lambda x: x.timestamp)

        evolution = []
        for i, version in enumerate(versions):
            evolution_entry = {
                "version": version.version,
                "timestamp": version.timestamp.isoformat(),
                "label": version.label,
                "confidence": version.confidence,
                "changes": []
            }

            # Compare with previous version
            if i > 0:
                prev_version = versions[i - 1]
                changes = self._compute_changes(prev_version, version)
                evolution_entry["changes"] = changes

            evolution.append(evolution_entry)

        # Get related edges for each time period
        related_edges = [
            edge for edge in self.temporal_edges
            if edge.source == entity_id or edge.target == entity_id
        ]

        return {
            "entity_id": entity_id,
            "evolution": evolution,
            "total_versions": len(versions),
            "related_edges": len(related_edges),
            "first_seen": versions[0].timestamp.isoformat() if versions else None,
            "last_updated": versions[-1].timestamp.isoformat() if versions else None
        }

    async def create_episode(self,
                           episode_data: Dict[str, Any],
                           timestamp: Optional[datetime] = None) -> str:
        """Create episodic processing entry."""

        if timestamp is None:
            timestamp = datetime.now()

        episode_id = f"episode_{len(self.episodes) + 1}_{int(timestamp.timestamp())}"

        episode = {
            "id": episode_id,
            "timestamp": timestamp.isoformat(),
            "data": episode_data,
            "entities_added": episode_data.get("entities_count", 0),
            "relations_added": episode_data.get("relations_count", 0),
            "source": episode_data.get("source", "unknown")
        }

        self.episodes.append(episode)

        # Store in Graphiti if available
        if self.graphiti_client:
            try:
                await self.graphiti_client.create_episode(episode)
            except Exception as e:
                logger.error(f"Error creating episode in Graphiti: {e}")

        return episode_id

    async def get_knowledge_timeline(self,
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get timeline of knowledge evolution."""

        if start_time is None:
            start_time = min(
                (min(versions, key=lambda v: v.timestamp).timestamp
                 for versions in self.temporal_nodes.values() if versions),
                default=datetime.now()
            )

        if end_time is None:
            end_time = datetime.now()

        # Collect all temporal events
        events = []

        # Node events
        for node_id, versions in self.temporal_nodes.items():
            for version in versions:
                if start_time <= version.timestamp <= end_time:
                    events.append({
                        "type": "node_update",
                        "timestamp": version.timestamp,
                        "entity_id": node_id,
                        "version": version.version,
                        "label": version.label,
                        "confidence": version.confidence
                    })

        # Edge events
        for edge in self.temporal_edges:
            if start_time <= edge.timestamp <= end_time:
                events.append({
                    "type": "relation_added",
                    "timestamp": edge.timestamp,
                    "source": edge.source,
                    "target": edge.target,
                    "relation_type": edge.relation_type,
                    "confidence": edge.confidence
                })

        # Episode events
        for episode in self.episodes:
            episode_time = datetime.fromisoformat(episode["timestamp"])
            if start_time <= episode_time <= end_time:
                events.append({
                    "type": "episode",
                    "timestamp": episode_time,
                    "episode_id": episode["id"],
                    "source": episode["source"],
                    "entities_added": episode["entities_added"],
                    "relations_added": episode["relations_added"]
                })

        # Sort events by timestamp
        events.sort(key=lambda x: x["timestamp"])

        # Create timeline summary
        timeline_stats = {
            "total_events": len(events),
            "node_updates": len([e for e in events if e["type"] == "node_update"]),
            "relation_additions": len([e for e in events if e["type"] == "relation_added"]),
            "episodes": len([e for e in events if e["type"] == "episode"]),
            "time_span": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_days": (end_time - start_time).days
            }
        }

        return {
            "timeline": [
                {
                    **event,
                    "timestamp": event["timestamp"].isoformat()
                } for event in events
            ],
            "stats": timeline_stats
        }

    def _compute_changes(self, prev_version: TemporalNode, curr_version: TemporalNode) -> List[Dict[str, Any]]:
        """Compute changes between two versions of a node."""
        changes = []

        # Label change
        if prev_version.label != curr_version.label:
            changes.append({
                "type": "label_change",
                "from": prev_version.label,
                "to": curr_version.label
            })

        # Confidence change
        if abs(prev_version.confidence - curr_version.confidence) > 0.01:
            changes.append({
                "type": "confidence_change",
                "from": prev_version.confidence,
                "to": curr_version.confidence
            })

        # Property changes
        prev_props = set(prev_version.properties.keys())
        curr_props = set(curr_version.properties.keys())

        # Added properties
        for prop in curr_props - prev_props:
            changes.append({
                "type": "property_added",
                "property": prop,
                "value": curr_version.properties[prop]
            })

        # Removed properties
        for prop in prev_props - curr_props:
            changes.append({
                "type": "property_removed",
                "property": prop,
                "value": prev_version.properties[prop]
            })

        # Modified properties
        for prop in prev_props & curr_props:
            if prev_version.properties[prop] != curr_version.properties[prop]:
                changes.append({
                    "type": "property_modified",
                    "property": prop,
                    "from": prev_version.properties[prop],
                    "to": curr_version.properties[prop]
                })

        return changes

    async def export_temporal_data(self, filepath: str) -> None:
        """Export temporal knowledge graph data."""
        temporal_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_nodes": sum(len(versions) for versions in self.temporal_nodes.values()),
                "total_edges": len(self.temporal_edges),
                "total_episodes": len(self.episodes)
            },
            "temporal_nodes": {
                node_id: [
                    {
                        "id": version.id,
                        "label": version.label,
                        "type": version.node_type,
                        "timestamp": version.timestamp.isoformat(),
                        "properties": version.properties,
                        "confidence": version.confidence,
                        "version": version.version,
                        "previous_version": version.previous_version
                    } for version in versions
                ] for node_id, versions in self.temporal_nodes.items()
            },
            "temporal_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation_type": edge.relation_type,
                    "timestamp": edge.timestamp.isoformat(),
                    "properties": edge.properties,
                    "confidence": edge.confidence,
                    "valid_from": edge.valid_from.isoformat() if edge.valid_from else None,
                    "valid_to": edge.valid_to.isoformat() if edge.valid_to else None
                } for edge in self.temporal_edges
            ],
            "episodes": self.episodes
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(temporal_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Temporal data exported to {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        """Get temporal knowledge graph statistics."""
        unique_entities = len(self.temporal_nodes)
        total_versions = sum(len(versions) for versions in self.temporal_nodes.values())

        if self.temporal_edges:
            edge_timestamps = [edge.timestamp for edge in self.temporal_edges]
            first_edge = min(edge_timestamps)
            last_edge = max(edge_timestamps)
        else:
            first_edge = last_edge = None

        return {
            "unique_entities": unique_entities,
            "total_node_versions": total_versions,
            "total_edges": len(self.temporal_edges),
            "total_episodes": len(self.episodes),
            "average_versions_per_entity": total_versions / max(unique_entities, 1),
            "temporal_span": {
                "first_edge": first_edge.isoformat() if first_edge else None,
                "last_edge": last_edge.isoformat() if last_edge else None,
                "span_days": (last_edge - first_edge).days if first_edge and last_edge else 0
            }
        }