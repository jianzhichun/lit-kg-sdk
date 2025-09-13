"""Neo4j LLM Graph Builder integration for LitKG SDK."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class Neo4jBuilder:
    """
    Neo4j LLM Graph Builder integration with latest features.

    Supports:
    - Community summaries with Leiden clustering
    - Parallel retrievers (global + local)
    - Custom prompt instructions
    - Multiple LLM providers
    """

    def __init__(self, config):
        """Initialize Neo4j builder."""
        self.config = config
        self.driver = None
        self.gds = None
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize Neo4j connection."""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                encrypted=True
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")

            # Try to initialize GDS for community detection
            try:
                from graphdatascience import GraphDataScience
                self.gds = GraphDataScience(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password)
                )
                logger.info("Neo4j Graph Data Science initialized")
            except ImportError:
                logger.warning("GraphDataScience not available. Community features limited.")

        except ImportError:
            logger.error("Neo4j driver not found. Install with: pip install neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")

    def is_available(self) -> bool:
        """Check if Neo4j is available."""
        return self.driver is not None

    def create_graph_from_kg(self, knowledge_graph) -> Dict[str, Any]:
        """Create Neo4j graph from KnowledgeGraph object."""
        if not self.is_available():
            raise RuntimeError("Neo4j not available")

        stats = {"nodes_created": 0, "relationships_created": 0, "communities": 0}

        try:
            with self.driver.session() as session:
                # Clear existing data (optional)
                if self.config.clear_existing:
                    session.run("MATCH (n) DETACH DELETE n")

                # Create nodes
                for entity in knowledge_graph.entities.values():
                    result = session.run(
                        f"""
                        CREATE (n:{entity.type} {{
                            id: $id,
                            label: $label,
                            confidence: $confidence,
                            source: $source
                        }})
                        SET n += $properties
                        RETURN n
                        """,
                        id=entity.id,
                        label=entity.label,
                        confidence=entity.confidence,
                        source=entity.source,
                        properties=entity.properties
                    )
                    stats["nodes_created"] += 1

                # Create relationships
                for relation in knowledge_graph.relations.values():
                    result = session.run(
                        f"""
                        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                        CREATE (a)-[r:{relation.type} {{
                            confidence: $confidence,
                            source: $source
                        }}]->(b)
                        SET r += $properties
                        RETURN r
                        """,
                        source_id=relation.source_id,
                        target_id=relation.target_id,
                        confidence=relation.confidence,
                        source=relation.source,
                        properties=relation.properties
                    )
                    stats["relationships_created"] += 1

                # Generate community summaries if enabled
                if self.config.enable_communities and self.gds:
                    community_stats = self._generate_community_summaries(session)
                    stats.update(community_stats)

            logger.info(f"Created {stats['nodes_created']} nodes and {stats['relationships_created']} relationships")
            return stats

        except Exception as e:
            logger.error(f"Error creating Neo4j graph: {e}")
            raise

    def _generate_community_summaries(self, session) -> Dict[str, Any]:
        """Generate community summaries using Leiden algorithm."""
        try:
            # Create in-memory graph for GDS
            session.run("""
                CALL gds.graph.project(
                    'knowledge-graph',
                    {Node: {label: '*'}},
                    {Relationship: {type: '*', orientation: 'UNDIRECTED'}}
                )
            """)

            # Run Leiden community detection
            result = session.run("""
                CALL gds.leiden.stream('knowledge-graph')
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId) AS node, communityId
            """)

            # Group nodes by community
            communities = {}
            for record in result:
                node = record["node"]
                community_id = record["communityId"]

                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append({
                    "id": node["id"],
                    "label": node["label"],
                    "type": list(node.labels)[0]
                })

            # Generate summaries for each community using LLM
            community_summaries = {}
            for comm_id, nodes in communities.items():
                if len(nodes) >= 3:  # Only summarize meaningful communities
                    summary = self._generate_community_summary(nodes)
                    community_summaries[comm_id] = summary

                    # Create community summary node
                    session.run("""
                        CREATE (c:CommunityNode {
                            id: $id,
                            summary: $summary,
                            size: $size,
                            community_id: $community_id
                        })
                    """,
                    id=f"community_{comm_id}",
                    summary=summary,
                    size=len(nodes),
                    community_id=comm_id)

            # Clean up GDS graph
            session.run("CALL gds.graph.drop('knowledge-graph')")

            return {
                "communities": len(communities),
                "community_summaries": len(community_summaries)
            }

        except Exception as e:
            logger.error(f"Error generating community summaries: {e}")
            return {"communities": 0, "community_summaries": 0}

    def _generate_community_summary(self, nodes: List[Dict[str, Any]]) -> str:
        """Generate summary for a community of nodes using LLM."""
        try:
            # Create prompt for LLM
            node_descriptions = []
            for node in nodes:
                node_descriptions.append(f"- {node['label']} ({node['type']})")

            prompt = f"""
            Analyze the following group of related entities and provide a concise summary (2-3 sentences) describing their common theme or relationship:

            Entities:
            {chr(10).join(node_descriptions)}

            Summary:
            """

            # This would use the session's LLM provider
            # For now, return a simple summary
            entity_types = set(node['type'] for node in nodes)
            entity_labels = [node['label'] for node in nodes[:5]]  # Top 5

            if len(entity_types) == 1:
                type_name = list(entity_types)[0]
                summary = f"A cluster of {len(nodes)} {type_name} entities including {', '.join(entity_labels[:3])}"
                if len(entity_labels) > 3:
                    summary += f" and {len(entity_labels) - 3} others"
            else:
                summary = f"A diverse cluster of {len(nodes)} entities spanning {', '.join(entity_types)} including {', '.join(entity_labels[:3])}"
                if len(entity_labels) > 3:
                    summary += f" and {len(entity_labels) - 3} others"

            return summary

        except Exception as e:
            logger.error(f"Error generating community summary: {e}")
            return f"Community of {len(nodes)} entities"

    def setup_parallel_retrievers(self) -> Dict[str, Any]:
        """Setup parallel retrievers for enhanced search."""
        if not self.is_available():
            raise RuntimeError("Neo4j not available")

        try:
            with self.driver.session() as session:
                # Create vector index for global retrieval
                session.run("""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (n:Node) ON (n.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)

                # Create full-text index for local retrieval
                session.run("""
                    CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
                    FOR (n) ON EACH [n.label, n.description]
                """)

                # Create community index
                session.run("""
                    CREATE INDEX community_index IF NOT EXISTS
                    FOR (n:CommunityNode) ON (n.community_id)
                """)

            return {"status": "success", "indexes": ["vector", "fulltext", "community"]}

        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
            return {"status": "error", "message": str(e)}

    async def global_community_retrieval(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Global community retrieval using vector similarity."""
        if not self.is_available():
            return []

        try:
            # This would normally generate embeddings for the query
            # and search using vector similarity
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:CommunityNode)
                    WHERE c.summary CONTAINS $query
                    RETURN c.id as id, c.summary as summary, c.size as size
                    ORDER BY c.size DESC
                    LIMIT $limit
                """, query=query, limit=limit)

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Error in global retrieval: {e}")
            return []

    async def local_entity_retrieval(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Local entity retrieval using full-text search."""
        if not self.is_available():
            return []

        try:
            with self.driver.session() as session:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
                    YIELD node, score
                    RETURN node.id as id, node.label as label,
                           labels(node)[0] as type, score
                    ORDER BY score DESC
                    LIMIT $limit
                """, query=query, limit=limit)

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Error in local retrieval: {e}")
            return []

    async def parallel_retrieval(self, query: str) -> Dict[str, Any]:
        """Run global and local retrievers in parallel."""
        try:
            # Run both retrievers concurrently
            global_task = asyncio.create_task(self.global_community_retrieval(query))
            local_task = asyncio.create_task(self.local_entity_retrieval(query))

            global_results, local_results = await asyncio.gather(global_task, local_task)

            return {
                "global_communities": global_results,
                "local_entities": local_results,
                "total_results": len(global_results) + len(local_results)
            }

        except Exception as e:
            logger.error(f"Error in parallel retrieval: {e}")
            return {"global_communities": [], "local_entities": [], "total_results": 0}

    def export_to_neo4j_file(self, knowledge_graph, filepath: Union[str, Path]) -> None:
        """Export KnowledgeGraph to Neo4j import format."""
        path = Path(filepath)

        # Create nodes CSV
        nodes_data = []
        for entity in knowledge_graph.entities.values():
            row = {
                "id": entity.id,
                "label": entity.label,
                "type": entity.type,
                "confidence": entity.confidence,
                "source": entity.source
            }
            row.update(entity.properties)
            nodes_data.append(row)

        # Create relationships CSV
        rels_data = []
        for relation in knowledge_graph.relations.values():
            row = {
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "type": relation.type,
                "confidence": relation.confidence,
                "source": relation.source
            }
            row.update(relation.properties)
            rels_data.append(row)

        # Write CSV files
        import pandas as pd
        nodes_df = pd.DataFrame(nodes_data)
        rels_df = pd.DataFrame(rels_data)

        nodes_path = path.with_suffix('.nodes.csv')
        rels_path = path.with_suffix('.relationships.csv')

        nodes_df.to_csv(nodes_path, index=False)
        rels_df.to_csv(rels_path, index=False)

        # Create import script
        script_path = path.with_suffix('.import.cypher')
        with open(script_path, 'w') as f:
            f.write(f"""
// Neo4j Import Script for {path.name}
// Generated by LitKG SDK

// Load nodes
LOAD CSV WITH HEADERS FROM 'file:///{nodes_path.name}' AS row
CREATE (n:Entity {{
    id: row.id,
    label: row.label,
    type: row.type,
    confidence: toFloat(row.confidence),
    source: row.source
}});

// Create type-specific labels
MATCH (n:Entity)
CALL apoc.create.addLabels(n, [n.type]) YIELD node
RETURN count(node);

// Load relationships
LOAD CSV WITH HEADERS FROM 'file:///{rels_path.name}' AS row
MATCH (a:Entity {{id: row.source_id}}), (b:Entity {{id: row.target_id}})
CALL apoc.create.relationship(a, row.type, {{
    confidence: toFloat(row.confidence),
    source: row.source
}}, b) YIELD rel
RETURN count(rel);

// Create indexes for performance
CREATE INDEX entity_id_index IF NOT EXISTS FOR (n:Entity) ON (n.id);
CREATE INDEX entity_type_index IF NOT EXISTS FOR (n:Entity) ON (n.type);
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (n:Entity) ON EACH [n.label, n.source];
""")

        logger.info(f"Neo4j import files created: {nodes_path}, {rels_path}, {script_path}")

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get Neo4j graph statistics."""
        if not self.is_available():
            return {}

        try:
            with self.driver.session() as session:
                # Node counts by type
                node_result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as type, count(n) as count
                    ORDER BY count DESC
                """)
                node_stats = [dict(record) for record in node_result]

                # Relationship counts by type
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY count DESC
                """)
                rel_stats = [dict(record) for record in rel_result]

                # Total counts
                total_result = session.run("""
                    MATCH (n)
                    OPTIONAL MATCH ()-[r]->()
                    RETURN count(DISTINCT n) as nodes, count(r) as relationships
                """)
                totals = dict(total_result.single())

                # Community stats if available
                community_result = session.run("""
                    MATCH (c:CommunityNode)
                    RETURN count(c) as communities, avg(c.size) as avg_size
                """)
                community_stats = dict(community_result.single()) if community_result.peek() else {}

                return {
                    "totals": totals,
                    "nodes_by_type": node_stats,
                    "relationships_by_type": rel_stats,
                    "communities": community_stats
                }

        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {}

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
        if self.gds:
            self.gds.close()