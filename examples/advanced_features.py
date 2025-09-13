"""
Advanced features example for LitKG SDK.

This example demonstrates advanced capabilities including:
- Batch processing multiple papers
- Neo4j integration
- Community detection
- Temporal analysis
- Custom entity/relation types
- Local LLM usage
"""

import litkg
import asyncio
from pathlib import Path
import json


async def neo4j_integration_example():
    """Example of Neo4j integration with community detection."""
    print("🗄️ Neo4j Integration Example")
    print("=" * 40)

    # Create session with Neo4j
    session = litkg.create_session(
        llm="gpt-4",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        enable_communities=True,
        enable_parallel_retrieval=True
    )

    # Create demo knowledge graph
    kg = litkg.KnowledgeGraph(session)

    # Add entities representing a research paper network
    from litkg.core.knowledge_graph import Entity, Relation

    entities = [
        # Authors
        Entity("author_1", "Geoffrey Hinton", "Person",
               {"affiliation": "University of Toronto", "field": "Deep Learning"}, 0.99, "paper1.pdf"),
        Entity("author_2", "Yann LeCun", "Person",
               {"affiliation": "Facebook AI", "field": "Computer Vision"}, 0.99, "paper2.pdf"),
        Entity("author_3", "Yoshua Bengio", "Person",
               {"affiliation": "University of Montreal", "field": "AI"}, 0.99, "paper3.pdf"),

        # Concepts
        Entity("concept_1", "Neural Networks", "Concept",
               {"domain": "Machine Learning", "year_introduced": 1943}, 0.95, "paper1.pdf"),
        Entity("concept_2", "Backpropagation", "Algorithm",
               {"complexity": "O(n)", "invented": 1986}, 0.94, "paper1.pdf"),
        Entity("concept_3", "Convolutional Neural Networks", "Architecture",
               {"application": "Computer Vision", "year": 1989}, 0.96, "paper2.pdf"),
        Entity("concept_4", "Transformers", "Architecture",
               {"attention_mechanism": True, "year": 2017}, 0.97, "paper3.pdf"),

        # Tools/Frameworks
        Entity("tool_1", "TensorFlow", "Framework",
               {"company": "Google", "language": "Python"}, 0.93, "paper4.pdf"),
        Entity("tool_2", "PyTorch", "Framework",
               {"company": "Meta", "dynamic": True}, 0.92, "paper4.pdf"),
    ]

    for entity in entities:
        kg.add_entity(entity)

    # Add relations
    relations = [
        # Author contributions
        Relation("rel_1", "author_1", "concept_2", "Developed",
                {"contribution": "co-invented"}, 0.96, "paper1.pdf"),
        Relation("rel_2", "author_2", "concept_3", "Invented",
                {"year": 1989}, 0.98, "paper2.pdf"),
        Relation("rel_3", "author_3", "concept_4", "Contributed",
                {"role": "attention_mechanism"}, 0.89, "paper3.pdf"),

        # Concept relationships
        Relation("rel_4", "concept_2", "concept_1", "AlgorithmFor",
                {"purpose": "training"}, 0.94, "paper1.pdf"),
        Relation("rel_5", "concept_3", "concept_1", "TypeOf",
                {"specialization": "vision"}, 0.91, "paper2.pdf"),
        Relation("rel_6", "concept_4", "concept_1", "EvolutionOf",
                {"improvement": "attention"}, 0.88, "paper3.pdf"),

        # Tool relationships
        Relation("rel_7", "tool_1", "concept_1", "Implements",
                {"support": "full"}, 0.90, "paper4.pdf"),
        Relation("rel_8", "tool_2", "concept_1", "Implements",
                {"dynamic": True}, 0.89, "paper4.pdf"),
    ]

    for relation in relations:
        kg.add_relation(relation)

    print(f"✅ Created research network: {len(kg.entities)} entities, {len(kg.relations)} relations")

    # Analyze communities
    communities = kg.analyze_communities()
    print(f"\n🏘️ Community Detection:")
    print(f"   Found {communities.get('num_communities', 0)} communities")

    if 'communities' in communities:
        for comm_id, members in communities['communities'].items():
            print(f"   Community {comm_id}: {len(members)} members")
            member_labels = [kg.entities[m].label for m in members if m in kg.entities]
            print(f"     {', '.join(member_labels[:3])}{'...' if len(member_labels) > 3 else ''}")

    # Neo4j integration (if available)
    try:
        neo4j_builder = session.neo4j_builder
        if neo4j_builder and neo4j_builder.is_available():
            print(f"\n🗄️ Exporting to Neo4j...")
            stats = neo4j_builder.create_graph_from_kg(kg)
            print(f"   Created {stats['nodes_created']} nodes, {stats['relationships_created']} relationships")

            # Setup parallel retrievers
            retriever_stats = neo4j_builder.setup_parallel_retrievers()
            if retriever_stats['status'] == 'success':
                print(f"   ✅ Parallel retrievers configured")

                # Test retrieval
                results = await neo4j_builder.parallel_retrieval("neural networks")
                print(f"   Found {results['total_results']} results for 'neural networks'")

        else:
            print(f"\n⚠️ Neo4j not available - would export to file")
            neo4j_builder.export_to_neo4j_file(kg, "research_network.neo4j")

    except Exception as e:
        print(f"   ❌ Neo4j integration failed: {e}")

    return kg


async def temporal_analysis_example():
    """Example of temporal knowledge graph analysis."""
    print("\n" + "=" * 50)
    print("⏰ Temporal Analysis Example")
    print("=" * 50)

    # Create session with temporal tracking
    session = litkg.create_session(
        llm="claude-3.5-sonnet",
        enable_temporal=True
    )

    # Initialize temporal knowledge graph
    try:
        from litkg.temporal.graphiti import TemporalKG
        temporal_kg = TemporalKG()

        # Add temporal entities (simulating evolution over time)
        from datetime import datetime, timedelta

        base_time = datetime(2020, 1, 1)

        # Evolution of AI concepts
        await temporal_kg.add_temporal_node(
            "ai_concept",
            "Artificial Intelligence",
            "Concept",
            {"definition": "rule-based systems", "capability": "narrow"},
            0.80,
            base_time
        )

        await temporal_kg.add_temporal_node(
            "ai_concept",
            "Artificial Intelligence",
            "Concept",
            {"definition": "machine learning systems", "capability": "broad"},
            0.85,
            base_time + timedelta(days=365)
        )

        await temporal_kg.add_temporal_node(
            "ai_concept",
            "Artificial Intelligence",
            "Concept",
            {"definition": "large language models", "capability": "general"},
            0.92,
            base_time + timedelta(days=730)
        )

        # Track evolution
        evolution = await temporal_kg.track_entity_evolution("ai_concept")
        print(f"📈 Entity Evolution:")
        print(f"   Total versions: {evolution['total_versions']}")
        print(f"   First seen: {evolution['first_seen']}")
        print(f"   Last updated: {evolution['last_updated']}")

        for i, version in enumerate(evolution['evolution']):
            print(f"   Version {version['version']}: {version['timestamp'][:10]} - {version['confidence']}")
            if version['changes']:
                for change in version['changes']:
                    print(f"     • {change['type']}: {change.get('from', 'N/A')} → {change.get('to', 'N/A')}")

        # Point-in-time query
        query_time = base_time + timedelta(days=500)
        snapshot = await temporal_kg.query_point_in_time(query_time)
        print(f"\n🕐 Snapshot at {query_time.strftime('%Y-%m-%d')}:")
        print(f"   Nodes: {len(snapshot['nodes'])}")
        print(f"   Edges: {len(snapshot['edges'])}")

        # Get timeline
        timeline = await temporal_kg.get_knowledge_timeline()
        print(f"\n📊 Knowledge Timeline:")
        print(f"   Total events: {timeline['stats']['total_events']}")
        print(f"   Time span: {timeline['stats']['time_span']['duration_days']} days")

        # Export temporal data
        await temporal_kg.export_temporal_data("temporal_knowledge.json")
        print(f"   ✅ Temporal data exported")

    except ImportError:
        print("   ⚠️ Graphiti not available - using basic temporal tracking")
        # Fallback to basic temporal analysis
        kg = litkg.KnowledgeGraph(session)
        evolution = kg.track_knowledge_evolution()
        print(f"   Basic evolution tracking: {len(evolution)} sources")


async def batch_processing_example():
    """Example of batch processing multiple documents."""
    print("\n" + "=" * 50)
    print("📚 Batch Processing Example")
    print("=" * 50)

    session = litkg.create_session(
        llm="gpt-4",
        batch_size=5,
        max_workers=3,
        domain="computer_science"
    )

    # Simulate processing multiple papers
    paper_files = [
        "attention_is_all_you_need.pdf",
        "bert_paper.pdf",
        "gpt2_paper.pdf",
        "transformer_xl.pdf",
        "electra_paper.pdf",
        "roberta_paper.pdf"
    ]

    print(f"📄 Would process {len(paper_files)} papers:")
    for paper in paper_files:
        print(f"   • {paper}")

    # For demo, create merged knowledge graph
    merged_kg = litkg.KnowledgeGraph(session)

    # Add entities from different papers
    from litkg.core.knowledge_graph import Entity, Relation

    papers_data = {
        "attention_paper": {
            "entities": [
                Entity("transformer_1", "Transformer", "Architecture",
                       {"attention": "multi-head", "year": 2017}, 0.98, "attention_paper.pdf"),
                Entity("attention_1", "Self-Attention", "Mechanism",
                       {"complexity": "O(n²)", "parallelizable": True}, 0.96, "attention_paper.pdf")
            ],
            "relations": [
                Relation("rel_t1", "transformer_1", "attention_1", "Uses",
                        {"primary_mechanism": True}, 0.95, "attention_paper.pdf")
            ]
        },
        "bert_paper": {
            "entities": [
                Entity("bert_1", "BERT", "Model",
                       {"bidirectional": True, "pretraining": "MLM"}, 0.99, "bert_paper.pdf"),
                Entity("mlm_1", "Masked Language Modeling", "Task",
                       {"objective": "pretraining"}, 0.94, "bert_paper.pdf")
            ],
            "relations": [
                Relation("rel_b1", "bert_1", "transformer_1", "BasedOn",
                        {"encoder_only": True}, 0.93, "bert_paper.pdf"),
                Relation("rel_b2", "bert_1", "mlm_1", "TrainedWith",
                        {"objective": "prediction"}, 0.91, "bert_paper.pdf")
            ]
        }
    }

    # Add all entities and relations
    for paper, data in papers_data.items():
        for entity in data["entities"]:
            merged_kg.add_entity(entity)
        for relation in data["relations"]:
            merged_kg.add_relation(relation)

    print(f"\n✅ Batch processing complete:")
    print(f"   📊 Total entities: {len(merged_kg.entities)}")
    print(f"   🔗 Total relations: {len(merged_kg.relations)}")

    # Analyze cross-paper connections
    source_stats = {}
    for entity in merged_kg.entities.values():
        source = entity.source
        if source not in source_stats:
            source_stats[source] = {"entities": 0, "avg_confidence": 0}
        source_stats[source]["entities"] += 1

    print(f"\n📈 Cross-paper Analysis:")
    for source, stats in source_stats.items():
        entities_from_source = [e for e in merged_kg.entities.values() if e.source == source]
        avg_conf = sum(e.confidence for e in entities_from_source) / len(entities_from_source)
        print(f"   {source}: {stats['entities']} entities, avg confidence: {avg_conf:.3f}")

    return merged_kg


async def custom_domain_example():
    """Example of domain-specific knowledge graph construction."""
    print("\n" + "=" * 50)
    print("🎯 Domain-Specific Example (Biomedical)")
    print("=" * 50)

    # Create biomedical domain session
    session = litkg.create_session(
        llm="gpt-4",
        domain="biomedical",
        custom_entities=["Gene", "Protein", "Disease", "Drug", "Pathway", "Organism"],
        custom_relations=["Encodes", "Treats", "CausedBy", "InteractsWith", "PartOf", "ExpressedIn"],
        confidence_threshold=0.8
    )

    kg = litkg.KnowledgeGraph(session)

    # Add biomedical entities
    from litkg.core.knowledge_graph import Entity, Relation

    bio_entities = [
        Entity("brca1_gene", "BRCA1", "Gene",
               {"chromosome": "17", "function": "DNA_repair"}, 0.99, "cancer_study.pdf"),
        Entity("brca1_protein", "BRCA1 Protein", "Protein",
               {"domains": "RING, BRCT", "cellular_location": "nucleus"}, 0.97, "cancer_study.pdf"),
        Entity("breast_cancer", "Breast Cancer", "Disease",
               {"type": "malignant", "affected_tissue": "breast"}, 0.95, "cancer_study.pdf"),
        Entity("tamoxifen", "Tamoxifen", "Drug",
               {"class": "SERM", "mechanism": "estrogen_receptor_modulator"}, 0.92, "treatment_study.pdf"),
        Entity("dna_repair", "DNA Repair Pathway", "Pathway",
               {"process": "homologous_recombination"}, 0.90, "molecular_study.pdf")
    ]

    for entity in bio_entities:
        kg.add_entity(entity)

    # Add biomedical relations
    bio_relations = [
        Relation("rel_bio1", "brca1_gene", "brca1_protein", "Encodes",
                {"translation": True}, 0.98, "cancer_study.pdf"),
        Relation("rel_bio2", "brca1_gene", "breast_cancer", "AssociatedWith",
                {"mutation_risk": "high"}, 0.94, "cancer_study.pdf"),
        Relation("rel_bio3", "tamoxifen", "breast_cancer", "Treats",
                {"efficacy": "hormone_receptor_positive"}, 0.91, "treatment_study.pdf"),
        Relation("rel_bio4", "brca1_protein", "dna_repair", "PartOf",
                {"role": "critical_component"}, 0.89, "molecular_study.pdf")
    ]

    for relation in bio_relations:
        kg.add_relation(relation)

    print(f"🧬 Biomedical Knowledge Graph:")
    print(f"   Entities: {len(kg.entities)}")
    print(f"   Relations: {len(kg.relations)}")

    # Domain-specific analysis
    entity_types = {}
    for entity in kg.entities.values():
        entity_types[entity.type] = entity_types.get(entity.type, 0) + 1

    print(f"\n🔬 Biomedical Entity Distribution:")
    for etype, count in sorted(entity_types.items()):
        print(f"   {etype}: {count}")

    relation_types = {}
    for relation in kg.relations.values():
        relation_types[relation.type] = relation_types.get(relation.type, 0) + 1

    print(f"\n🔗 Biomedical Relation Distribution:")
    for rtype, count in sorted(relation_types.items()):
        print(f"   {rtype}: {count}")

    # Export for biomedical databases
    kg.export("biomedical_knowledge_graph.json")
    kg.export("biomedical_kg.graphml")  # For Cytoscape
    print(f"\n💾 Exported biomedical knowledge graph")

    return kg


async def local_llm_example():
    """Example using local LLM with Ollama."""
    print("\n" + "=" * 50)
    print("🏠 Local LLM Example (Privacy-First)")
    print("=" * 50)

    try:
        # Create session with local model
        local_session = litkg.create_session(
            llm="ollama/llama3.1",
            confidence_threshold=0.75,
            enable_communities=True,
            local_processing=True
        )

        print("✅ Local session created with Llama 3.1")
        print("🔒 All processing stays on your machine")

        # Create simple knowledge graph with local processing
        kg = litkg.KnowledgeGraph(local_session)

        # Add privacy-focused entities
        from litkg.core.knowledge_graph import Entity, Relation

        privacy_entities = [
            Entity("local_llm", "Local Language Models", "Concept",
                   {"privacy": "high", "deployment": "on_premise"}, 0.90, "privacy_paper.pdf"),
            Entity("ollama", "Ollama", "Framework",
                   {"purpose": "local_deployment", "models": "various"}, 0.88, "tools_paper.pdf"),
            Entity("privacy", "Data Privacy", "Concept",
                   {"importance": "critical", "regulation": "GDPR"}, 0.95, "privacy_paper.pdf")
        ]

        for entity in privacy_entities:
            kg.add_entity(entity)

        relations = [
            Relation("rel_p1", "ollama", "local_llm", "Enables",
                    {"deployment_method": "local"}, 0.87, "tools_paper.pdf"),
            Relation("rel_p2", "local_llm", "privacy", "Enhances",
                    {"benefit": "data_control"}, 0.92, "privacy_paper.pdf")
        ]

        for relation in relations:
            kg.add_relation(relation)

        print(f"🔒 Privacy-focused knowledge graph:")
        print(f"   Entities: {len(kg.entities)}")
        print(f"   Relations: {len(kg.relations)}")
        print(f"   All data processed locally ✅")

        # Export without cloud dependencies
        kg.export("local_knowledge_graph.json")

    except Exception as e:
        print(f"❌ Local LLM not available: {e}")
        print("💡 To use local models:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Run: ollama pull llama3.1")
        print("   3. Ensure Ollama service is running")


async def main():
    """Run all advanced examples."""
    print("🚀 LitKG SDK - Advanced Features Demo")
    print("=" * 60)

    examples = [
        ("Neo4j Integration", neo4j_integration_example),
        ("Temporal Analysis", temporal_analysis_example),
        ("Batch Processing", batch_processing_example),
        ("Domain-Specific (Biomedical)", custom_domain_example),
        ("Local LLM Processing", local_llm_example)
    ]

    for name, example_func in examples:
        try:
            print(f"\n🔄 Running: {name}")
            await example_func()
            print(f"✅ Completed: {name}")
        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    print("\n" + "=" * 60)
    print("🎉 Advanced features demonstration complete!")
    print("\n💡 Next steps:")
    print("   • Try with your own PDF documents")
    print("   • Set up Neo4j for graph database storage")
    print("   • Configure domain-specific entity types")
    print("   • Explore temporal analysis with real data")
    print("   • Deploy with local LLMs for privacy")


if __name__ == "__main__":
    # Run async examples
    asyncio.run(main())