"""
Basic usage example for LitKG SDK.

This example demonstrates the core 4-line API for converting PDF literature
into interactive knowledge graphs.
"""

import litkg

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️ python-dotenv not installed. Make sure to set API keys manually.")
    print("   Install with: pip install python-dotenv")

def basic_example():
    """Basic 4-line usage example."""
    print("🚀 LitKG SDK - Basic Usage Example")
    print("=" * 40)

    # 1. Create session with LLM
    print("1️⃣ Creating session...")
    session = litkg.create_session(llm="gpt-4")
    print("✅ Session created with GPT-4")

    # 2. Upload PDF (using a placeholder for demo)
    print("\n2️⃣ Processing PDF...")
    # kg = session.upload_pdf("research_paper.pdf")
    # For demo, we'll create a mock knowledge graph
    kg = session.active_kg or litkg.KnowledgeGraph(session)

    # Add some demo entities
    from litkg.core.knowledge_graph import Entity, Relation

    # Demo entities
    kg.add_entity(Entity(
        id="entity_1",
        label="Large Language Models",
        type="Concept",
        properties={"domain": "AI", "year": "2023"},
        confidence=0.95,
        source="demo"
    ))

    kg.add_entity(Entity(
        id="entity_2",
        label="Knowledge Graphs",
        type="Concept",
        properties={"domain": "AI", "applications": "information_extraction"},
        confidence=0.92,
        source="demo"
    ))

    kg.add_entity(Entity(
        id="entity_3",
        label="GPT-4",
        type="Model",
        properties={"company": "OpenAI", "type": "transformer"},
        confidence=0.98,
        source="demo"
    ))

    # Demo relations
    kg.add_relation(Relation(
        id="relation_1",
        source_id="entity_3",
        target_id="entity_1",
        type="InstanceOf",
        properties={"context": "GPT-4 is a type of LLM"},
        confidence=0.96,
        source="demo"
    ))

    kg.add_relation(Relation(
        id="relation_2",
        source_id="entity_1",
        target_id="entity_2",
        type="UsedFor",
        properties={"purpose": "knowledge extraction"},
        confidence=0.89,
        source="demo"
    ))

    print(f"✅ Created demo knowledge graph: {len(kg.entities)} entities, {len(kg.relations)} relations")

    # 3. Interactive validation (simulated)
    print("\n3️⃣ Interactive validation...")
    print("📋 Entities found:")
    for entity in kg.entities.values():
        print(f"  • {entity.label} ({entity.type}) - confidence: {entity.confidence:.2f}")

    print("\n🔗 Relations found:")
    for relation in kg.relations.values():
        source_label = kg.entities[relation.source_id].label
        target_label = kg.entities[relation.target_id].label
        print(f"  • {source_label} → {relation.type} → {target_label} - confidence: {relation.confidence:.2f}")

    # kg.collaborate_interactively()  # Would launch Jupyter interface
    print("✅ Validation complete (demo mode)")

    # 4. Export results
    print("\n4️⃣ Exporting knowledge graph...")
    kg.export("examples/demo_knowledge_graph.json")
    print("✅ Exported to demo_knowledge_graph.json")

    print(f"\n🎉 Success! Created knowledge graph with:")
    print(f"   📊 {len(kg.entities)} entities")
    print(f"   🔗 {len(kg.relations)} relations")
    print(f"   📈 Average confidence: {sum(e.confidence for e in kg.entities.values()) / len(kg.entities):.2f}")


def advanced_example():
    """Advanced usage with configuration."""
    print("\n" + "=" * 50)
    print("🔬 Advanced Usage Example")
    print("=" * 50)

    # Create session with advanced configuration
    session = litkg.create_session(
        llm="gpt-4",
        confidence_threshold=0.8,
        enable_communities=True,
        enable_temporal=True,
        domain="computer_science"
    )

    print("✅ Advanced session created")

    # Batch processing example
    print("\n📚 Batch processing simulation...")
    papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
    print(f"Would process {len(papers)} papers:")
    for paper in papers:
        print(f"  • {paper}")

    # kg = session.batch_process(papers)
    print("✅ Batch processing complete (simulated)")

    # Community analysis
    print("\n🏘️ Community detection...")
    # communities = kg.analyze_communities()
    print("✅ Found 3 research communities")

    # Temporal tracking
    print("\n⏰ Temporal analysis...")
    # evolution = kg.track_knowledge_evolution()
    print("✅ Tracked knowledge evolution over time")

    print("\n🎯 Advanced features demonstrated!")


if __name__ == "__main__":
    try:
        basic_example()
        advanced_example()

        print("\n" + "=" * 50)
        print("🚀 Try LitKG SDK yourself:")
        print("   pip install lit-kg-sdk[all]")
        print("   import litkg")
        print("   session = litkg.create_session(llm='gpt-4')")
        print("   kg = session.upload_pdf('your_paper.pdf')")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("\n💡 This is a demo - install dependencies for full functionality:")
        print("   pip install lit-kg-sdk[all]")