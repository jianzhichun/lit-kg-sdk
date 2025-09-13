"""
Demo mode example for LitKG SDK without requiring API keys.

This example demonstrates the SDK functionality using mock data
and simulated LLM responses.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import litkg
from litkg.core.knowledge_graph import Entity, Relation


def demo_without_api_keys():
    """Demonstrate LitKG SDK functionality without API keys."""
    print("🎯 LitKG SDK - Demo Mode (No API Keys Required)")
    print("=" * 60)

    try:
        # 1. Create session (will use fallback when no API key)
        print("1️⃣ Creating demo session...")
        config = litkg.Config(
            llm_provider="demo",
            llm_model="demo-model",
            api_key="demo-key"  # Mock API key for demo
        )
        session = litkg.Session(config)
        print("✅ Demo session created")

        # 2. Create knowledge graph manually (simulating PDF processing)
        print("\n2️⃣ Creating knowledge graph...")
        kg = litkg.KnowledgeGraph(session)

        # Add sample entities (simulating LLM extraction)
        entities = [
            Entity("llm_1", "Large Language Models", "Concept",
                   {"field": "AI", "year": 2023, "description": "Advanced AI models for text processing"},
                   0.95, "research_paper.pdf"),
            Entity("gpt4_1", "GPT-4", "Model",
                   {"company": "OpenAI", "parameters": "1.76T", "release_year": 2023},
                   0.98, "research_paper.pdf"),
            Entity("transformer_1", "Transformer Architecture", "Architecture",
                   {"attention_mechanism": "multi-head", "year": 2017},
                   0.94, "research_paper.pdf"),
            Entity("bert_1", "BERT", "Model",
                   {"bidirectional": True, "pretraining": "MLM"},
                   0.92, "bert_paper.pdf"),
            Entity("attention_1", "Attention Mechanism", "Concept",
                   {"type": "self-attention", "complexity": "O(n²)"},
                   0.90, "attention_paper.pdf")
        ]

        for entity in entities:
            kg.add_entity(entity)

        # Add sample relations
        relations = [
            Relation("rel_1", "gpt4_1", "llm_1", "InstanceOf",
                     {"specificity": "high", "confidence_reason": "GPT-4 is clearly an LLM"},
                     0.96, "research_paper.pdf"),
            Relation("rel_2", "bert_1", "llm_1", "InstanceOf",
                     {"bidirectional": True},
                     0.93, "bert_paper.pdf"),
            Relation("rel_3", "transformer_1", "attention_1", "Uses",
                     {"primary_mechanism": True},
                     0.91, "attention_paper.pdf"),
            Relation("rel_4", "gpt4_1", "transformer_1", "BasedOn",
                     {"architecture": "decoder-only"},
                     0.89, "research_paper.pdf"),
            Relation("rel_5", "bert_1", "transformer_1", "BasedOn",
                     {"architecture": "encoder-only"},
                     0.87, "bert_paper.pdf")
        ]

        for relation in relations:
            kg.add_relation(relation)

        print(f"✅ Knowledge graph created:")
        print(f"   📊 {len(kg.entities)} entities")
        print(f"   🔗 {len(kg.relations)} relations")

        # 3. Display extracted knowledge
        print("\n3️⃣ Knowledge extraction results:")
        print("\n📋 Entities found:")
        for entity in kg.entities.values():
            print(f"  • {entity.label} ({entity.type}) - confidence: {entity.confidence:.2f}")
            if entity.properties:
                props = ", ".join([f"{k}: {v}" for k, v in list(entity.properties.items())[:2]])
                print(f"    Properties: {props}")

        print("\n🔗 Relations found:")
        for relation in kg.relations.values():
            source_label = kg.entities[relation.source_id].label
            target_label = kg.entities[relation.target_id].label
            print(f"  • {source_label} → {relation.type} → {target_label}")
            print(f"    Confidence: {relation.confidence:.2f}")

        # 4. Analysis features
        print("\n4️⃣ Knowledge graph analysis:")

        # Entity type distribution
        entity_types = {}
        for entity in kg.entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1

        print(f"\n📊 Entity type distribution:")
        for etype, count in sorted(entity_types.items()):
            print(f"   {etype}: {count}")

        # Confidence statistics
        all_confidences = [e.confidence for e in kg.entities.values()] + [r.confidence for r in kg.relations.values()]
        avg_confidence = sum(all_confidences) / len(all_confidences)
        print(f"\n📈 Confidence statistics:")
        print(f"   Average: {avg_confidence:.3f}")
        print(f"   Range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")

        # 5. Community detection (simulated)
        print("\n5️⃣ Community detection:")
        communities = kg.analyze_communities()
        print(f"   Found {communities.get('num_communities', 0)} communities")

        # 6. Export functionality
        print("\n6️⃣ Export options:")
        export_formats = ["JSON", "GraphML", "Neo4j Cypher", "CSV"]

        for fmt in export_formats:
            try:
                filename = f"demo_kg.{fmt.lower().replace(' ', '_')}"
                if fmt == "JSON":
                    kg.export(filename, "json")
                    print(f"   ✅ {fmt}: {filename}")
                else:
                    print(f"   📋 {fmt}: {filename} (available)")
            except Exception as e:
                print(f"   ⚠️ {fmt}: {e}")

        # 7. Advanced features preview
        print("\n7️⃣ Advanced features available:")
        features = [
            "🤖 Multi-LLM Support (OpenAI, Claude, Gemini, Local models)",
            "🔄 Human-in-the-loop validation",
            "🏘️ Community detection and clustering",
            "⏰ Temporal knowledge evolution tracking",
            "🗄️ Neo4j database integration",
            "📊 Interactive Jupyter notebook interface",
            "📄 Advanced PDF processing with structure preservation",
            "🔍 Parallel knowledge retrieval"
        ]

        for feature in features:
            print(f"   {feature}")

        # 8. Usage statistics
        print(f"\n8️⃣ Session statistics:")
        stats = {
            "entities_created": len(kg.entities),
            "relations_created": len(kg.relations),
            "avg_entity_confidence": sum(e.confidence for e in kg.entities.values()) / len(kg.entities),
            "avg_relation_confidence": sum(r.confidence for r in kg.relations.values()) / len(kg.relations),
            "unique_sources": len(set(e.source for e in kg.entities.values())),
            "knowledge_density": len(kg.relations) / len(kg.entities) if kg.entities else 0
        }

        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value}")

        print(f"\n🎉 Demo completed successfully!")
        print(f"\n💡 To use with real PDFs and LLMs:")
        print(f"   1. Set up API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)")
        print(f"   2. Install additional dependencies: uv pip install lit-kg-sdk[all]")
        print(f"   3. Use: session = litkg.create_session(llm='gpt-4')")
        print(f"   4. Process: kg = session.upload_pdf('your_paper.pdf')")

        return kg

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
        return None


def interactive_demo():
    """Interactive demo with user choices."""
    print("\n" + "=" * 60)
    print("🎮 Interactive Demo Mode")
    print("=" * 60)

    kg = demo_without_api_keys()
    if not kg:
        return

    while True:
        print(f"\n🔧 Interactive options:")
        print(f"   1. View detailed entity information")
        print(f"   2. Explore graph structure")
        print(f"   3. Filter by confidence threshold")
        print(f"   4. Show graph statistics")
        print(f"   5. Export knowledge graph")
        print(f"   6. Exit")

        try:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                print(f"\n📋 Detailed entity information:")
                for i, (entity_id, entity) in enumerate(kg.entities.items(), 1):
                    print(f"\n{i}. {entity.label} (ID: {entity_id})")
                    print(f"   Type: {entity.type}")
                    print(f"   Confidence: {entity.confidence:.3f}")
                    print(f"   Source: {entity.source}")
                    if entity.properties:
                        print(f"   Properties:")
                        for key, value in entity.properties.items():
                            print(f"     • {key}: {value}")

            elif choice == "2":
                print(f"\n🕸️ Graph structure exploration:")
                for entity_id, entity in kg.entities.items():
                    neighbors = kg.get_neighbors(entity_id)
                    print(f"\n📍 {entity.label}:")
                    if neighbors:
                        for neighbor_id in neighbors:
                            neighbor = kg.entities.get(neighbor_id)
                            if neighbor:
                                print(f"   → Connected to: {neighbor.label}")
                    else:
                        print(f"   → No direct connections")

            elif choice == "3":
                try:
                    threshold = float(input("Enter confidence threshold (0.0-1.0): "))
                    if 0.0 <= threshold <= 1.0:
                        filtered_kg = kg.filter_by_confidence(threshold)
                        print(f"\n📊 Filtered results (confidence ≥ {threshold}):")
                        print(f"   Entities: {len(filtered_kg.entities)} (was {len(kg.entities)})")
                        print(f"   Relations: {len(filtered_kg.relations)} (was {len(kg.relations)})")
                    else:
                        print(f"❌ Please enter a value between 0.0 and 1.0")
                except ValueError:
                    print(f"❌ Please enter a valid number")

            elif choice == "4":
                print(f"\n📈 Graph statistics:")
                print(f"   Total nodes: {len(kg.entities)}")
                print(f"   Total edges: {len(kg.relations)}")
                print(f"   Graph density: {len(kg.relations) / (len(kg.entities) * (len(kg.entities) - 1)) * 2:.3f}")

                # Degree distribution
                degrees = {}
                for entity_id in kg.entities:
                    degree = len(kg.get_neighbors(entity_id))
                    degrees[degree] = degrees.get(degree, 0) + 1

                print(f"   Degree distribution:")
                for degree in sorted(degrees.keys()):
                    print(f"     Degree {degree}: {degrees[degree]} nodes")

            elif choice == "5":
                format_choice = input("Export format (json/graphml/cypher): ").lower()
                if format_choice in ["json", "graphml", "cypher"]:
                    filename = f"interactive_export.{format_choice}"
                    try:
                        kg.export(filename, format_choice)
                        print(f"✅ Exported to {filename}")
                    except Exception as e:
                        print(f"❌ Export failed: {e}")
                else:
                    print(f"❌ Unsupported format. Use: json, graphml, or cypher")

            elif choice == "6":
                print(f"👋 Thanks for exploring LitKG SDK!")
                break

            else:
                print(f"❌ Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print(f"\n👋 Demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    try:
        # Run basic demo
        demo_without_api_keys()

        # Ask if user wants interactive mode (only if stdin is available)
        try:
            if sys.stdin.isatty():  # Check if running in interactive terminal
                response = input("\n🎮 Run interactive demo? (y/n): ")
                if response.lower().startswith('y'):
                    interactive_demo()
            else:
                print("\n💡 Non-interactive mode - skipping interactive demo")
                print("   Run directly in terminal for interactive features")
        except (EOFError, OSError):
            print("\n💡 Non-interactive environment detected")

    except KeyboardInterrupt:
        print(f"\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()