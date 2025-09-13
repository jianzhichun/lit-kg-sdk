"""Session management for LitKG SDK."""

import asyncio
import logging
from typing import Optional, List, Union, Dict, Any
from pathlib import Path

from .config import Config
from .knowledge_graph import KnowledgeGraph, Entity, Relation
from ..providers.llm_providers import get_llm_provider


logger = logging.getLogger(__name__)


class Session:
    """Main session class for LitKG SDK with simplified API."""

    def __init__(self, config: Config):
        """Initialize session with configuration."""
        self.config = config
        self.config.validate()

        # Initialize LLM provider
        self.llm_provider = get_llm_provider(config)

        # Initialize components
        self._pdf_processor = None
        self._neo4j_builder = None
        self._human_loop = None
        self._temporal_tracker = None

        # Session state
        self.active_kg: Optional[KnowledgeGraph] = None
        self.processing_history: List[Dict[str, Any]] = []

    @property
    def pdf_processor(self):
        """Lazy load PDF processor."""
        if self._pdf_processor is None:
            try:
                from ..processing.pdf_processor import PDFProcessor
                self._pdf_processor = PDFProcessor(self.config)
            except ImportError:
                logger.warning("PDF processing not available. Install with: pip install lit-kg-sdk[pdf]")
                self._pdf_processor = None
        return self._pdf_processor

    @property
    def neo4j_builder(self):
        """Lazy load Neo4j builder."""
        if self._neo4j_builder is None:
            try:
                from ..providers.neo4j_builder import Neo4jBuilder
                self._neo4j_builder = Neo4jBuilder(self.config)
            except ImportError:
                logger.warning("Neo4j integration not available. Install neo4j package.")
                self._neo4j_builder = None
        return self._neo4j_builder

    @property
    def human_loop(self):
        """Lazy load human loop interface."""
        if self._human_loop is None:
            try:
                from ..human_loop.langgraph_workflow import HumanLoopWorkflow
                self._human_loop = HumanLoopWorkflow(self.config, self.llm_provider)
            except ImportError:
                logger.warning("Human loop not available. Install with: pip install langgraph")
                self._human_loop = None
        return self._human_loop

    def upload_pdf(self,
                   filepath: Union[str, Path],
                   preserve_structure: bool = True,
                   chunking_strategy: str = "semantic") -> KnowledgeGraph:
        """
        Upload and process PDF into knowledge graph.

        Args:
            filepath: Path to PDF file
            preserve_structure: Whether to preserve document structure
            chunking_strategy: "semantic" or "fixed"

        Returns:
            KnowledgeGraph: Processed knowledge graph
        """
        logger.info(f"Processing PDF: {filepath}")

        # Create new knowledge graph
        kg = KnowledgeGraph(session=self)

        try:
            # Extract text from PDF
            if self.pdf_processor:
                text_chunks = self.pdf_processor.extract_text(
                    filepath,
                    preserve_structure=preserve_structure,
                    chunking_strategy=chunking_strategy
                )
            else:
                # Fallback to simple text extraction
                text_chunks = self._fallback_pdf_extraction(filepath)

            # Extract entities and relations using LLM
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")

                # LLM extraction
                extraction_result = asyncio.run(
                    self._extract_knowledge_from_text(chunk, source=f"chunk_{i}")
                )

                # Add to knowledge graph
                for entity_data in extraction_result.get("entities", []):
                    entity = Entity(
                        id=entity_data["id"],
                        label=entity_data["label"],
                        type=entity_data["type"],
                        properties=entity_data.get("properties", {}),
                        confidence=entity_data.get("confidence", 0.8),
                        source=str(filepath)
                    )
                    kg.add_entity(entity)

                for relation_data in extraction_result.get("relations", []):
                    relation = Relation(
                        id=relation_data["id"],
                        source_id=relation_data["source_id"],
                        target_id=relation_data["target_id"],
                        type=relation_data["type"],
                        properties=relation_data.get("properties", {}),
                        confidence=relation_data.get("confidence", 0.8),
                        source=str(filepath)
                    )
                    kg.add_relation(relation)

            # Apply post-processing
            if self.config.enable_communities:
                kg.analyze_communities()

            if self.config.enable_temporal:
                kg.track_knowledge_evolution()

            # Store as active knowledge graph
            self.active_kg = kg

            # Record processing history
            self.processing_history.append({
                "type": "pdf_upload",
                "file": str(filepath),
                "entities": len(kg.entities),
                "relations": len(kg.relations),
                "timestamp": kg.metadata["created_at"]
            })

            logger.info(f"Extracted {len(kg.entities)} entities and {len(kg.relations)} relations")
            return kg

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def batch_process(self,
                     filepaths: List[Union[str, Path]],
                     merge_graphs: bool = True) -> Union[KnowledgeGraph, List[KnowledgeGraph]]:
        """
        Process multiple PDFs in batch.

        Args:
            filepaths: List of PDF file paths
            merge_graphs: Whether to merge all graphs into one

        Returns:
            KnowledgeGraph or List[KnowledgeGraph]: Processed graphs
        """
        logger.info(f"Batch processing {len(filepaths)} files")

        graphs = []
        for filepath in filepaths:
            try:
                kg = self.upload_pdf(filepath)
                graphs.append(kg)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                continue

        if merge_graphs and graphs:
            # Merge all graphs into the first one
            merged_kg = graphs[0]
            for kg in graphs[1:]:
                merged_kg.merge_with(kg)

            self.active_kg = merged_kg
            return merged_kg

        return graphs

    def load_corpus(self,
                   directory: Union[str, Path],
                   pattern: str = "*.pdf",
                   batch_size: Optional[int] = None) -> KnowledgeGraph:
        """
        Load and process entire corpus of documents.

        Args:
            directory: Directory containing documents
            pattern: File pattern to match
            batch_size: Number of files to process at once

        Returns:
            KnowledgeGraph: Merged knowledge graph
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))

        if not files:
            raise ValueError(f"No files found matching pattern '{pattern}' in {directory}")

        logger.info(f"Found {len(files)} files to process")

        if batch_size is None:
            batch_size = self.config.batch_size

        # Process in batches
        all_graphs = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}")

            batch_graphs = self.batch_process(batch, merge_graphs=False)
            if isinstance(batch_graphs, list):
                all_graphs.extend(batch_graphs)
            else:
                all_graphs.append(batch_graphs)

        # Merge all graphs
        if all_graphs:
            final_kg = all_graphs[0]
            for kg in all_graphs[1:]:
                final_kg.merge_with(kg)

            self.active_kg = final_kg
            return final_kg

        return KnowledgeGraph(session=self)

    async def _extract_knowledge_from_text(self,
                                         text: str,
                                         source: str = "unknown") -> Dict[str, Any]:
        """Extract entities and relations from text using LLM."""

        # Create extraction prompt
        prompt = self._create_extraction_prompt(text)

        try:
            # Call LLM
            response = await self.llm_provider.extract_structured(prompt)

            # Parse response
            if isinstance(response, str):
                import json
                response = json.loads(response)

            # Validate and clean response
            return self._validate_extraction_result(response, source)

        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            return {"entities": [], "relations": []}

    def _create_extraction_prompt(self, text: str) -> str:
        """Create extraction prompt for LLM."""

        # Use custom prompt if provided
        if self.config.extraction_prompt:
            return self.config.extraction_prompt.format(text=text)

        # Default prompt
        prompt = f"""
Extract entities and relationships from the following text. Return JSON format:

{{
  "entities": [
    {{
      "id": "unique_id",
      "label": "entity_name",
      "type": "entity_type",
      "properties": {{}},
      "confidence": 0.9
    }}
  ],
  "relations": [
    {{
      "id": "unique_id",
      "source_id": "entity1_id",
      "target_id": "entity2_id",
      "type": "relation_type",
      "properties": {{}},
      "confidence": 0.8
    }}
  ]
}}

Entity types to focus on: {", ".join(self.config.custom_entities) if self.config.custom_entities else "Person, Organization, Concept, Method, Tool, Location"}
Relation types to focus on: {", ".join(self.config.custom_relations) if self.config.custom_relations else "RelatedTo, UsedBy, PartOf, AuthoredBy, LocatedIn"}

Text to analyze:
{text}
"""
        return prompt

    def _validate_extraction_result(self,
                                  result: Dict[str, Any],
                                  source: str) -> Dict[str, Any]:
        """Validate and clean extraction result."""

        # Ensure required keys exist
        if "entities" not in result:
            result["entities"] = []
        if "relations" not in result:
            result["relations"] = []

        # Clean and validate entities
        cleaned_entities = []
        for entity in result["entities"]:
            if isinstance(entity, dict) and "id" in entity and "label" in entity:
                # Generate ID if missing
                if not entity["id"]:
                    entity["id"] = f"entity_{len(cleaned_entities)}_{source}"

                # Set defaults
                entity.setdefault("type", "Unknown")
                entity.setdefault("properties", {})
                entity.setdefault("confidence", 0.7)

                cleaned_entities.append(entity)

        # Clean and validate relations
        cleaned_relations = []
        entity_ids = {e["id"] for e in cleaned_entities}

        for relation in result["relations"]:
            if (isinstance(relation, dict) and
                "source_id" in relation and
                "target_id" in relation and
                relation["source_id"] in entity_ids and
                relation["target_id"] in entity_ids):

                # Generate ID if missing
                if not relation.get("id"):
                    relation["id"] = f"relation_{len(cleaned_relations)}_{source}"

                # Set defaults
                relation.setdefault("type", "RelatedTo")
                relation.setdefault("properties", {})
                relation.setdefault("confidence", 0.7)

                cleaned_relations.append(relation)

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations
        }

    def _fallback_pdf_extraction(self, filepath: Union[str, Path]) -> List[str]:
        """Fallback PDF text extraction using basic tools."""
        try:
            import PyPDF2
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

            # Simple chunking
            chunks = []
            chunk_size = 2000
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i + chunk_size])

            return chunks

        except ImportError:
            # Ultimate fallback - just read as text if possible
            logger.warning("No PDF processing library available")
            return [f"Could not process PDF: {filepath}"]

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        stats = {
            "total_files_processed": len(self.processing_history),
            "active_kg_stats": None,
            "processing_history": self.processing_history,
            "config": self.config.to_dict()
        }

        if self.active_kg:
            stats["active_kg_stats"] = {
                "entities": len(self.active_kg.entities),
                "relations": len(self.active_kg.relations),
                "metadata": self.active_kg.metadata
            }

        return stats


def create_session(llm: str = "gpt-4", **kwargs) -> Session:
    """
    Create a new LitKG session with simplified interface.

    Args:
        llm: LLM model to use ("gpt-4", "claude-3.5-sonnet", "ollama/llama3")
        **kwargs: Additional configuration options

    Returns:
        Session: Configured session instance
    """

    # Parse LLM string
    if "/" in llm:
        provider, model = llm.split("/", 1)
    else:
        # Map common model names to providers
        model_map = {
            "gpt-4": ("openai", "gpt-4"),
            "gpt-4o": ("openai", "gpt-4o"),
            "gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
            "claude-3.5-sonnet": ("anthropic", "claude-3-5-sonnet-20241022"),
            "claude-3-opus": ("anthropic", "claude-3-opus-20240229"),
            "gemini-pro": ("google", "gemini-1.5-pro"),
            "gemini-flash": ("google", "gemini-1.5-flash")
        }

        if llm in model_map:
            provider, model = model_map[llm]
        else:
            # Default to OpenAI
            provider, model = "openai", llm

    # Create configuration
    config_kwargs = {
        "llm_provider": provider,
        "llm_model": model,
        **kwargs
    }

    config = Config(**config_kwargs)

    # Create and return session
    return Session(config)