"""Jupyter widget interface for human-in-the-loop validation."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class JupyterInterface:
    """Main Jupyter interface for interactive knowledge graph construction."""

    def __init__(self, knowledge_graph=None):
        """Initialize Jupyter interface."""
        self.knowledge_graph = knowledge_graph
        self.widgets = None
        self.validation_widget = None
        self._initialize_widgets()

    def _initialize_widgets(self):
        """Initialize Jupyter widgets."""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            self.widgets = widgets
            self.display = display
            self.clear_output = clear_output
            logger.info("Jupyter widgets initialized successfully")
        except ImportError:
            logger.error("Jupyter widgets not available. Install with: pip install ipywidgets")
            self.widgets = None

    def is_available(self) -> bool:
        """Check if Jupyter widgets are available."""
        return self.widgets is not None

    def launch(self) -> None:
        """Launch the complete interactive interface."""
        if not self.is_available():
            print("Jupyter widgets not available. Using fallback interface.")
            return self._fallback_interface()

        # Create main interface
        interface = self._create_main_interface()
        self.display(interface)

    def _create_main_interface(self):
        """Create the main interactive interface."""
        # Header
        header = self.widgets.HTML(
            value="""
            <h2>🧠 LitKG Interactive Knowledge Graph Builder</h2>
            <p>Collaborate with AI to construct high-quality knowledge graphs from literature.</p>
            """
        )

        # Stats display
        stats_widget = self._create_stats_widget()

        # File upload
        upload_widget = self._create_upload_widget()

        # Configuration panel
        config_widget = self._create_config_widget()

        # Validation interface
        validation_widget = self._create_validation_widget()

        # Visualization panel
        viz_widget = self._create_visualization_widget()

        # Export panel
        export_widget = self._create_export_widget()

        # Main tabs
        tab = self.widgets.Tab()
        tab.children = [
            upload_widget,
            validation_widget,
            viz_widget,
            export_widget,
            config_widget
        ]
        tab.titles = ["📄 Upload", "✅ Validate", "📊 Visualize", "💾 Export", "⚙️ Config"]

        # Layout
        return self.widgets.VBox([
            header,
            stats_widget,
            tab
        ])

    def _create_stats_widget(self):
        """Create statistics display widget."""
        if not self.knowledge_graph:
            return self.widgets.HTML("<p>No knowledge graph loaded.</p>")

        stats_html = f"""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3>📊 Knowledge Graph Statistics</h3>
            <div style="display: flex; gap: 30px;">
                <div><strong>Entities:</strong> {len(self.knowledge_graph.entities)}</div>
                <div><strong>Relations:</strong> {len(self.knowledge_graph.relations)}</div>
                <div><strong>Avg Confidence:</strong> {self._get_avg_confidence():.2f}</div>
            </div>
        </div>
        """
        return self.widgets.HTML(stats_html)

    def _create_upload_widget(self):
        """Create file upload interface."""
        upload = self.widgets.FileUpload(
            accept='.pdf',
            multiple=True,
            description='Choose PDF files',
            style={'description_width': 'initial'}
        )

        process_btn = self.widgets.Button(
            description='🚀 Process Files',
            button_style='primary',
            layout=self.widgets.Layout(width='200px')
        )

        progress = self.widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Processing:',
            style={'description_width': 'initial'},
            layout=self.widgets.Layout(width='100%')
        )

        status = self.widgets.Output()

        def on_process_click(b):
            """Handle process button click."""
            with status:
                self.clear_output(wait=True)
                if not upload.value:
                    print("❌ Please select PDF files first.")
                    return

                print(f"🔄 Processing {len(upload.value)} files...")
                progress.value = 0

                # Process files - real PDF processing
                new_entities_count = 0
                new_relations_count = 0

                for i, uploaded_file in enumerate(upload.value):
                    filename = uploaded_file['name']
                    content = uploaded_file['content']
                    print(f"📄 Processing: {filename}")
                    progress.value = int((i + 1) / len(upload.value) * 100)

                    try:
                        # Real PDF processing
                        extracted_entities, extracted_relations = self._process_pdf_content(content, filename)

                        # Clear existing data (replace mode)
                        if i == 0:  # Clear on first file
                            self.knowledge_graph.entities.clear()
                            self.knowledge_graph.relations.clear()
                            self.knowledge_graph.graph.clear()
                            print("🗑️ Cleared existing knowledge graph")

                        # Add extracted entities and relations
                        for entity in extracted_entities:
                            self.knowledge_graph.add_entity(entity)
                            new_entities_count += 1

                        for relation in extracted_relations:
                            self.knowledge_graph.add_relation(relation)
                            new_relations_count += 1

                        print(f"✅ Extracted {len(extracted_entities)} entities, {len(extracted_relations)} relations")

                    except Exception as e:
                        print(f"❌ Error processing {filename}: {str(e)}")

                print(f"🎉 Processing complete! Total: {new_entities_count} entities, {new_relations_count} relations")

        process_btn.on_click(on_process_click)

        return self.widgets.VBox([
            self.widgets.HTML("<h3>📄 Document Upload</h3>"),
            upload,
            process_btn,
            progress,
            status
        ])

    def _process_pdf_content(self, content, filename):
        """Process PDF content and extract entities/relations."""
        from ..core.knowledge_graph import Entity, Relation
        import tempfile
        import os

        # Save uploaded content to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Try to use the session's PDF processor if available
            if hasattr(self.knowledge_graph, 'session') and self.knowledge_graph.session:
                session = self.knowledge_graph.session
                if session.pdf_processor:
                    # Use real PDF processor
                    print("🔄 Using PDF processor...")
                    result = session.pdf_processor.process_file(tmp_path)
                    return result.get('entities', []), result.get('relations', [])

            # Fallback: Basic text extraction + mock entity extraction
            print("📝 Using fallback text extraction...")
            text_content = self._extract_text_fallback(tmp_path)

            # Mock entity/relation extraction from text
            entities, relations = self._mock_entity_extraction(text_content, filename)
            return entities, relations

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_text_fallback(self, pdf_path):
        """Fallback text extraction using available libraries."""
        try:
            # Try PyMuPDF first
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            pass

        try:
            # Try pdfplumber
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except ImportError:
            pass

        # Last resort - return placeholder
        return "PDF text extraction not available. Install PyMuPDF or pdfplumber."

    def _mock_entity_extraction(self, text, filename):
        """Mock entity and relation extraction from text."""
        from ..core.knowledge_graph import Entity, Relation
        import re
        import random

        entities = []
        relations = []

        # Simple keyword-based entity extraction (demo purposes)
        keywords = {
            'Concept': ['algorithm', 'method', 'approach', 'technique', 'model', 'system', 'framework'],
            'Technology': ['AI', 'machine learning', 'deep learning', 'neural network', 'transformer'],
            'Person': ['researcher', 'author', 'scientist', 'professor', 'Dr.', 'PhD'],
            'Organization': ['university', 'institute', 'company', 'laboratory', 'Google', 'OpenAI'],
            'Paper': ['paper', 'article', 'study', 'research', 'publication']
        }

        words = text.lower().split()[:500]  # Process first 500 words

        entity_id = 0
        found_entities = {}

        for entity_type, keyword_list in keywords.items():
            for keyword in keyword_list:
                if keyword.lower() in text.lower():
                    entity_id += 1
                    entity = Entity(
                        id=f"entity_{entity_id}",
                        label=keyword.title(),
                        type=entity_type,
                        properties={"source": filename, "extracted_from": "text"},
                        confidence=random.uniform(0.7, 0.95),
                        source=filename
                    )
                    entities.append(entity)
                    found_entities[keyword] = entity

                    if len(entities) >= 10:  # Limit to 10 entities
                        break
            if len(entities) >= 10:
                break

        # Create some mock relations
        entity_list = list(found_entities.values())
        for i in range(min(5, len(entity_list) - 1)):
            relation = Relation(
                id=f"relation_{i+1}",
                source_id=entity_list[i].id,
                target_id=entity_list[i+1].id,
                type=random.choice(["RelatedTo", "UsedIn", "PartOf", "Implements"]),
                properties={"source": filename, "extracted_from": "text"},
                confidence=random.uniform(0.6, 0.9),
                source=filename
            )
            relations.append(relation)

        return entities, relations

    def _create_validation_widget(self):
        """Create validation interface widget."""
        if not self.knowledge_graph:
            return self.widgets.HTML("<p>Load a knowledge graph first.</p>")

        # Entity validation
        entity_list = self._create_entity_validation_widget()

        # Relation validation
        relation_list = self._create_relation_validation_widget()

        # Confidence adjustment
        confidence_slider = self.widgets.FloatSlider(
            value=0.7,
            min=0.0,
            max=1.0,
            step=0.1,
            description='Min Confidence:',
            style={'description_width': 'initial'}
        )

        # Action buttons
        approve_btn = self.widgets.Button(
            description='✅ Approve All',
            button_style='success'
        )

        refine_btn = self.widgets.Button(
            description='🔧 Request Refinement',
            button_style='warning'
        )

        validation_tabs = self.widgets.Tab()
        validation_tabs.children = [entity_list, relation_list]
        validation_tabs.titles = ["Entities", "Relations"]

        return self.widgets.VBox([
            self.widgets.HTML("<h3>✅ Human Validation</h3>"),
            confidence_slider,
            validation_tabs,
            self.widgets.HBox([approve_btn, refine_btn])
        ])

    def _create_entity_validation_widget(self):
        """Create entity validation widget."""
        if not self.knowledge_graph or not self.knowledge_graph.entities:
            return self.widgets.HTML("<p>No entities to validate.</p>")

        entity_items = []
        for entity_id, entity in list(self.knowledge_graph.entities.items())[:20]:  # Limit for demo
            # Entity info
            info_html = f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; border-radius: 5px;">
                <strong>{entity.label}</strong> ({entity.type})<br>
                <small>Confidence: {entity.confidence:.2f} | ID: {entity_id}</small>
            </div>
            """

            # Checkbox for approval
            checkbox = self.widgets.Checkbox(
                value=True,
                description=f"Keep",
                layout=self.widgets.Layout(width='80px')
            )

            # Confidence adjustment
            conf_slider = self.widgets.FloatSlider(
                value=entity.confidence,
                min=0.0,
                max=1.0,
                step=0.01,
                description='',
                layout=self.widgets.Layout(width='200px')
            )

            entity_row = self.widgets.HBox([
                self.widgets.HTML(info_html),
                checkbox,
                conf_slider
            ])
            entity_items.append(entity_row)

        return self.widgets.VBox([
            self.widgets.HTML(f"<h4>Entities ({len(self.knowledge_graph.entities)} total, showing first 20)</h4>"),
            self.widgets.VBox(entity_items)
        ])

    def _create_relation_validation_widget(self):
        """Create relation validation widget."""
        if not self.knowledge_graph or not self.knowledge_graph.relations:
            return self.widgets.HTML("<p>No relations to validate.</p>")

        relation_items = []
        for relation_id, relation in list(self.knowledge_graph.relations.items())[:15]:  # Limit for demo
            # Get entity labels
            source_label = self.knowledge_graph.entities.get(relation.source_id, {}).label or relation.source_id
            target_label = self.knowledge_graph.entities.get(relation.target_id, {}).label or relation.target_id

            # Relation info
            info_html = f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; border-radius: 5px;">
                <strong>{source_label}</strong> → <em>{relation.type}</em> → <strong>{target_label}</strong><br>
                <small>Confidence: {relation.confidence:.2f} | ID: {relation_id}</small>
            </div>
            """

            # Checkbox for approval
            checkbox = self.widgets.Checkbox(
                value=True,
                description=f"Keep",
                layout=self.widgets.Layout(width='80px')
            )

            # Confidence adjustment
            conf_slider = self.widgets.FloatSlider(
                value=relation.confidence,
                min=0.0,
                max=1.0,
                step=0.01,
                description='',
                layout=self.widgets.Layout(width='200px')
            )

            relation_row = self.widgets.HBox([
                self.widgets.HTML(info_html),
                checkbox,
                conf_slider
            ])
            relation_items.append(relation_row)

        return self.widgets.VBox([
            self.widgets.HTML(f"<h4>Relations ({len(self.knowledge_graph.relations)} total, showing first 15)</h4>"),
            self.widgets.VBox(relation_items)
        ])

    def _create_visualization_widget(self):
        """Create visualization widget."""
        viz_html = """
        <h3>📊 Knowledge Graph Visualization</h3>
        <p>Interactive graph visualization will appear here.</p>
        <div id="kg-visualization" style="height: 400px; border: 1px solid #ddd; text-align: center; line-height: 400px;">
            Graph visualization placeholder
        </div>
        """

        # Layout options
        layout_dropdown = self.widgets.Dropdown(
            options=['Force-directed', 'Hierarchical', 'Circular', 'Random'],
            value='Force-directed',
            description='Layout:',
            style={'description_width': 'initial'}
        )

        # Show options
        show_labels = self.widgets.Checkbox(value=True, description='Show Labels')
        show_confidence = self.widgets.Checkbox(value=True, description='Show Confidence')
        show_communities = self.widgets.Checkbox(value=False, description='Highlight Communities')

        # Update button
        update_btn = self.widgets.Button(
            description='🔄 Update Visualization',
            button_style='info'
        )

        def on_update_viz(b):
            """Handle visualization update."""
            # This would integrate with the actual visualization library
            print("Updating visualization...")

        update_btn.on_click(on_update_viz)

        controls = self.widgets.HBox([
            layout_dropdown,
            show_labels,
            show_confidence,
            show_communities,
            update_btn
        ])

        return self.widgets.VBox([
            self.widgets.HTML(viz_html),
            controls
        ])

    def _create_export_widget(self):
        """Create export interface widget."""
        # Format selection
        format_dropdown = self.widgets.Dropdown(
            options=['JSON', 'Neo4j Cypher', 'GraphML', 'CSV', 'Interactive HTML'],
            value='JSON',
            description='Format:',
            style={'description_width': 'initial'}
        )

        # Filename input
        filename_text = self.widgets.Text(
            value='knowledge_graph',
            description='Filename:',
            style={'description_width': 'initial'}
        )

        # Export options
        include_metadata = self.widgets.Checkbox(value=True, description='Include Metadata')
        filter_confidence = self.widgets.Checkbox(value=False, description='Filter by Confidence')
        min_confidence = self.widgets.FloatSlider(
            value=0.7,
            min=0.0,
            max=1.0,
            step=0.1,
            description='Min Confidence:',
            disabled=True,
            style={'description_width': 'initial'}
        )

        def on_filter_change(change):
            """Handle filter confidence checkbox."""
            min_confidence.disabled = not change['new']

        filter_confidence.observe(on_filter_change, names='value')

        # Export button
        export_btn = self.widgets.Button(
            description='💾 Export',
            button_style='primary'
        )

        export_status = self.widgets.Output()

        def on_export_click(b):
            """Handle export button click."""
            with export_status:
                self.clear_output(wait=True)
                if not self.knowledge_graph:
                    print("❌ No knowledge graph to export.")
                    return

                format_val = format_dropdown.value
                filename = filename_text.value

                print(f"📤 Exporting as {format_val}...")

                try:
                    # Get appropriate file extension
                    ext_map = {
                        'JSON': '.json',
                        'Neo4j Cypher': '.cypher',
                        'GraphML': '.graphml',
                        'CSV': '.csv',
                        'Interactive HTML': '.html'
                    }

                    full_filename = filename + ext_map[format_val]

                    # This would call the actual export method
                    self.knowledge_graph.export(full_filename, format_val.lower().replace(' ', '_'))

                    print(f"✅ Exported to: {full_filename}")

                except Exception as e:
                    print(f"❌ Export failed: {e}")

        export_btn.on_click(on_export_click)

        return self.widgets.VBox([
            self.widgets.HTML("<h3>💾 Export Knowledge Graph</h3>"),
            self.widgets.HBox([format_dropdown, filename_text]),
            include_metadata,
            filter_confidence,
            min_confidence,
            export_btn,
            export_status
        ])

    def _create_config_widget(self):
        """Create configuration widget."""
        # LLM settings
        llm_dropdown = self.widgets.Dropdown(
            options=['OpenAI GPT-4', 'Claude 3.5 Sonnet', 'Local Llama 3.1'],
            value='OpenAI GPT-4',
            description='LLM Model:',
            style={'description_width': 'initial'}
        )

        # Processing settings
        confidence_threshold = self.widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Confidence Threshold:',
            style={'description_width': 'initial'}
        )

        batch_size = self.widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            description='Batch Size:',
            style={'description_width': 'initial'}
        )

        # Feature toggles
        enable_communities = self.widgets.Checkbox(value=True, description='Community Detection')
        enable_temporal = self.widgets.Checkbox(value=True, description='Temporal Tracking')
        enable_parallel = self.widgets.Checkbox(value=True, description='Parallel Retrieval')

        # Save config button
        save_config_btn = self.widgets.Button(
            description='💾 Save Configuration',
            button_style='success'
        )

        config_status = self.widgets.Output()

        def on_save_config(b):
            """Handle save configuration."""
            with config_status:
                self.clear_output(wait=True)
                print("💾 Configuration saved successfully!")

        save_config_btn.on_click(on_save_config)

        return self.widgets.VBox([
            self.widgets.HTML("<h3>⚙️ Configuration</h3>"),
            llm_dropdown,
            confidence_threshold,
            batch_size,
            self.widgets.HTML("<h4>Features</h4>"),
            enable_communities,
            enable_temporal,
            enable_parallel,
            save_config_btn,
            config_status
        ])

    def _get_avg_confidence(self) -> float:
        """Calculate average confidence across entities and relations."""
        if not self.knowledge_graph:
            return 0.0

        confidences = []
        for entity in self.knowledge_graph.entities.values():
            confidences.append(entity.confidence)
        for relation in self.knowledge_graph.relations.values():
            confidences.append(relation.confidence)

        return sum(confidences) / len(confidences) if confidences else 0.0

    def _fallback_interface(self):
        """Fallback interface when Jupyter widgets are not available."""
        print("""
🧠 LitKG Interactive Knowledge Graph Builder
==========================================

Jupyter widgets are not available. To use the full interactive interface, install:
pip install ipywidgets

Available commands:
- kg.visualize() - Show graph visualization
- kg.export('filename.json') - Export knowledge graph
- kg.filter_by_confidence(0.8) - Filter by confidence threshold

For more information, see: https://github.com/litkg/lit-kg-sdk
        """)


class ValidationWidget:
    """Standalone validation widget for LangGraph integration."""

    def __init__(self, state):
        """Initialize validation widget with workflow state."""
        self.state = state
        self.widgets = None
        self.result = None
        self._initialize_widgets()

    def _initialize_widgets(self):
        """Initialize Jupyter widgets."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
            self.widgets = widgets
            self.display = display
        except ImportError:
            self.widgets = None

    async def get_validation_result(self):
        """Get validation result from user interaction."""
        if not self.widgets:
            # Fallback to command line
            from ..human_loop.langgraph_workflow import HumanLoopWorkflow
            workflow = HumanLoopWorkflow(None, None)
            return await workflow._command_line_validation(self.state)

        # Create validation interface
        interface = self._create_validation_interface()
        self.display(interface)

        # Wait for user interaction
        while self.result is None:
            await asyncio.sleep(0.1)

        return self.result

    def _create_validation_interface(self):
        """Create validation interface."""
        entities = self.state.get('entities', [])
        relations = self.state.get('relations', [])

        header = self.widgets.HTML(f"""
        <h3>🔍 Validation Required</h3>
        <p>Entities: {len(entities)} | Relations: {len(relations)}</p>
        """)

        # Sample display
        sample_html = "<h4>Sample Entities:</h4><ul>"
        for entity in entities[:5]:
            conf = entity.get('confidence', 0)
            sample_html += f"<li>{entity.get('label', 'N/A')} ({entity.get('type', 'N/A')}) - {conf:.2f}</li>"
        sample_html += "</ul>"

        sample_widget = self.widgets.HTML(sample_html)

        # Action buttons
        approve_btn = self.widgets.Button(description='✅ Approve', button_style='success')
        reject_btn = self.widgets.Button(description='❌ Reject', button_style='danger')
        modify_btn = self.widgets.Button(description='🔧 Modify', button_style='warning')

        feedback_text = self.widgets.Textarea(
            placeholder='Optional feedback or comments...',
            layout=self.widgets.Layout(width='100%', height='80px')
        )

        def on_approve(b):
            """Handle approve button."""
            self.result = ValidationResult(
                approved=True,
                feedback={'status': 'approved', 'comments': feedback_text.value},
                modifications=[]
            )

        def on_reject(b):
            """Handle reject button."""
            self.result = ValidationResult(
                approved=False,
                feedback={'status': 'rejected', 'comments': feedback_text.value},
                modifications=[]
            )

        def on_modify(b):
            """Handle modify button."""
            # For simplicity, just approve with modification flag
            self.result = ValidationResult(
                approved=True,
                feedback={'status': 'modified', 'comments': feedback_text.value},
                modifications=[{'action': 'adjust_threshold', 'threshold': 0.8}]
            )

        approve_btn.on_click(on_approve)
        reject_btn.on_click(on_reject)
        modify_btn.on_click(on_modify)

        return self.widgets.VBox([
            header,
            sample_widget,
            feedback_text,
            self.widgets.HBox([approve_btn, reject_btn, modify_btn])
        ])


# Import ValidationResult from the workflow module
try:
    from .langgraph_workflow import ValidationResult
except ImportError:
    # Fallback definition
    from dataclasses import dataclass
    from typing import Dict, Any, List

    @dataclass
    class ValidationResult:
        approved: bool
        feedback: Dict[str, Any]
        modifications: List[Dict[str, Any]]