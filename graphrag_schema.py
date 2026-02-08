import networkx as nx
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class GraphRAGSchema:
    """
    GraphRAG Schema implementation using NetworkX.
    Defines node labels, relationship types, and their properties.
    """
    
    def __init__(self):
        """Initialize directed graph for GraphRAG."""
        self.graph = nx.DiGraph()
        
        # Define valid node types
        self.NODE_TYPES = {
            'DOCUMENT',
            'CHUNK', 
            'ENTITY',
            'PERSON',
            'ORGANIZATION',
            'CONCEPT'
        }
        
        # Define valid relationship types
        self.RELATIONSHIP_TYPES = {
            'CONTAINS',      # Document -> Chunk
            'MENTIONS',      # Chunk -> Entity/Person/Organization/Concept
            'RELATES_TO',    # Entity <-> Entity
            'NEXT_CHUNK'     # Chunk -> Chunk (sequential)
        }
    
    # ==================== NODE OPERATIONS ====================
    
    def add_document_node(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """
        Add a Document node.
        
        Properties:
        - id: unique document identifier
        - type: 'DOCUMENT'
        - metadata: dict with filename, source, upload_date, etc.
        """
        self.graph.add_node(
            doc_id,
            node_type='DOCUMENT',
            id=doc_id,
            metadata=metadata,
            created_at=datetime.now().isoformat()
        )
    
    def add_chunk_node(
        self, 
        chunk_id: str, 
        text: str, 
        embedding: Optional[np.ndarray] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a Chunk node.
        
        Properties:
        - id: unique chunk identifier
        - text: chunk content
        - type: 'CHUNK'
        - embedding: vector embedding (optional)
        - metadata: dict with chunk_index, doc_id, etc.
        """
        node_attrs = {
            'node_type': 'CHUNK',
            'id': chunk_id,
            'text': text,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        if embedding is not None:
            node_attrs['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        self.graph.add_node(chunk_id, **node_attrs)
    
    def add_entity_node(
        self,
        entity_name: str,
        entity_type: str,  # ENTITY, PERSON, ORGANIZATION, CONCEPT
        description: str = "",
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add an Entity node (generic or specific type).
        
        Properties:
        - id: entity name (used as unique identifier)
        - type: ENTITY/PERSON/ORGANIZATION/CONCEPT
        - text: entity name
        - metadata: dict with description, source_chunks, etc.
        """
        if entity_type not in self.NODE_TYPES:
            entity_type = 'ENTITY'
        
        # Use entity name as ID (normalized)
        entity_id = entity_name.strip().title()
        
        node_attrs = {
            'node_type': entity_type,
            'id': entity_id,
            'text': entity_name,
            'description': description,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        # If entity already exists, merge metadata
        if self.graph.has_node(entity_id):
            existing_attrs = self.graph.nodes[entity_id]
            # Merge source chunks if present
            if 'source_chunks' in (metadata or {}):
                if 'source_chunks' in existing_attrs.get('metadata', {}):
                    existing_attrs['metadata']['source_chunks'].extend(
                        metadata['source_chunks']
                    )
                else:
                    existing_attrs['metadata']['source_chunks'] = metadata['source_chunks']
            self.graph.nodes[entity_id].update(node_attrs)
        else:
            self.graph.add_node(entity_id, **node_attrs)
    
    # ==================== RELATIONSHIP OPERATIONS ====================
    
    def add_contains_relationship(
        self,
        doc_id: str,
        chunk_id: str,
        context: str = ""
    ) -> None:
        """
        Add CONTAINS relationship: Document -> Chunk.
        
        Properties:
        - weight: importance score (default 1.0)
        - confidence: extraction confidence (default 1.0)
        - context: additional context
        """
        self.graph.add_edge(
            doc_id,
            chunk_id,
            relationship_type='CONTAINS',
            weight=1.0,
            confidence=1.0,
            context=context,
            created_at=datetime.now().isoformat()
        )
    
    def add_mentions_relationship(
        self,
        chunk_id: str,
        entity_id: str,
        confidence: float = 1.0,
        context: str = ""
    ) -> None:
        """
        Add MENTIONS relationship: Chunk -> Entity.
        
        Properties:
        - weight: mention frequency or importance
        - confidence: extraction confidence
        - context: surrounding text or description
        """
        # Calculate weight based on existing mentions
        weight = 1.0
        if self.graph.has_edge(chunk_id, entity_id):
            weight = self.graph[chunk_id][entity_id].get('weight', 0) + 1.0
        
        self.graph.add_edge(
            chunk_id,
            entity_id,
            relationship_type='MENTIONS',
            weight=weight,
            confidence=confidence,
            context=context,
            created_at=datetime.now().isoformat()
        )
    
    def add_relates_to_relationship(
        self,
        entity1_id: str,
        entity2_id: str,
        relationship_description: str = "",
        weight: float = 1.0,
        confidence: float = 1.0,
        context: str = ""
    ) -> None:
        """
        Add RELATES_TO relationship: Entity <-> Entity.
        
        Properties:
        - weight: relationship strength
        - confidence: extraction confidence
        - context: relationship description and context
        """
        self.graph.add_edge(
            entity1_id,
            entity2_id,
            relationship_type='RELATES_TO',
            relationship_description=relationship_description,
            weight=weight,
            confidence=confidence,
            context=context,
            created_at=datetime.now().isoformat()
        )
    
    def add_next_chunk_relationship(
        self,
        chunk_id: str,
        next_chunk_id: str,
        context: str = ""
    ) -> None:
        """
        Add NEXT_CHUNK relationship: Chunk -> Chunk (sequential).
        
        Properties:
        - weight: always 1.0 (sequential order)
        - confidence: always 1.0
        - context: additional sequencing info
        """
        self.graph.add_edge(
            chunk_id,
            next_chunk_id,
            relationship_type='NEXT_CHUNK',
            weight=1.0,
            confidence=1.0,
            context=context,
            created_at=datetime.now().isoformat()
        )
    
    # ==================== QUERY OPERATIONS ====================
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node attributes by ID."""
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None
    
    def get_nodes_by_type(self, node_type: str) -> List[tuple]:
        """Get all nodes of a specific type."""
        return [
            (node, attrs) 
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get('node_type') == node_type
        ]
    
    def get_relationships(
        self,
        source_id: str,
        relationship_type: Optional[str] = None
    ) -> List[tuple]:
        """Get all relationships from a source node, optionally filtered by type."""
        edges = []
        for _, target, attrs in self.graph.out_edges(source_id, data=True):
            if relationship_type is None or attrs.get('relationship_type') == relationship_type:
                edges.append((source_id, target, attrs))
        return edges
    
    def get_entity_mentions(self, entity_id: str) -> List[str]:
        """Get all chunks that mention a specific entity."""
        chunks = []
        for source, _, attrs in self.graph.in_edges(entity_id, data=True):
            if attrs.get('relationship_type') == 'MENTIONS':
                chunks.append(source)
        return chunks
    
    def get_chunk_entities(self, chunk_id: str) -> List[str]:
        """Get all entities mentioned in a specific chunk."""
        entities = []
        for _, target, attrs in self.graph.out_edges(chunk_id, data=True):
            if attrs.get('relationship_type') == 'MENTIONS':
                entities.append(target)
        return entities
    
    def get_related_entities(
        self,
        entity_id: str,
        max_hops: int = 1
    ) -> List[tuple]:
        """
        Get entities related to a given entity within max_hops.
        Returns list of (entity_id, distance, path).
        """
        if not self.graph.has_node(entity_id):
            return []
        
        related = []
        try:
            # Use BFS to find related entities
            lengths = nx.single_source_shortest_path_length(
                self.graph,
                entity_id,
                cutoff=max_hops
            )
            
            for target, distance in lengths.items():
                if distance > 0:  # Exclude self
                    node_type = self.graph.nodes[target].get('node_type')
                    if node_type in {'ENTITY', 'PERSON', 'ORGANIZATION', 'CONCEPT'}:
                        path = nx.shortest_path(self.graph, entity_id, target)
                        related.append((target, distance, path))
            
        except nx.NodeNotFound:
            pass
        
        return related
    
    # ==================== UTILITY OPERATIONS ====================
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'relationship_types': {}
        }
        
        # Count nodes by type
        for _, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('node_type', 'UNKNOWN')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count edges by type
        for _, _, attrs in self.graph.edges(data=True):
            rel_type = attrs.get('relationship_type', 'UNKNOWN')
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
        
        return stats
    
    def save_graph(self, filepath: str) -> None:
        """Save graph to file (GraphML format)."""
        nx.write_graphml(self.graph, filepath)
    
    def load_graph(self, filepath: str) -> None:
        """Load graph from file (GraphML format)."""
        self.graph = nx.read_graphml(filepath)


# ==================== USAGE EXAMPLE ====================

def build_graphrag_from_extractions(
    doc_id: str,
    chunks: List[str],
    extracted_entities: List[Dict],
    extracted_relationships: List[Dict],
    embeddings: Optional[List[np.ndarray]] = None
) -> GraphRAGSchema:
    """
    Build complete GraphRAG from extraction results.
    """
    graph_schema = GraphRAGSchema()
    
    # Step 1: Add document node
    graph_schema.add_document_node(
        doc_id=doc_id,
        metadata={
            'filename': f'{doc_id}.pdf',
            'num_chunks': len(chunks),
            'processed_at': datetime.now().isoformat()
        }
    )
    
    # Step 2: Add chunk nodes and CONTAINS relationships
    chunk_ids = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx:04d}"
        chunk_ids.append(chunk_id)
        
        # Add chunk node
        embedding = embeddings[idx] if embeddings else None
        graph_schema.add_chunk_node(
            chunk_id=chunk_id,
            text=chunk,
            embedding=embedding,
            metadata={'chunk_index': idx, 'doc_id': doc_id}
        )
        
        # Add CONTAINS relationship
        graph_schema.add_contains_relationship(
            doc_id=doc_id,
            chunk_id=chunk_id,
            context=f"Chunk {idx} of {len(chunks)}"
        )
    
    # Step 3: Add NEXT_CHUNK relationships
    for i in range(len(chunk_ids) - 1):
        graph_schema.add_next_chunk_relationship(
            chunk_id=chunk_ids[i],
            next_chunk_id=chunk_ids[i + 1]
        )
    
    # Step 4: Add entity nodes
    for entity in extracted_entities:
        graph_schema.add_entity_node(
            entity_name=entity['name'],
            entity_type=entity['type'],
            description=entity.get('description', ''),
            metadata={
                'source_chunks': [entity.get('source_chunk_id')],
                'source_doc_id': entity.get('source_doc_id')
            }
        )
    
    # Step 5: Add MENTIONS relationships (Chunk -> Entity)
    for entity in extracted_entities:
        entity_id = entity['name'].strip().title()
        chunk_id = entity.get('source_chunk_id')
        
        if chunk_id:
            graph_schema.add_mentions_relationship(
                chunk_id=chunk_id,
                entity_id=entity_id,
                confidence=0.9,
                context=entity.get('description', '')
            )
    
    # Step 6: Add RELATES_TO relationships (Entity <-> Entity)
    for rel in extracted_relationships:
        source_id = rel['source'].strip().title()
        target_id = rel['target'].strip().title()
        
        graph_schema.add_relates_to_relationship(
            entity1_id=source_id,
            entity2_id=target_id,
            relationship_description=rel.get('relationship', 'RELATED_TO'),
            weight=1.0,
            confidence=0.85,
            context=rel.get('description', '')
        )
    
    return graph_schema
