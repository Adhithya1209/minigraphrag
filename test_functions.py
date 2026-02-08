from graphrag_schema import GraphRAGSchema
import networkx as nx
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


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


# Example usage:
if __name__ == "__main__":
    # Sample data
    doc_id = "research_paper_001"
    chunks = ["Text of chunk 1...", "Text of chunk 2..."]
    entities = [
        {'name': 'GraphRAG', 'type': 'CONCEPT', 'description': 'Graph-based retrieval', 'source_chunk_id': 'research_paper_001_chunk_0000'},
        {'name': 'Knowledge Graph', 'type': 'CONCEPT', 'description': 'Structured knowledge', 'source_chunk_id': 'research_paper_001_chunk_0000'}
    ]
    relationships = [
        {'source': 'GraphRAG', 'target': 'Knowledge Graph', 'relationship': 'USES', 'description': 'GraphRAG uses knowledge graphs'}
    ]
    
    # Build graph
    graph = build_graphrag_from_extractions(doc_id, chunks, entities, relationships)
    
    # Get stats
    print(graph.get_graph_stats())
     # Save graph
    graph.save_graph("graphrag_schema.json")
    graph.save_networkx_to_neo4j()