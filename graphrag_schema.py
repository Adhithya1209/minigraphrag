import networkx as nx
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from networkx.readwrite import json_graph
import json
from neo4j import GraphDatabase

        
class GraphRAGSchema:
    """
    GraphRAG Schema implementation using NetworkX.
    Defines node labels, relationship types, and their properties.
    """
    
    def __init__(self):
        """Initialize directed graph for GraphRAG."""
        self.graph = nx.DiGraph()
        self.neo4j_driver = None

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
    
    def save_graph(self, filepath: str = "graph.json"):
        """Save graph in JSON format (supports all Python types)"""
        graph_data = json_graph.node_link_data(self.graph)
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph saved to {filepath}")

    def load_graph(self, filepath: str = "graph.json"):
        """Load graph from JSON format"""

        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        self.graph = json_graph.node_link_graph(graph_data)
        print(f"Graph loaded from {filepath}")

    def save_networkx_to_neo4j(self, uri: str = "neo4j+s://3bbc5bbf.databases.neo4j.io",
                                user: str = "neo4j",
                                password: str = "",
                                clear_existing: bool = False):
        """
        Save NetworkX graph directly to Neo4j with proper data serialization.
        """
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            if clear_existing:
                session.run("MATCH (n) DETACH DELETE n")
                print("✓ Cleared existing Neo4j data")
            
            # Batch create nodes
            print("Creating nodes...")
            node_batch = []
            
            for node_id, data in self.graph.nodes(data=True):
                node_props = self._prepare_neo4j_properties(data)
                node_props['node_id'] = node_id
                label = data.get('type', 'Node')
                
                node_batch.append({
                    'id': node_id,
                    'label': label,
                    'props': node_props
                })
                
                # Batch insert every 1000 nodes
                if len(node_batch) >= 1000:
                    self._batch_create_nodes(session, node_batch)
                    node_batch = []
            
            # Insert remaining nodes
            if node_batch:
                self._batch_create_nodes(session, node_batch)
            
            # Batch create relationships
            print("Creating relationships...")
            rel_batch = []
            
            for source, target, data in self.graph.edges(data=True):
                rel_props = self._prepare_neo4j_properties(data)
                rel_type = data.get('type', 'RELATES_TO')
                
                rel_batch.append({
                    'source': source,
                    'target': target,
                    'type': rel_type,
                    'props': rel_props
                })
                
                # Batch insert every 1000 relationships
                if len(rel_batch) >= 1000:
                    self._batch_create_relationships(session, rel_batch)
                    rel_batch = []
            
            # Insert remaining relationships
            if rel_batch:
                self._batch_create_relationships(session, rel_batch)
            
            print(f"✓ NetworkX graph saved to Neo4j")
            print(f"  Nodes: {self.graph.number_of_nodes()}")
            print(f"  Edges: {self.graph.number_of_edges()}")
        
        driver.close()
    
    def _prepare_neo4j_properties(self, data: Dict) -> Dict:
        """
        Prepare properties for Neo4j (handle numpy arrays, dicts, etc.).
        """
        props = {}
        
        for key, value in data.items():
            if key == 'type':  # Skip type, it's used as label
                continue
            elif isinstance(value, np.ndarray):
                # Convert numpy array to list
                props[key] = value.tolist()
            elif isinstance(value, (dict, list)):
                # Serialize to JSON string
                props[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)):
                props[key] = value
            elif value is None:
                continue  # Skip None values
            else:
                props[key] = str(value)
        
        return props
    
    def _batch_create_nodes(self, session, node_batch: List[Dict]):
        """Batch create nodes in Neo4j."""
        cypher = """
        UNWIND $batch as node
        CALL apoc.create.node([node.label], node.props) YIELD node as n
        RETURN count(n)
        """
        
        # Fallback if APOC not available
        try:
            session.run(cypher, batch=node_batch)
        except:
            # Create nodes one by one without APOC
            for node in node_batch:
                label = node['label']
                props = node['props']
                session.run(
                    f"CREATE (n:{label} $props)",
                    props=props
                )
    
    def _batch_create_relationships(self, session, rel_batch: List[Dict]):
        """Batch create relationships in Neo4j."""
        cypher = """
        UNWIND $batch as rel
        MATCH (a {node_id: rel.source})
        MATCH (b {node_id: rel.target})
        CALL apoc.create.relationship(a, rel.type, rel.props, b) YIELD rel as r
        RETURN count(r)
        """
        
        # Fallback if APOC not available
        try:
            session.run(cypher, batch=rel_batch)
        except:
            # Create relationships one by one without APOC
            for rel in rel_batch:
                rel_type = rel['type']
                props = rel['props']
                session.run(
                    f"""
                    MATCH (a {{node_id: $source}})
                    MATCH (b {{node_id: $target}})
                    CREATE (a)-[r:{rel_type} $props]->(b)
                    """,
                    source=rel['source'],
                    target=rel['target'],
                    props=props
                )