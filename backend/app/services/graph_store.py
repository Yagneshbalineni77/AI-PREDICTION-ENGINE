"""
Local Graph Store — Drop-in replacement for Zep Cloud.

Uses SQLite for storage and Google Gemini Embeddings for semantic search.
Implements the same API surface as zep_cloud.client.Zep so consuming code
only needs import swaps.
"""

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.graph_store')

# Default database path
_DB_DIR = os.path.join(os.path.dirname(__file__), '../../data')
_DB_PATH = os.path.join(_DB_DIR, 'graph_store.db')


# ─── Data classes matching Zep's return types ───────────────────────────

@dataclass
class NodeData:
    """Mirrors Zep node object."""
    uuid_: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    created_at: Optional[str] = None
    name_embedding: Optional[List[float]] = None

    @property
    def uuid(self):
        return self.uuid_


@dataclass
class EdgeData:
    """Mirrors Zep edge object."""
    uuid_: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: Dict[str, Any]
    created_at: Optional[str] = None

    @property
    def uuid(self):
        return self.uuid_


@dataclass
class EpisodeData:
    """Mirrors Zep EpisodeData."""
    data: str
    type: str = "text"


@dataclass
class EpisodeResult:
    """Mirrors Zep episode result."""
    uuid_: str
    status: str = "processed"
    data: str = ""

    @property
    def uuid(self):
        return self.uuid_
        
    @property
    def processed(self) -> bool:
        return self.status == "processed"


@dataclass
class EntityEdgeSourceTarget:
    """Mirrors Zep EntityEdgeSourceTarget."""
    source: str = "Entity"
    target: str = "Entity"


@dataclass
class SearchResultEdge:
    """Single edge in search results."""
    uuid_: str = ""
    name: str = ""
    fact: str = ""
    source_node_uuid: str = ""
    target_node_uuid: str = ""

    @property
    def uuid(self):
        return self.uuid_


@dataclass
class SearchResultNode:
    """Single node in search results."""
    uuid_: str = ""
    name: str = ""
    labels: List[str] = field(default_factory=list)
    summary: str = ""

    @property
    def uuid(self):
        return self.uuid_


@dataclass
class SearchResults:
    """Mirrors Zep search results."""
    edges: List[SearchResultEdge] = field(default_factory=list)
    nodes: List[SearchResultNode] = field(default_factory=list)


# ─── Embedding helper ──────────────────────────────────────────────────

def _get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Gemini API (text-embedding-004)."""
    if not text or not text.strip():
        return None
    try:
        import requests
        api_key = Config.LLM_API_KEY
        if not api_key:
            return None

        # Use Gemini embedding endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
        resp = requests.post(url, json={
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text[:2000]}]}  # Limit text length
        }, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            return data.get("embedding", {}).get("values", None)
        else:
            logger.warning(f"Embedding API returned {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─── SQLite Database ───────────────────────────────────────────────────

class _DB:
    """Thread-safe SQLite connection manager."""

    def __init__(self, db_path: str = _DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS graphs (
                        graph_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        ontology TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT (datetime('now'))
                    );
                    CREATE TABLE IF NOT EXISTS nodes (
                        uuid TEXT PRIMARY KEY,
                        graph_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        labels TEXT DEFAULT '[]',
                        summary TEXT DEFAULT '',
                        attributes TEXT DEFAULT '{}',
                        embedding TEXT DEFAULT NULL,
                        created_at TEXT DEFAULT (datetime('now')),
                        FOREIGN KEY (graph_id) REFERENCES graphs(graph_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS edges (
                        uuid TEXT PRIMARY KEY,
                        graph_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        fact TEXT DEFAULT '',
                        source_node_uuid TEXT NOT NULL,
                        target_node_uuid TEXT NOT NULL,
                        attributes TEXT DEFAULT '{}',
                        embedding TEXT DEFAULT NULL,
                        created_at TEXT DEFAULT (datetime('now')),
                        FOREIGN KEY (graph_id) REFERENCES graphs(graph_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS episodes (
                        uuid TEXT PRIMARY KEY,
                        graph_id TEXT NOT NULL,
                        data TEXT NOT NULL,
                        type TEXT DEFAULT 'text',
                        status TEXT DEFAULT 'processed',
                        created_at TEXT DEFAULT (datetime('now')),
                        FOREIGN KEY (graph_id) REFERENCES graphs(graph_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_nodes_graph ON nodes(graph_id);
                    CREATE INDEX IF NOT EXISTS idx_edges_graph ON edges(graph_id);
                    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_uuid);
                    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_uuid);
                    CREATE INDEX IF NOT EXISTS idx_episodes_graph ON episodes(graph_id);
                """)
                conn.commit()
            finally:
                conn.close()

    def execute(self, sql, params=None):
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(sql, params or ())
                conn.commit()
                return cur
            finally:
                conn.close()

    def executemany(self, sql, params_list):
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.executemany(sql, params_list)
                conn.commit()
                return cur
            finally:
                conn.close()

    def fetchone(self, sql, params=None):
        with self._lock:
            conn = self._get_conn()
            try:
                return conn.execute(sql, params or ()).fetchone()
            finally:
                conn.close()

    def fetchall(self, sql, params=None):
        with self._lock:
            conn = self._get_conn()
            try:
                return conn.execute(sql, params or ()).fetchall()
            finally:
                conn.close()


# ─── Graph sub-API classes (mirrors Zep client.graph.xxx) ──────────────

class _NodeAPI:
    """Mirrors client.graph.node.*"""

    def __init__(self, db: _DB):
        self._db = db

    def get(self, uuid_: str) -> Optional[NodeData]:
        """Get a single node by UUID."""
        row = self._db.fetchone("SELECT * FROM nodes WHERE uuid = ?", (uuid_,))
        if not row:
            return None
        return NodeData(
            uuid_=row["uuid"],
            name=row["name"],
            labels=json.loads(row["labels"]),
            summary=row["summary"],
            attributes=json.loads(row["attributes"]),
            created_at=row["created_at"],
        )

    def get_by_graph_id(
        self,
        graph_id: str,
        limit: int = 100,
        uuid_cursor: Optional[str] = None,
    ) -> List[NodeData]:
        """Get all nodes for a graph (paginated)."""
        if uuid_cursor:
            rows = self._db.fetchall(
                "SELECT * FROM nodes WHERE graph_id = ? AND uuid > ? ORDER BY uuid LIMIT ?",
                (graph_id, uuid_cursor, limit)
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM nodes WHERE graph_id = ? ORDER BY uuid LIMIT ?",
                (graph_id, limit)
            )
        return [
            NodeData(
                uuid_=r["uuid"], name=r["name"],
                labels=json.loads(r["labels"]),
                summary=r["summary"],
                attributes=json.loads(r["attributes"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_entity_edges(self, node_uuid: str) -> List[EdgeData]:
        """Get all edges connected to a node."""
        rows = self._db.fetchall(
            "SELECT * FROM edges WHERE source_node_uuid = ? OR target_node_uuid = ?",
            (node_uuid, node_uuid)
        )
        return [
            EdgeData(
                uuid_=r["uuid"], name=r["name"], fact=r["fact"],
                source_node_uuid=r["source_node_uuid"],
                target_node_uuid=r["target_node_uuid"],
                attributes=json.loads(r["attributes"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]


class _EdgeAPI:
    """Mirrors client.graph.edge.*"""

    def __init__(self, db: _DB):
        self._db = db

    def get_by_graph_id(
        self,
        graph_id: str,
        limit: int = 100,
        uuid_cursor: Optional[str] = None,
    ) -> List[EdgeData]:
        """Get all edges for a graph (paginated)."""
        if uuid_cursor:
            rows = self._db.fetchall(
                "SELECT * FROM edges WHERE graph_id = ? AND uuid > ? ORDER BY uuid LIMIT ?",
                (graph_id, uuid_cursor, limit)
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM edges WHERE graph_id = ? ORDER BY uuid LIMIT ?",
                (graph_id, limit)
            )
        return [
            EdgeData(
                uuid_=r["uuid"], name=r["name"], fact=r["fact"],
                source_node_uuid=r["source_node_uuid"],
                target_node_uuid=r["target_node_uuid"],
                attributes=json.loads(r["attributes"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]


class _EpisodeAPI:
    """Mirrors client.graph.episode.*"""

    def __init__(self, db: _DB):
        self._db = db

    def get(self, uuid_: str) -> Optional[EpisodeResult]:
        """Get episode by UUID."""
        row = self._db.fetchone("SELECT * FROM episodes WHERE uuid = ?", (uuid_,))
        if not row:
            return None
        return EpisodeResult(
            uuid_=row["uuid"],
            status=row["status"],
            data=row["data"],
        )


class _GraphAPI:
    """Mirrors client.graph.* — the main API surface."""

    def __init__(self, db: _DB):
        self._db = db
        self.node = _NodeAPI(db)
        self.edge = _EdgeAPI(db)
        self.episode = _EpisodeAPI(db)

    def create(self, graph_id: str, name: str, description: str = "") -> str:
        """Create a new graph."""
        self._db.execute(
            "INSERT OR REPLACE INTO graphs (graph_id, name, description) VALUES (?, ?, ?)",
            (graph_id, name, description)
        )
        logger.info(f"Created graph: {graph_id} ({name})")
        return graph_id

    def delete(self, graph_id: str):
        """Delete a graph and all its data."""
        self._db.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
        logger.info(f"Deleted graph: {graph_id}")

    def set_ontology(self, graph_ids: List[str], entities=None, edges=None):
        """Store ontology schema for graphs.
        
        Simplified: we just store the type names so entity extraction 
        knows what types to look for.
        """
        ontology_data = {"entity_types": [], "edge_types": []}

        if entities:
            for name, cls in entities.items():
                ontology_data["entity_types"].append({
                    "name": name,
                    "description": getattr(cls, '__doc__', '') or '',
                })

        if edges:
            for name, (cls, source_targets) in edges.items():
                st_list = []
                for st in source_targets:
                    st_list.append({
                        "source": getattr(st, 'source', 'Entity'),
                        "target": getattr(st, 'target', 'Entity'),
                    })
                ontology_data["edge_types"].append({
                    "name": name,
                    "description": getattr(cls, '__doc__', '') or '',
                    "source_targets": st_list,
                })

        ontology_json = json.dumps(ontology_data, ensure_ascii=False)
        for gid in graph_ids:
            self._db.execute(
                "UPDATE graphs SET ontology = ? WHERE graph_id = ?",
                (ontology_json, gid)
            )
        logger.info(f"Set ontology for {len(graph_ids)} graph(s): "
                     f"{len(ontology_data['entity_types'])} entity types, "
                     f"{len(ontology_data['edge_types'])} edge types")

    def add(self, graph_id: str, data: str, type: str = "text") -> EpisodeResult:
        """Add a single text episode and extract entities/edges."""
        ep_uuid = uuid.uuid4().hex
        self._db.execute(
            "INSERT INTO episodes (uuid, graph_id, data, type, status) VALUES (?, ?, ?, ?, ?)",
            (ep_uuid, graph_id, data, type, "processed")
        )

        # Extract entities and relationships from text
        self._extract_and_store(graph_id, data)

        return EpisodeResult(uuid_=ep_uuid, status="processed", data=data)

    def add_batch(
        self, graph_id: str, episodes: List[EpisodeData]
    ) -> List[EpisodeResult]:
        """Add multiple episodes, extract entities/edges from each concurrently."""
        import uuid
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        ep_data_list = []
        
        # First quickly insert all episodes into DB
        for ep in episodes:
            ep_uuid = uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO episodes (uuid, graph_id, data, type, status) VALUES (?, ?, ?, ?, ?)",
                (ep_uuid, graph_id, ep.data, ep.type, "processed")
            )
            results.append(EpisodeResult(uuid_=ep_uuid, status="processed", data=ep.data))
            ep_data_list.append(ep.data)
            
        # Then process LLM extraction in parallel (up to 5 concurrently)
        logger.info(f"Extracting entities from {len(episodes)} episodes in parallel...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            for data in ep_data_list:
                executor.submit(self._extract_and_store, graph_id, data)
                
        return results

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str = "cross_encoder",
    ) -> SearchResults:
        """Semantic search over graph nodes/edges using embeddings."""
        query_embedding = _get_embedding(query)

        result_edges = []
        result_nodes = []

        if scope in ("edges", "both"):
            rows = self._db.fetchall(
                "SELECT * FROM edges WHERE graph_id = ? AND embedding IS NOT NULL",
                (graph_id,)
            )
            scored = []
            for r in rows:
                emb = json.loads(r["embedding"]) if r["embedding"] else None
                if emb and query_embedding:
                    score = _cosine_similarity(query_embedding, emb)
                else:
                    # Fallback: keyword match
                    score = 1.0 if query.lower() in (r["fact"] or "").lower() else 0.0
                scored.append((score, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            for score, r in scored[:limit]:
                if score > 0.1:  # Minimum threshold
                    result_edges.append(SearchResultEdge(
                        uuid_=r["uuid"], name=r["name"], fact=r["fact"],
                        source_node_uuid=r["source_node_uuid"],
                        target_node_uuid=r["target_node_uuid"],
                    ))

        if scope in ("nodes", "both"):
            rows = self._db.fetchall(
                "SELECT * FROM nodes WHERE graph_id = ? AND embedding IS NOT NULL",
                (graph_id,)
            )
            scored = []
            for r in rows:
                emb = json.loads(r["embedding"]) if r["embedding"] else None
                if emb and query_embedding:
                    score = _cosine_similarity(query_embedding, emb)
                else:
                    score = 1.0 if query.lower() in (r["summary"] or "").lower() else 0.0
                scored.append((score, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            for score, r in scored[:limit]:
                if score > 0.1:
                    result_nodes.append(SearchResultNode(
                        uuid_=r["uuid"], name=r["name"],
                        labels=json.loads(r["labels"]),
                        summary=r["summary"],
                    ))

        # If no embedding results, fall back to keyword search
        if not result_edges and not result_nodes:
            return self._keyword_search(graph_id, query, limit, scope)

        return SearchResults(edges=result_edges, nodes=result_nodes)

    def _keyword_search(
        self, graph_id: str, query: str, limit: int, scope: str
    ) -> SearchResults:
        """Fallback keyword-based search."""
        result_edges = []
        result_nodes = []
        q = f"%{query}%"

        if scope in ("edges", "both"):
            rows = self._db.fetchall(
                "SELECT * FROM edges WHERE graph_id = ? AND (fact LIKE ? OR name LIKE ?) LIMIT ?",
                (graph_id, q, q, limit)
            )
            for r in rows:
                result_edges.append(SearchResultEdge(
                    uuid_=r["uuid"], name=r["name"], fact=r["fact"],
                    source_node_uuid=r["source_node_uuid"],
                    target_node_uuid=r["target_node_uuid"],
                ))

        if scope in ("nodes", "both"):
            rows = self._db.fetchall(
                "SELECT * FROM nodes WHERE graph_id = ? AND (summary LIKE ? OR name LIKE ?) LIMIT ?",
                (graph_id, q, q, limit)
            )
            for r in rows:
                result_nodes.append(SearchResultNode(
                    uuid_=r["uuid"], name=r["name"],
                    labels=json.loads(r["labels"]),
                    summary=r["summary"],
                ))

        return SearchResults(edges=result_edges, nodes=result_nodes)

    def _extract_and_store(self, graph_id: str, text: str):
        """Extract entities and relationships from text using Gemini LLM, then store."""
        if not text or len(text.strip()) < 10:
            return

        # Get ontology to know what entity/edge types to look for
        row = self._db.fetchone(
            "SELECT ontology FROM graphs WHERE graph_id = ?", (graph_id,)
        )
        ontology = json.loads(row["ontology"]) if row and row["ontology"] else {}
        entity_types = [et["name"] for et in ontology.get("entity_types", [])]
        edge_types = [et["name"] for et in ontology.get("edge_types", [])]

        # Use LLM to extract entities and relationships
        try:
            from ..utils.llm_client import LLMClient

            type_hint = ""
            if entity_types:
                type_hint += f"\nKnown entity types: {', '.join(entity_types)}"
            if edge_types:
                type_hint += f"\nKnown relationship types: {', '.join(edge_types)}"

            prompt = f"""Extract entities and relationships from the following text.
{type_hint}

Return a JSON object with:
- "entities": list of {{"name": "...", "type": "...", "summary": "brief description"}}
- "relationships": list of {{"source": "entity_name", "target": "entity_name", "type": "relationship_type", "fact": "description of relationship"}}

Only extract clearly stated facts. Be concise.

Text:
{text[:3000]}"""

            llm = LLMClient()
            response = llm.chat(
                messages=[
                    {"role": "system", "content": "You are an entity/relationship extraction engine. Return ONLY valid JSON, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            # Parse response
            content = response.strip()
            # Remove markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                elif "```" in content:
                    content = content[:content.rfind("```")]

            extracted = json.loads(content)

            # Store entities as nodes
            node_name_to_uuid = {}
            for entity in extracted.get("entities", []):
                name = entity.get("name", "").strip()
                if not name:
                    continue

                # Check if node already exists in this graph
                existing = self._db.fetchone(
                    "SELECT uuid FROM nodes WHERE graph_id = ? AND name = ?",
                    (graph_id, name)
                )
                if existing:
                    node_uuid = existing["uuid"]
                    # Update summary if better
                    if entity.get("summary"):
                        self._db.execute(
                            "UPDATE nodes SET summary = ? WHERE uuid = ?",
                            (entity["summary"], node_uuid)
                        )
                else:
                    node_uuid = uuid.uuid4().hex
                    labels = ["Entity"]
                    if entity.get("type"):
                        labels.append(entity["type"])

                    # Get embedding for the node
                    embedding = _get_embedding(f"{name}: {entity.get('summary', '')}")
                    embedding_json = json.dumps(embedding) if embedding else None

                    self._db.execute(
                        "INSERT INTO nodes (uuid, graph_id, name, labels, summary, attributes, embedding) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (node_uuid, graph_id, name, json.dumps(labels),
                         entity.get("summary", ""), json.dumps({}), embedding_json)
                    )

                node_name_to_uuid[name] = node_uuid

            # Store relationships as edges
            for rel in extracted.get("relationships", []):
                source_name = rel.get("source", "").strip()
                target_name = rel.get("target", "").strip()
                if not source_name or not target_name:
                    continue

                source_uuid = node_name_to_uuid.get(source_name)
                target_uuid = node_name_to_uuid.get(target_name)

                # If source/target not in current extraction, try to find in DB
                if not source_uuid:
                    row = self._db.fetchone(
                        "SELECT uuid FROM nodes WHERE graph_id = ? AND name = ?",
                        (graph_id, source_name)
                    )
                    if row:
                        source_uuid = row["uuid"]
                if not target_uuid:
                    row = self._db.fetchone(
                        "SELECT uuid FROM nodes WHERE graph_id = ? AND name = ?",
                        (graph_id, target_name)
                    )
                    if row:
                        target_uuid = row["uuid"]

                if not source_uuid or not target_uuid:
                    continue

                edge_uuid = uuid.uuid4().hex
                fact = rel.get("fact", f"{source_name} {rel.get('type', 'related_to')} {target_name}")

                # Get embedding for the edge fact
                embedding = _get_embedding(fact)
                embedding_json = json.dumps(embedding) if embedding else None

                self._db.execute(
                    "INSERT INTO edges (uuid, graph_id, name, fact, source_node_uuid, target_node_uuid, attributes, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (edge_uuid, graph_id, rel.get("type", "related_to"), fact,
                     source_uuid, target_uuid, json.dumps({}), embedding_json)
                )

            entity_count = len(extracted.get("entities", []))
            rel_count = len(extracted.get("relationships", []))
            if entity_count > 0 or rel_count > 0:
                logger.info(f"Extracted {entity_count} entities, {rel_count} relationships from text ({len(text)} chars)")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM extraction response: {e}")
        except Exception as e:
            logger.warning(f"Entity extraction failed (non-critical): {e}")


# ─── Main GraphStore class (drop-in for Zep client) ───────────────────

class GraphStore:
    """
    Drop-in replacement for zep_cloud.client.Zep.
    
    Usage (same as Zep):
        client = GraphStore()
        client.graph.create(graph_id="...", name="...")
        client.graph.add(graph_id="...", data="...", type="text")
        results = client.graph.search(graph_id="...", query="...")
        nodes = client.graph.node.get_by_graph_id(graph_id)
    """

    def __init__(self, api_key: Optional[str] = None, db_path: str = _DB_PATH):
        """Initialize. api_key is accepted for compatibility but ignored."""
        self._db = _DB(db_path)
        self.graph = _GraphAPI(self._db)
        logger.info(f"GraphStore initialized (SQLite: {db_path})")
