"""
ZepRetrievalToolService
Encapsulates graph search, NodeRead, 边Query等Tool, 供Report Agent使用

Core retrieval tools(After optimization): 
1. InsightForge(Deep insight retrieval)- 最强大的混合Retrieval, Auto-generate sub-questions并Multi-dimensional retrieval
2. PanoramaSearch(广度Search)- Get全貌, 包括过期Content
3. QuickSearch(简单Search)- 快速Retrieval
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .graph_store import GraphStore

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_tools')


@dataclass
class SearchResult:
    """SearchResult"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """Convert to text format for LLM understanding"""
        text_parts = [f"SearchQuery: {self.query}", f"找到 {self.total_count} 条相关Info"]
        
        if self.facts:
            text_parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """NodeInfo"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """Convert to text format"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "未知Type")
        return f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge info"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # 时间Info
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relation: {source} --[{self.name}]--> {target}\n事实: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "未知"
            invalid_at = self.invalid_at or "至今"
            base_text += f"\n时效: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (已过期: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """Whether expired"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """Whether invalid"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Deep insight retrieval results (InsightForge)
    Contains multiple sub-question retrieval results and comprehensive analysis
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    # 各维度Retrieval results
    semantic_facts: List[str] = field(default_factory=list)  # 语义SearchResult
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # Entity洞察
    relationship_chains: List[str] = field(default_factory=list)  # Relationship chain
    
    # StatisticsInfo
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """Convert to detailed text format for LLM understanding"""
        text_parts = [
            f"## 未来Prediction深度Analysis",
            f"Analysis问题: {self.query}",
            f"PredictionScenario: {self.simulation_requirement}",
            f"\n### PredictionDataStatistics",
            f"- 相关Prediction事实: {self.total_facts}条",
            f"- 涉及Entity: {self.total_entities}个",
            f"- Relationship chain: {self.total_relationships}条"
        ]
        
        # Sub-question
        if self.sub_queries:
            text_parts.append(f"\n### Analysis的Sub-question")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # 语义SearchResult
        if self.semantic_facts:
            text_parts.append(f"\n### 【关键事实】(请在Report中引用这些原文)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Entity洞察
        if self.entity_insights:
            text_parts.append(f"\n### 【核心Entity】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  Summary: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  相关事实: {len(entity.get('related_facts', []))}条")
        
        # Relationship chain
        if self.relationship_chains:
            text_parts.append(f"\n### 【Relationship chain】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Panoramic search result (Panorama)
    Contains all related info including expired content
    """
    query: str
    
    # 全部Node
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # All edges (including expired ones)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # 当前Valid的事实
    active_facts: List[str] = field(default_factory=list)
    # 已过期/失效的事实(历史Record)
    historical_facts: List[str] = field(default_factory=list)
    
    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """Convert to text format (full version, no truncation)"""
        text_parts = [
            f"## Panoramic search result(未来全景视图)",
            f"Query: {self.query}",
            f"\n### StatisticsInfo",
            f"- 总Node数: {self.total_nodes}",
            f"- 总边数: {self.total_edges}",
            f"- 当前Valid事实: {self.active_count}条",
            f"- 历史/过期事实: {self.historical_count}条"
        ]
        
        # 当前Valid的事实(完整Output, 不截断)
        if self.active_facts:
            text_parts.append(f"\n### 【当前Valid事实】(SimulationResult原文)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # 历史/过期事实(完整Output, 不截断)
        if self.historical_facts:
            text_parts.append(f"\n### 【历史/过期事实】(演变过程Record)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # 关键Entity(完整Output, 不截断)
        if self.all_nodes:
            text_parts.append(f"\n### 【涉及Entity】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Single agent interview result"""
    agent_name: str
    agent_role: str  # RoleType(e.g., students, teachers, media, etc.)
    agent_bio: str  # 简介
    question: str  # Interview问题
    response: str  # Interview回答
    key_quotes: List[str] = field(default_factory=list)  # 关键引言
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # 显示完整的agent_bio, 不截断
        text += f"_简介: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**关键引言:**\n"
            for quote in self.key_quotes:
                # 清理各种引号
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # 去掉开头的标点
                while clean_quote and clean_quote[0] in ', ,;;: :, .!?\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Filter out junk containing question numbersContent(问题1-9)
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # 截断过长Content(truncate at period, not hard truncation)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    InterviewResult (Interview)
    Contains multiple simulation agent interview responses
    """
    interview_topic: str  # Interview主题
    interview_questions: List[str]  # Interview问题List
    
    # Interview选择的Agent
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # 各Agent的Interview回答
    interviews: List[AgentInterview] = field(default_factory=list)
    
    # 选择Agent的理由
    selection_reasoning: str = ""
    # Integrate后的InterviewSummary
    summary: str = ""
    
    # Statistics
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """Convert to detailed text format for LLM understanding and report citation"""
        text_parts = [
            "## 深度InterviewReport",
            f"**Interview主题:** {self.interview_topic}",
            f"**Interview人数:** {self.interviewed_count} / {self.total_agents} 位SimulationAgent",
            "\n### Interview对象选择理由",
            self.selection_reasoning or "(Auto选择)",
            "\n---",
            "\n### Interview实录",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(无InterviewRecord)\n\n---")

        text_parts.append("\n### InterviewSummary与核心观点")
        text_parts.append(self.summary or "(无Summary)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    ZepRetrievalToolService
    
    【Core retrieval tools - After optimization】
    1. insight_forge - Deep insight retrieval(最强大, Auto-generate sub-questions, Multi-dimensional retrieval)
    2. panorama_search - 广度Search(Get全貌, 包括过期Content)
    3. quick_search - 简单Search(快速Retrieval)
    4. interview_agents - 深度Interview(InterviewSimulationAgent, Get多视角观点)
    
    【基础Tool】
    - search_graph - Graph语义Search
    - get_all_nodes - GetGraph所有Node
    - get_all_edges - GetGraph所有边(含时间Info)
    - get_node_detail - GetNode详细Info
    - get_node_edges - GetNode相关的边
    - get_entities_by_type - 按TypeGet entity
    - get_entity_summary - Get entity的RelationSummary
    """
    
    # RetryConfig
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY 未Config")
        
        self.client = GraphStore(api_key=self.api_key)
        # LLM客户端用于InsightForgeGenerateSub-question
        self._llm_client = llm_client
        logger.info("ZepToolsService InitializeComplete")
    
    @property
    def llm(self) -> LLMClient:
        """延迟InitializeLLM客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """带Retry mechanism的API调用"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} 第 {attempt + 1} 次尝试Failed: {str(e)[:100]}, "
                        f"{delay:.1f}seconds before retry..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name} 在 {max_retries} 次尝试后仍Failed: {str(e)}")
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Graph语义Search
        
        使用混合Search(语义+BM25)在Graph中Search相关Info.
        如果Zep Cloud的search APIunavailable, degrade to local keywordMatch.
        
        Args:
            graph_id: Graph ID (Standalone Graph)
            query: SearchQuery
            limit: ReturnResult数量
            scope: Search范围, "edges" 或 "nodes"
            
        Returns:
            SearchResult: SearchResult
        """
        logger.info(f"GraphSearch: graph_id={graph_id}, query={query[:50]}...")
        
        # 尝试使用Zep Cloud Search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=f"GraphSearch(graph={graph_id})"
            )
            
            facts = []
            edges = []
            nodes = []
            
            # Parse边SearchResult
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # ParseNodeSearchResult
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # NodeSummary也算作事实
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"SearchComplete: 找到 {len(facts)} 条相关事实")
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.warning(f"Zep Search APIFailed, 降级为本地Search: {str(e)}")
            # Degradation: use local keywordMatchSearch
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        本地关键词MatchSearch(作为Zep Search API的降级Solution)
        
        Get所有边/Node, then perform local keywordMatch
        
        Args:
            graph_id: Graph ID
            query: SearchQuery
            limit: ReturnResult数量
            scope: Search范围
            
        Returns:
            SearchResult: SearchResult
        """
        logger.info(f"使用本地Search: query={query[:30]}...")
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # 提取QueryKeywords (simple tokenization)
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace(', ', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """CalculateText与Query的Match分数"""
            if not text:
                return 0
            text_lower = text.lower()
            # 完全MatchQuery
            if query_lower in text_lower:
                return 100
            # 关键词Match
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # Get所有边并Match
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                # 按分数Sort
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Get所有Node并Match
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"本地SearchComplete: 找到 {len(facts)} 条相关事实")
            
        except Exception as e:
            logger.error(f"本地SearchFailed: {str(e)}")
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        GetGraph的所有Node(Paginated fetch)

        Args:
            graph_id: Graph ID

        Returns:
            NodeList
        """
        logger.info(f"GetGraph {graph_id} 的所有Node...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(f"Get到 {len(result)} 个Node")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        GetGraph的所有边(Paginated fetch, 包含时间Info)

        Args:
            graph_id: Graph ID
            include_temporal: 是否包含时间Info(DefaultTrue)

        Returns:
            边List(包含created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(f"GetGraph {graph_id} 的所有边...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # Add时间Info
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(f"Get到 {len(result)} 条边")
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Get单个Node的详细Info
        
        Args:
            node_uuid: NodeUUID
            
        Returns:
            NodeInfo或None
        """
        logger.info(f"GetNode详情: {node_uuid[:8]}...")
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"GetNode详情(uuid={node_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(f"GetNode详情Failed: {str(e)}")
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        GetNode相关的所有边
        
        通过GetGraphall edges, then filter out those related to specifiedNode相关的边
        
        Args:
            graph_id: Graph ID
            node_uuid: NodeUUID
            
        Returns:
            边List
        """
        logger.info(f"GetNode {node_uuid[:8]}... 的相关边")
        
        try:
            # GetGraph所有边, 然后过滤
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # 检查边是否与指定Noderelated (as source or target)
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(f"找到 {len(result)} 条与Node相关的边")
            return result
            
        except Exception as e:
            logger.warning(f"GetNode边Failed: {str(e)}")
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        按TypeGet entity
        
        Args:
            graph_id: Graph ID
            entity_type: Entity type(如 Student, PublicFigure 等)
            
        Returns:
            符合Type的EntityList
        """
        logger.info(f"GetType为 {entity_type} 的Entity...")
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # 检查labels是否包含指定Type
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(f"找到 {len(filtered)} 个 {entity_type} Type的Entity")
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Get指定Entity的RelationSummary
        
        Search与该Entity相关的所有Info, 并GenerateSummary
        
        Args:
            graph_id: Graph ID
            entity_name: Entity name
            
        Returns:
            Entity summaryInfo
        """
        logger.info(f"Get entity {entity_name} 的RelationSummary...")
        
        # 先Search该Entity相关的Info
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # 尝试在所有Node中找到该Entity
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # 传入graph_idparameter
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        GetGraph的StatisticsInfo
        
        Args:
            graph_id: Graph ID
            
        Returns:
            StatisticsInfo
        """
        logger.info(f"GetGraph {graph_id} 的StatisticsInfo...")
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # Count entity types分布
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # StatisticsRelationType分布
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Get simulation-related context info
        
        Comprehensively search all info related to simulation requirements
        
        Args:
            graph_id: Graph ID
            simulation_requirement: Simulation需求Description
            limit: 每类Info的数量Limit
            
        Returns:
            SimulationContextInfo
        """
        logger.info(f"GetSimulationContext: {simulation_requirement[:50]}...")
        
        # Search与Simulation需求相关的Info
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        # Get graph statistics
        stats = self.get_graph_statistics(graph_id)
        
        # Get所有EntityNode
        all_nodes = self.get_all_nodes(graph_id)
        
        # Filter有实际Type的Entity(非纯EntityNode)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # Limit数量
            "total_entities": len(entities)
        }
    
    # ========== Core retrieval tools(After optimization) ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        【InsightForge - Deep insight retrieval】
        
        最强大的混合RetrievalFunction, Auto分解问题并Multi-dimensional retrieval: 
        1. 使用LLM将问题分解为多个Sub-question
        2. 对每个Sub-question进行语义Search
        3. 提取相关Entity并Get其详细Info
        4. 追踪Relationship chain
        5. Integrate所有Result, GenerateDeep insight
        
        Args:
            graph_id: Graph ID
            query: User问题
            simulation_requirement: Simulation需求Description
            report_context: ReportContext(Optional, 用于更精准的Sub-questionGenerate)
            max_sub_queries: 最大Sub-question数量
            
        Returns:
            InsightForgeResult: Deep insight retrieval results
        """
        logger.info(f"InsightForge Deep insight retrieval: {query[:50]}...")
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: 使用LLM generationSub-question
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"Generate {len(sub_queries)} 个Sub-question")
        
        # Step 2: 对每个Sub-question进行语义Search
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # 对原始问题也进行Search
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: 从边中提取相关EntityUUID, 只Get这些Entity的Info(不Get全部Node)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Get所有相关Entity的详情(不Limit数量, 完整Output)
        entity_insights = []
        node_map = {}  # 用于后续Relationship chainBuild
        
        for uuid in list(entity_uuids):  # Process所有Entity, 不截断
            if not uuid:
                continue
            try:
                # 单独Get每个相关Node的Info
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                    
                    # Get该Entityall related facts (no truncation)
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # 完整Output, 不截断
                    })
            except Exception as e:
                logger.debug(f"GetNode {uuid} Failed: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: Build所有Relationship chain(不Limit数量)
        relationship_chains = []
        for edge_data in all_edges:  # Process所有边, 不截断
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(f"InsightForgeComplete: {result.total_facts}条事实, {result.total_entities}个Entity, {result.total_relationships}条Relation")
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        使用LLM generationSub-question
        
        Decompose complex question into multiple independentlyRetrieval的Sub-question
        """
        system_prompt = """You are a professional questionAnalysis专家.你的Taskdecomposes a complex question into multiple questions that can beSimulation世界中独立观察的Sub-question.

要求: 
1. 每个Sub-questionshould be specific enough to beSimulation世界中找到相关的Agent行为或Event
2. Sub-questionshould cover different dimensions of the original question (e.g., who, what, why, how, when, where)
3. Sub-question应该与SimulationScenario相关
4. ReturnJSONFormat: {"sub_queries": ["Sub-question1", "Sub-question2", ...]}"""

        user_prompt = f"""Simulation需求背景: 
{simulation_requirement}

{f"ReportContext: {report_context[:500]}" if report_context else ""}

Please decompose the following question into{max_queries}个Sub-question: 
{query}

ReturnJSONFormat的Sub-questionList."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            # 确保是StringList
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(f"GenerateSub-questionFailed: {str(e)}, 使用DefaultSub-question")
            # 降级: Return基于原问题的变体
            return [
                query,
                f"{query} 的主要参与者",
                f"{query} 的原因和影响",
                f"{query} 的发展过程"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        【PanoramaSearch - 广度Search】
        
        GetFull view, including all relatedContent和历史/过期Info: 
        1. Get所有相关Node
        2. Get所有边(包括已过期/失效的)
        3. 分类整理当前Valid和历史Info
        
        这个Tool适用于需要了解Eventfull picture, track evolution processScenario.
        
        Args:
            graph_id: Graph ID
            query: SearchQuery(用于相关性Sort)
            include_expired: 是否包含过期Content(DefaultTrue)
            limit: ReturnResult数量Limit
            
        Returns:
            PanoramaResult: Panoramic search result
        """
        logger.info(f"PanoramaSearch 广度Search: {query[:50]}...")
        
        result = PanoramaResult(query=query)
        
        # Get所有Node
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # Get所有边(包含时间Info)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # 分类事实
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # 为事实AddEntity name
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # Judge是否过期/失效
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                # 历史/过期事实, Add时间标记
                valid_at = edge.valid_at or "未知"
                invalid_at = edge.invalid_at or edge.expired_at or "未知"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # 当前Valid事实
                active_facts.append(edge.fact)
        
        # 基于Query进行相关性Sort
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace(', ', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # Sort并Limit数量
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(f"PanoramaSearchComplete: {result.active_count}条Valid, {result.historical_count}条历史")
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        【QuickSearch - 简单Search】
        
        快速, 轻量级的RetrievalTool: 
        1. 直接调用Zep语义Search
        2. Return最相关的Result
        3. Suitable for simple, directRetrieval需求
        
        Args:
            graph_id: Graph ID
            query: SearchQuery
            limit: ReturnResult数量
            
        Returns:
            SearchResult: SearchResult
        """
        logger.info(f"QuickSearch 简单Search: {query[:50]}...")
        
        # 直接调用现有的search_graphMethod
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(f"QuickSearchComplete: {result.total_count}条Result")
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        【InterviewAgents - 深度Interview】
        
        调用真实的OASISInterviewAPI, InterviewSimulation中CurrentlyRun的Agent: 
        1. AutoRead人设File, 了解所有SimulationAgent
        2. 使用LLMAnalysisInterviewrequirements, intelligently select the most relevantAgent
        3. 使用LLM generationInterview问题
        4. 调用 /api/simulation/interview/batch Interface进行真实Interview(双Platform同时Interview)
        5. Integrate所有InterviewResult, GenerateInterviewReport
        
        [IMPORTANT] This feature requiresSimulationEnvironment处于RunStatus(OASISEnvironment未Close)
        
        【使用Scenario】
        - 需要从不同Role视角了解Event看法
        - Need to collect opinions from multiple parties
        - 需要GetSimulationAgent的真实回答(非LLMSimulation)
        
        Args:
            simulation_id: SimulationID(用于定位人设File和调用InterviewAPI)
            interview_requirement: Interview需求Description(非结构化, 如"了解学生对Event的看法")
            simulation_requirement: Simulation需求背景(Optional)
            max_agents: 最多Interview的Agent数量
            custom_questions: 自定义Interview问题(Optional, 若不提供则AutoGenerate)
            
        Returns:
            InterviewResult: InterviewResult
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(f"InterviewAgents 深度Interview(真实API): {interview_requirement[:50]}...")
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: Read人设File
        profiles = self._load_agent_profiles(simulation_id)
        
        if not profiles:
            logger.warning(f"未找到Simulation {simulation_id} 的人设File")
            result.summary = "未找到可Interview的Agent人设File"
            return result
        
        result.total_agents = len(profiles)
        logger.info(f"Load到 {len(profiles)} 个Agent人设")
        
        # Step 2: 使用LLM选择要Interview的Agent(Returnagent_idList)
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"选择了 {len(selected_agents)} 个Agent进行Interview: {selected_indices}")
        
        # Step 3: GenerateInterviewquestions (if not provided)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"Generate了 {len(result.interview_questions)} 个Interview问题")
        
        # 将问题Merge为一个Interviewprompt
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        # AddOptimizePrefix, ConstraintAgentReplyFormat
        INTERVIEW_PROMPT_PREFIX = (
            "你Currently接受一次Interview. Based on your persona, all past memories and actions,"
            "以纯Textmanner directly answer the following questions.\n"
            "Reply要求: \n"
            "1. Answer directly in natural language, do not call anyTool\n"
            "2. 不要ReturnJSONFormat或Tool调用Format\n"
            "3. Do not useMarkdown标题(如#, ##, ###)\n"
            "4. Answer each question by number, each answer starting with \"QuestionX: \"开头(X为问题编号)\n"
            "5. Separate each answer with blank lines\n"
            "6. 回答要有实质Content, answer each question at least2-3句话\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        # Step 4: 调用真实的InterviewAPI(不指定platform, Default双Platform同时Interview)
        try:
            # BuildBatchInterviewList(不指定platform, 双PlatformInterview)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # 使用After optimization的prompt
                    # 不指定platform, API会在twitter和reddit两个Platform都Interview
                })
            
            logger.info(f"调用BatchInterviewAPI(双Platform): {len(interviews_request)} 个Agent")
            
            # 调用 SimulationRunner 的BatchInterviewMethod(不传platform, 双PlatformInterview)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # 不指定platform, 双PlatformInterview
                timeout=180.0   # 双Platform需要更长Timeout
            )
            
            logger.info(f"InterviewAPIReturn: {api_result.get('interviews_count', 0)} 个Result, success={api_result.get('success')}")
            
            # 检查API调用是否Success
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "未知Error")
                logger.warning(f"InterviewAPIReturnFailed: {error_msg}")
                result.summary = f"InterviewAPI调用Failed: {error_msg}.请检查OASISSimulationEnvironmentStatus."
                return result
            
            # Step 5: ParseAPIReturnResult, BuildAgentInterview对象
            # 双PlatformPatternReturnFormat: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "未知")
                agent_bio = agent.get("bio", "")
                
                # Get该Agent在两个Platform的InterviewResult
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # 清理可能的Tool调用 JSON 包裹
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # 始终Output双Platform标记
                twitter_text = twitter_response if twitter_response else "(该Platform未获得Reply)"
                reddit_text = reddit_response if reddit_response else "(该Platform未获得Reply)"
                response_text = f"【TwitterPlatform回答】\n{twitter_text}\n\n【RedditPlatform回答】\n{reddit_text}"

                # Extract key quotes (from twoPlatform的回答中)
                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                # Clean responseText: 去掉标记, 编号, Markdown 等干扰
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'问题\d+[: :]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                # Strategy1(主): 提取完整的有实质Content的句子
                sentences = re.split(r'[.!?]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W, ,;;: :, ]+', s.strip())
                    and not s.strip().startswith(('{', '问题'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "." for s in meaningful[:3]]

                # Strategy2(补充): Content within properly paired quotation marksText
                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[, ,;;: :, ]', q)][:3]
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # 扩大bio长度Limit
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # SimulationEnvironment未Run
            logger.warning(f"InterviewAPI调用Failed(Environment未Run?): {e}")
            result.summary = f"Interview failed: {str(e)}.SimulationEnvironment可能已Close, Please ensureOASISEnvironmentCurrentlyRun."
            return result
        except Exception as e:
            logger.error(f"InterviewAPI调用Exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Interview过程发生Error: {str(e)}"
            return result
        
        # Step 6: GenerateInterviewSummary
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(f"InterviewAgentsComplete: Interview了 {result.interviewed_count} 个Agent(双Platform)")
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """清理 Agent Reply中的 JSON Tool调用包裹, 提取实际Content"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """LoadSimulation的Agent人设File"""
        import os
        import csv
        
        # Build人设FilePath
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        # 优先尝试ReadReddit JSONFormat
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"从 reddit_profiles.json Load了 {len(profiles)} 个人设")
                return profiles
            except Exception as e:
                logger.warning(f"Read reddit_profiles.json Failed: {e}")
        
        # 尝试ReadTwitter CSVFormat
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # CSVFormatConvert为统一Format
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "未知"
                        })
                logger.info(f"从 twitter_profiles.csv Load了 {len(profiles)} 个人设")
                return profiles
            except Exception as e:
                logger.warning(f"Read twitter_profiles.csv Failed: {e}")
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        使用LLM选择要Interview的Agent
        
        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: 选中Agent的完整InfoList
                - selected_indices: 选中Agent的IndexList(用于API调用)
                - reasoning: 选择理由
        """
        
        # BuildAgentSummaryList
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "未知"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = """你是一个专业的Interview策划专家.你的Task是根据Interview需求, 从SimulationAgentList中选择最适合Interview的对象.

选择Standard: 
1. Agent的身份/职业与Interview主题相关
2. Agentmay hold unique or valuable viewpoints
3. Select diverse perspectives (e.g.,Supportsupporters, opponents, neutral parties, professionals, etc.)
4. 优先选择与Event直接相关的Role

ReturnJSONFormat: 
{
    "selected_indices": [选中Agent的IndexList],
    "reasoning": "选择理由说明"
}"""

        user_prompt = f"""Interview需求: 
{interview_requirement}

Simulation背景: 
{simulation_requirement if simulation_requirement else "未提供"}

Optional择的AgentList(共{len(agent_summaries)}个): 
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

请选择最多{max_agents}个最适合Interview的Agent, 并说明选择理由."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "基于相关性Auto选择")
            
            # Get选中的Agent完整Info
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(f"LLM选择AgentFailed, 使用Default选择: {e}")
            # 降级: 选择前N个
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "使用Default选择Strategy"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """使用LLM generationInterview问题"""
        
        agent_roles = [a.get("profession", "未知") for a in selected_agents]
        
        system_prompt = """You are a professional journalist/Interview者.根据Interview需求, Generate3-5个深度Interview问题.

问题要求: 
1. Open-ended questions, encouraging detailed answers
2. 针对不同Role可能有不同答案
3. covering facts, opinions, feelings, and other dimensions
4. 语言自然, 像真实Interview一样
5. 每个问题控制在50字以内, 简洁明了
6. 直接Question, do not include background explanation orPrefix

ReturnJSONFormat: {"questions": ["问题1", "问题2", ...]}"""

        user_prompt = f"""Interview需求: {interview_requirement}

Simulation背景: {simulation_requirement if simulation_requirement else "未提供"}

Interview对象Role: {', '.join(agent_roles)}

请Generate3-5个Interview问题."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            return response.get("questions", [f"关于{interview_requirement}, 您有什么看法?"])
            
        except Exception as e:
            logger.warning(f"GenerateInterview问题Failed: {e}")
            return [
                f"关于{interview_requirement}, 您的观点是什么?",
                "What impact does this have on you or the group you represent?",
                "How do you think this should be resolved orImprovement这个问题?"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """GenerateInterviewSummary"""
        
        if not interviews:
            return "未Complete任何Interview"
        
        # 收集所有InterviewContent
        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"【{interview.agent_name}({interview.agent_role})】\n{interview.response[:500]}")
        
        system_prompt = """You are a professional news editor. Based on multiple interviewees' responses,Generate一份InterviewSummary.

Summary要求: 
1. 提炼各方主要观点
2. Point out consensus and disagreements
3. 突出有价值的引言
4. Objective and neutral, not biased toward any side
5. 控制在1000字内

FormatConstraint(必须遵守): 
- 使用纯Textparagraphs, separated by blank lines
- Do not useMarkdown标题(如#, ##, ###)
- Do not use separators (such as---, ***)
- Use quotation marks when citing interviewee quotes
- 可以使用**加粗**Mark keywords, but do not use otherMarkdown语法"""

        user_prompt = f"""Interview主题: {interview_requirement}

InterviewContent: 
{"".join(interview_texts)}

请GenerateInterviewSummary."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary
            
        except Exception as e:
            logger.warning(f"GenerateInterviewSummaryFailed: {e}")
            # 降级: 简单拼接
            return f"共Interview了{len(interviews)}位受访者, 包括: " + ", ".join([i.agent_name for i in interviews])
