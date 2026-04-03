"""
ZepEntityRead与过滤Service
从ZepGraph中ReadNode, Filter出符合预定义Entity type的Node
"""

import time
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar
from dataclasses import dataclass, field

from .graph_store import GraphStore

from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_entity_reader')

# 用于泛型ReturnType
T = TypeVar('T')


@dataclass
class EntityNode:
    """EntityNodeData结构"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # 相关的Edge info
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # 相关的其他NodeInfo
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }
    
    def get_entity_type(self) -> Optional[str]:
        """Get entityType(排除Default的Entity标签)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """过滤后的Entity集合"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class ZepEntityReader:
    """
    ZepEntityRead与过滤Service
    
    主要功能: 
    1. 从ZepGraphRead所有Node
    2. Filter出符合预定义Entity type的Node(Labels不只是Entity的Node)
    3. Get每个Entity的相关边和关联NodeInfo
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZEP_API_KEY or "local-sqlite-key"
        
        self.client = GraphStore(api_key=self.api_key)
    
    def _call_with_retry(
        self, 
        func: Callable[[], T], 
        operation_name: str,
        max_retries: int = 3,
        initial_delay: float = 2.0
    ) -> T:
        """
        带Retry mechanism的Zep API调用
        
        Args:
            func: 要Execute的Function(无parameter的lambda或callable)
            operation_name: 操作Name, 用于日志
            max_retries: Max retries(Default3次, 即最多尝试3次)
            initial_delay: 初始延迟秒数
            
        Returns:
            API调用Result
        """
        last_exception = None
        delay = initial_delay
        
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
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Zep {operation_name} 在 {max_retries} 次尝试后仍Failed: {str(e)}")
        
        raise last_exception
    
    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        GetGraph的所有Node(Paginated fetch)

        Args:
            graph_id: Graph ID

        Returns:
            NodeList
        """
        logger.info(f"GetGraph {graph_id} 的所有Node...")

        nodes = fetch_all_nodes(self.client, graph_id)

        nodes_data = []
        for node in nodes:
            nodes_data.append({
                "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                "name": node.name or "",
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
            })

        logger.info(f"共Get {len(nodes_data)} 个Node")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        GetGraph的所有边(Paginated fetch)

        Args:
            graph_id: Graph ID

        Returns:
            边List
        """
        logger.info(f"GetGraph {graph_id} 的所有边...")

        edges = fetch_all_edges(self.client, graph_id)

        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                "name": edge.name or "",
                "fact": edge.fact or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "attributes": edge.attributes or {},
            })

        logger.info(f"共Get {len(edges_data)} 条边")
        return edges_data
    
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """
        Get指定Node的所有相关边(带Retry mechanism)
        
        Args:
            node_uuid: NodeUUID
            
        Returns:
            边List
        """
        try:
            # 使用Retry mechanism调用Zep API
            edges = self._call_with_retry(
                func=lambda: self.client.graph.node.get_entity_edges(node_uuid=node_uuid),
                operation_name=f"GetNode边(node={node_uuid[:8]}...)"
            )
            
            edges_data = []
            for edge in edges:
                edges_data.append({
                    "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                    "name": edge.name or "",
                    "fact": edge.fact or "",
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "attributes": edge.attributes or {},
                })
            
            return edges_data
        except Exception as e:
            logger.warning(f"GetNode {node_uuid} 的边Failed: {str(e)}")
            return []
    
    def filter_defined_entities(
        self, 
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        Filter出符合预定义Entity type的Node
        
        Filter逻辑: 
        - 如果Node的Labels只有一个"Entity", 说明这个Entity不符合我们预定义的Type, Skip
        - 如果Node的Labels包含除"Entity"和"Node"之外的标签, 说明符合预定义Type, 保留
        
        Args:
            graph_id: Graph ID
            defined_entity_types: 预定义的Entity typeList(Optional, 如果提供则只保留这些Type)
            enrich_with_edges: 是否Get每个Entity的相关Edge info
            
        Returns:
            FilteredEntities: 过滤后的Entity集合
        """
        logger.info(f"StartFilterGraph {graph_id} 的Entity...")
        
        # Get所有Node
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)
        
        # Get所有边(用于后续关联Find)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        
        # BuildNodeUUID到NodeData的Mapping
        node_map = {n["uuid"]: n for n in all_nodes}
        
        # Filter符合Condition的Entity
        filtered_entities = []
        entity_types_found = set()
        
        for node in all_nodes:
            labels = node.get("labels", [])
            
            # Filter逻辑: Labels必须包含除"Entity"和"Node"之外的标签
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]
            
            if not custom_labels:
                # 只有Default标签, Skip
                continue
            
            # 如果指定了预定义Type, 检查是否Match
            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]
            
            entity_types_found.add(entity_type)
            
            # CreateEntityNode对象
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )
            
            # Get相关边和Node
            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()
                
                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])
                
                entity.related_edges = related_edges
                
                # Get关联Node的基本Info
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node["labels"],
                            "summary": related_node.get("summary", ""),
                        })
                
                entity.related_nodes = related_nodes
            
            filtered_entities.append(entity)
        
        logger.info(f"FilterComplete: 总Node {total_count}, 符合Condition {len(filtered_entities)}, "
                   f"Entity type: {entity_types_found}")
        
        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )
    
    def get_entity_with_context(
        self, 
        graph_id: str, 
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """
        Get单个Entity及其完整Context(边和关联Node, 带Retry mechanism)
        
        Args:
            graph_id: Graph ID
            entity_uuid: EntityUUID
            
        Returns:
            EntityNode或None
        """
        try:
            # 使用Retry mechanismGetNode
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=entity_uuid),
                operation_name=f"GetNode详情(uuid={entity_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            # GetNode的边
            edges = self.get_node_edges(entity_uuid)
            
            # Get所有Node用于关联Find
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}
            
            # Process相关边和Node
            related_edges = []
            related_node_uuids = set()
            
            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])
            
            # Get关联NodeInfo
            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node["labels"],
                        "summary": related_node.get("summary", ""),
                    })
            
            return EntityNode(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )
            
        except Exception as e:
            logger.error(f"Get entity {entity_uuid} Failed: {str(e)}")
            return None
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """
        Get all entities of specified type
        
        Args:
            graph_id: Graph ID
            entity_type: Entity type(如 "Student", "PublicFigure" 等)
            enrich_with_edges: Whether to get related edge info
            
        Returns:
            EntityList
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities


