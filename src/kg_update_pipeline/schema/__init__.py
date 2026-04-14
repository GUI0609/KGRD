"""统一 schema：节点与边的数据模型及归一化工具。"""

from kg_update_pipeline.schema.models import EdgeRecord, NodeRecord
from kg_update_pipeline.schema.normalizer import (
    merge_edge_props,
    merge_node_records,
    merge_nodes_by_id,
)

__all__ = [
    "NodeRecord",
    "EdgeRecord",
    "merge_node_records",
    "merge_nodes_by_id",
    "merge_edge_props",
]
