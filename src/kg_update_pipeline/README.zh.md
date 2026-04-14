# KG Update Pipeline（知识图谱增量更新管线）

面向罕见病与本体风格知识图谱：**版本化下载 → 统一解析模型 → Neo4j 幂等 MERGE**。每次运行保留原始文件，可选导出 JSONL/CSV 摘要；**不通过本管线做全库清空**，核心路径仅使用 `MERGE`。

**英文说明：** [README.md](README.md)

---

## 目录

- [概述](#概述)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [Neo4j 对接](#neo4j-对接)
- [命令行参数](#命令行参数)
- [运行产物](#运行产物)
- [增量模式行为](#增量模式行为)
- [性能与调优](#性能与调优)
- [可选 Docker 备份](#可选-docker-备份)
- [测试](#测试)
- [代码结构](#代码结构)
- [安全建议](#安全建议)

---

## 概述

| 方面 | 行为 |
|------|------|
| **写入方式** | 节点按 `entity_id`（等于解析器 `primary_id`）与关系做 `MERGE`；核心实现不包含 `DELETE` / `DETACH DELETE`。 |
| **关系类型** | `neo4j.restrict_to_existing_relationship_types: true`（默认）时，仅合并 `CALL db.relationshipTypes()` 中**已存在**的类型；解析器给出的未知类型会被跳过。 |
| **节点标签** | 解析器 `NodeRecord.category` 经默认表与 `neo4j.category_label_map` 映射为 Neo4j 标签；可选 `restrict_to_existing_node_labels` 仅向 `db.labels()` 中已有标签写入。 |
| **刷新 vs 仅新建** | 默认 `refresh_existing_nodes` / `refresh_existing_relationships` 为 **false**，属性主要在 **`ON CREATE`** 写入（增量更快）；设为 **true** 则每次运行刷新已有实体名称、同义词、证据等属性。 |
| **附加标签** | `neo4j.additional_node_labels`（如 `entity_id`）在 MERGE 后为节点追加副标签，以匹配 `(:disease:entity_id)` 这类模型。 |
| **Neo4j 流程** | 每轮将库导出为 JSONL，在本地比对后仅把**增量**批量 `MERGE` 回库。 |
| **下载** | `update.skip_download: true` 时默认不发起 HTTP 下载，使用 `versions/.../raw/` 下已有文件；某次要拉取线上数据时用 CLI `--force-download`。 |

**解析器**输出 `NodeRecord` / `EdgeRecord`（见 `schema/models.py`）。已支持来源包括 HPO、MONDO、HGNC、Monarch 节点 TSV，以及 OMIM、NCBI、BioMart、Orphanet 等占位实现。**Monarch 边**暂未纳入解析，需自行扩展解析器后才会批量写入关系。

一次性超大批量导入可优先考虑 `neo4j-admin import` 或 `LOAD CSV`；本工程侧重**可重复、带审计与版本目录的增量 MERGE**。

---

## 环境要求

- Python 3.10+（推荐）
- Neo4j 4.4+ 或 5.x，且可经 Bolt 访问
- 依赖：仓库根目录的 `requirements_kg_update.txt`

```bash
pip install -r requirements_kg_update.txt
```

---

## 安装

在 **`kg_update_pipeline/` 的父目录** 下安装依赖，以便 `import kg_update_pipeline` 能正确解析：

```bash
cd /path/to/rd-project    # 或你的检出根目录
pip install -r requirements_kg_update.txt
```

---

## 快速开始

1. 复制 `templates/my_kg_update.yaml` 为本地配置（例如 `my_kg_update.yaml`）。
2. 填写 `neo4j.uri`、`neo4j.user`、`neo4j.password` 及各类路径（`data_root` 等）。**勿将密钥提交进 Git。**
3. 为实际会 MERGE 的每个标签，在 `entity_id` 上建好**索引**（见下文 [Neo4j 对接](#neo4j-对接)）。
4. 在仓库根目录执行：

```bash
python kg_update_pipeline/scripts/run_kg_update.py --config kg_update_pipeline/my_kg_update.yaml
```

---

## 配置说明

| 配置块 | 作用 |
|--------|------|
| `project_root`、`data_root`、`log_root`、`backup_root`、`state_root` | 路径；相对路径相对于**配置文件所在目录**解析。 |
| `neo4j.*` | 连接信息、类别到标签映射、批大小、读写限制、可选 `additional_node_labels`。 |
| `update.*` | `incremental` / `full`、`dry_run`、`skip_download`、备份、HTTP 超时、解析导出上限、Docker dump 开关等。 |
| `sources.<name>` | 每数据源：`enabled`、`type`（如 `http`、`http_tar_gz`）、`url`、`filename`、`fallback_urls`、可选 `archive_member`。 |

**标签映射**

- `use_legacy_entity_labels: true`：内置映射（如 `Disease` → `disease`）与 `category_label_map` **合并**，YAML 中键与内置冲突时以 YAML 为准。
- `use_legacy_entity_labels: false`：**仅**使用 `category_label_map`，须列举本次运行会产出的全部解析器 `category`，且目标标签须与 `CALL db.labels()` 一致。

**校验**

- `validate_category_label_targets: true`：连接后校验映射目标标签及 `additional_node_labels` 中的每一项都存在于 `db.labels()`。

示例配置见 `templates/my_kg_update.yaml`。可将当前库的 `CALL db.labels()` / `CALL db.relationshipTypes()` 快照保存在 `state/` 下作为参照（例如 `neo4j_schema_snapshot.yaml`）。

---

## Neo4j 对接

### 本地快照与增量 MERGE

每次运行会在版本目录下生成 `neo4j_snapshot/nodes.jsonl` 与 `edges.jsonl`，在本地用节点指纹与边三元组集合做差集，仅对**新建/变更节点**与**尚不存在的边**执行批量 MERGE。审计写入 `update_audit.jsonl`，摘要见 `local_diff_summary.json`。每轮仍会对库做一次导出扫描。若 `refresh_existing_relationships: true`，关系阶段会对**全部解析出的边**执行 MERGE（以便刷新已有关系属性），节点阶段仍为增量。

### 索引

MERGE 很慢时，多数是缺少 `(n:Label {entity_id: …})` 上的索引。每个库执行一次：

- [`scripts/ensure_entity_id_indexes.cypher`](scripts/ensure_entity_id_indexes.cypher)：常见主标签，以及使用 `additional_node_labels: [entity_id]` 时需要的 **`(:entity_id)`** 上的索引。

创建后请等待索引状态为 **ONLINE**（`SHOW INDEXES`）。

### 可选副标签

若数据模型为 `(:gene/protein:entity_id)` 等，可在配置中写：

```yaml
neo4j:
  additional_node_labels:
    - entity_id
```

并确保库内存在 `entity_id` 标签，且索引脚本已覆盖该组合。

### 调优

- **`merge_batch_size`**（默认 `2000`，允许 `50`–`20000`）：批越大往返越少，单次事务内存峰值越高。
- 大属性 payload 时，可配合调整 `neo4j.conf` 中的堆与 page cache。

---

## 命令行参数

| 参数 | 说明 |
|------|------|
| `-c` / `--config` PATH | YAML 配置路径（必填）。拼写容错：`--congfig`。 |
| `--dry-run` | 不写 Neo4j，仅记录拟执行操作。 |
| `--mode full` | 按「全量」风格运行（见 [增量模式行为](#增量模式行为)）。 |
| `--skip-download` | 本次运行跳过 HTTP 下载（与 `update.skip_download: true` 效果相同）。 |
| `--force-download` | 即使 YAML 里 `update.skip_download: true` 也强制下载。 |
| `--skip-neo4j` | 仅解析与版本化，不对 Bolt 做 MERGE。 |
| `--source NAME` | 可重复；仅这些源视为启用并参与流程。 |
| `-v` / `--verbose` | 详细日志。 |

入口脚本：[`scripts/run_kg_update.py`](scripts/run_kg_update.py)。

---

## 运行产物

目录位于 `data_root/versions/<YYYY-MM-DD>/`（逻辑见 `version_manager.py`）。

| 产物 | 说明 |
|------|------|
| **`manifest.json`** | 运行 ID、时间戳、下载哈希、解析与 Neo4j 统计（如 `nodes_created`、`nodes_updated`、`relationships_*` 等）。 |
| **`neo4j_merge_delta.json`** | 本次 MERGE 摘要；若有新建节点/关系，包含对应 JSONL 路径。 |
| **`neo4j_created_nodes.jsonl`** | 每个新建节点一行：`label`、`entity_id`。 |
| **`neo4j_created_relationships.jsonl`** | 每个新建关系一行：`relationship_type`、`source_id`、`target_id`。 |
| **`raw/`、`parsed/`** | 版本化原始数据；可选 `{source}_nodes.jsonl`、`_edges.jsonl`、`_summary.csv`。 |

---

## 增量模式行为

当 `update.mode: incremental` 且 `update.skip_parse_if_unchanged: true` 时，若某数据源下载文件的 **SHA-256 与上次运行相同**，则会**跳过解析**，该源在当次运行中**不会产生新的 MERGE 工作**。

若需强制重新解析并 MERGE：使用 **`--mode full`**，或在 YAML 中将 **`skip_parse_if_unchanged`** 设为 **`false`**。

---

## 性能与调优

1. 每个主标签（及副标签场景下）为 **`entity_id` 建索引**——大图谱几乎必需。  
2. 调整 **`merge_batch_size`**，在延迟与事务体量之间折中。  
3. 默认 **`refresh_existing_*: false`**，避免每次运行大规模回写已有属性。  
4. 极限体量离线导入可单独使用 `neo4j-admin import`；本管线优化目标是**持续性 MERGE 更新**。

---

## 可选 Docker 备份

在下载/MERGE 前做一次离线 `neo4j-admin dump`：

- 设置 `update.pre_update_docker_dump: true` 且 `update.backup_before_update: true`。
- 配置 `update.docker_container` 或环境变量 **`KG_NEO4J_DOCKER_CONTAINER`**。

说明见 [`neo4j_db/backup.py`](neo4j_db/backup.py) 文件头注释，辅助脚本：[`scripts/kg_neo4j_dump.py`](scripts/kg_neo4j_dump.py)。

---

## 测试

冒烟 / 集成入口（最小 OBO 夹具，可选 Neo4j dry-run 相关参数）：

```bash
python kg_update_pipeline/e2e_full_verify.py --help
```

---

## 代码结构

| 路径 | 作用 |
|------|------|
| `kg_update_pipeline/` | Python 包根目录。 |
| `kg_update_pipeline/neo4j_db/` | Bolt 客户端、库内 schema 探测、MERGE 写入（目录名避免与 PyPI 包 `neo4j` 冲突）。 |
| `kg_update_pipeline/parsers/` | 分数据源解析器；`base_parser.py` 含 OBO/TSV 等共用逻辑。 |
| `kg_update_pipeline/scripts/` | CLI 与维护用 Cypher。 |
| `kg_update_pipeline/templates/` | 示例 YAML。 |

YAML 中的相对路径相对于**配置文件所在目录**解析（绝对路径则直接使用）。

---

## 安全建议

- Neo4j 账号口令使用环境专用配置或密钥管理，不要写入公开仓库。  
- `update.allow_destructive_ops` 用于约束将来的清空类 Cypher；默认管线逻辑仅执行 **MERGE**。
