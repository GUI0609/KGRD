from flask import Flask, request, jsonify
from neo4j import GraphDatabase
import json
with open('PATH/TO/config.json', 'r') as f:
    config = json.load(f)

app = Flask(__name__)


driver = GraphDatabase.driver(config['NEO4J']['NEO4J_URI'], auth=(config['NEO4J']['NEO4J_USERNAME'], config['NEO4J']['NEO4J_PASSWORD']))

def run_query(query, params):
    with driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]

@app.route("/query_gene_to_disease", methods=["POST"])
def query_gene_to_disease():
    gene_ids = request.json.get("gene_ids", [])
    query = """
        MATCH (g:`gene/protein`)
        WHERE g.entity_id IN $gene_ids
        WITH collect(g) AS genes
        UNWIND genes AS g1
        UNWIND genes AS g2
        WITH g1, g2 WHERE id(g1) < id(g2)
        MATCH path = allShortestPaths((g1)-[*..10]-(g2))
        WITH collect(path) AS paths
        WITH apoc.coll.flatten([p IN paths | nodes(p)]) AS allNodes
        UNWIND allNodes AS node
        WITH node
        WHERE node:disease
        RETURN node.entity_id AS disease, COUNT(*) AS freq
        ORDER BY freq DESC
        LIMIT 100
    """
    return jsonify(run_query(query, {"gene_ids": gene_ids}))

@app.route("/query_phenotype_to_disease", methods=["POST"])
def query_phenotype_to_disease():
    phenotype_ids = request.json.get("phenotype_ids", [])
    query = """
        MATCH (p:`effect/phenotype`)
        WHERE p.entity_id IN $phenotype_ids
        WITH collect(p) AS phenotypes
        UNWIND phenotypes AS p1
        UNWIND phenotypes AS p2
        WITH p1, p2 WHERE id(p1) < id(p2)
        MATCH path = allShortestPaths((p1)-[*..10]-(p2))
        WITH collect(path) AS paths
        WITH apoc.coll.flatten([p IN paths | nodes(p)]) AS allNodes
        UNWIND allNodes AS node
        WITH node
        WHERE node:disease
        RETURN node.entity_id AS disease, COUNT(*) AS freq
        ORDER BY freq DESC
        LIMIT 100
    """
    return jsonify(run_query(query, {"phenotype_ids": phenotype_ids}))

@app.route("/query_phenotype_to_gene", methods=["POST"])
def query_phenotype_to_gene():
    phenotype_ids = request.json.get("phenotype_ids", [])
    query = """
        MATCH (p:`effect/phenotype`)
        WHERE p.entity_id IN $phenotype_ids
        WITH collect(p) AS phenotypes
        UNWIND phenotypes AS p1
        UNWIND phenotypes AS p2
        WITH p1, p2 WHERE id(p1) < id(p2)
        MATCH path = allShortestPaths((p1)-[*..10]-(p2))
        WITH collect(path) AS paths
        WITH apoc.coll.flatten([p IN paths | nodes(p)]) AS allNodes
        UNWIND allNodes AS node
        WITH node
        WHERE node:`gene/protein`
        RETURN node.entity_id AS gene, COUNT(*) AS freq
        ORDER BY freq DESC
        LIMIT 100
    """
    return jsonify(run_query(query, {"phenotype_ids": phenotype_ids}))

@app.route("/query_min_subgraph", methods=["POST"])
def query_min_subgraph():
    node_ids = request.json.get("node_ids", [])
    query = """
        MATCH (n)
        WHERE n.entity_id IN $node_ids
        WITH collect(n) AS targets
        UNWIND targets AS n1
        UNWIND targets AS n2
        WITH n1, n2 WHERE id(n1) < id(n2)
        MATCH path = allShortestPaths((n1)-[*..3]-(n2))
        WITH collect(path) AS paths
        RETURN apoc.coll.toSet(apoc.coll.flatten([p IN paths | nodes(p)])) AS nodes,
               apoc.coll.toSet(apoc.coll.flatten([p IN paths | relationships(p)])) AS rels
    """
    return jsonify(run_query(query, {"node_ids": node_ids}))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8194, debug=True)
