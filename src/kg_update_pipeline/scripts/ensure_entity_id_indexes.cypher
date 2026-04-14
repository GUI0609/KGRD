// Run once in Neo4j Browser or cypher-shell against your database.
// Speeds up MERGE (n:Label {entity_id: ...}) used by kg_update_pipeline.
//
// Neo4j 4.4+ / 5.x: CREATE INDEX ... IF NOT EXISTS FOR ...

CREATE INDEX entity_id_disease IF NOT EXISTS FOR (n:disease) ON (n.entity_id);
CREATE INDEX entity_id_gene_protein IF NOT EXISTS FOR (n:`gene/protein`) ON (n.entity_id);
CREATE INDEX entity_id_effect_phenotype IF NOT EXISTS FOR (n:`effect/phenotype`) ON (n.entity_id);
CREATE INDEX entity_id_external_resource IF NOT EXISTS FOR (n:ExternalResource) ON (n.entity_id);
CREATE INDEX entity_id_unknown_entity IF NOT EXISTS FOR (n:UnknownEntity) ON (n.entity_id);
CREATE INDEX entity_id_named_thing IF NOT EXISTS FOR (n:NamedThing) ON (n.entity_id);
CREATE INDEX entity_id_biological_process IF NOT EXISTS FOR (n:biological_process) ON (n.entity_id);
CREATE INDEX entity_id_pathway IF NOT EXISTS FOR (n:pathway) ON (n.entity_id);
CREATE INDEX entity_id_chemical_entity IF NOT EXISTS FOR (n:ChemicalEntity) ON (n.entity_id);
// 副标签 :entity_id（与 gene/protein 等共存）
CREATE INDEX entity_id_on_entity_id_label IF NOT EXISTS FOR (n:entity_id) ON (n.entity_id);

// After creation, wait until indexes are ONLINE (SHOW INDEXES).
