/* Line segemnts create pipeline nodes only connected by two edges. We need to find and mark those*/
MATCH p=(a:PipelineNode)-[r]-(b:PipelineNode)
WHERE size((a)--()) = 2
AND size((b)--()) = 2
SET r.on_line_segment = True

/* Create an edge skipping over the line segment nodes */
MATCH p=(a:PipelineNode)-[:PIPELINE_CONNECTION]-(:PipelineNode)-[r:PIPELINE_CONNECTION* {on_line_segment:True}]-(:PipelineNode)-[:PIPELINE_CONNECTION]-(b:PipelineNode)
WHERE size((a)--()) > 2
AND size((b)--()) > 2
CREATE (a)-[:SIMPLIFIED_PIPELINE {impedance: REDUCE (totalImp = 0, n IN r| totalImp + n.impedance)}]->(b)

/* Delete line segment connections */
MATCH ()-[r:PIPELINE_CONNECTION]-()
DELETE r

/* Find railway line segments */
MATCH p=(a:RailwayNode)-[r]-(b:RailwayNode)
WHERE size((a)--()) = 2
AND size((b)--()) = 2
SET r.on_line_segment = True

/* Create edges skipping over railway line segments */
MATCH p=(a:RailwayNode)-[:RAILWAY_CONNECTION]-(:RailwayNode)-[r:RAILWAY_CONNECTION* {on_line_segment:True}]-(:RailwayNode)-[:RAILWAY_CONNECTION]-(b:RailwayNode)
WHERE size((a)--()) > 2
AND size((b)--()) > 2
CREATE (a)-[:SIMPLIFIED_RAILWAY {impedance: REDUCE (totalImp = 0, n IN r| totalImp + n.impedance)}]->(b)

/* Delete railway line segment connections */
MATCH ()-[r:RAILWAY_CONNECTION]-()
DELETE r

/* Delete all nodes with no connections */
MATCH (a) WHERE size((a)--()) = 0 DELETE a