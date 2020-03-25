neo4j-admin import \
--nodes=ffsc_kedro_pipeline/results/output/shipping_node_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/shipping_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/port_node_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/port_ship_edge_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/port_pipeline_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/pipeline_node_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/pipeline_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/refinery_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/refinery_pipeline_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/oil_field_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/oil_field_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/well_pad_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/well_pad_pipeline_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/lng_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/lng_pipeline_edge_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/lng_shipping_route_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/processing_plant_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/processing_plant_pipeline_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/power_station_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/power_station_pipeline_edge_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/power_station_railway_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/railway_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/railway_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/coal_mines_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/coal_mine_railway_edge_dataframe.csv \
\
--nodes=ffsc_kedro_pipeline/results/output/cities_nodes_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/cities_pipelines_edge_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/cities_railways_edge_dataframe.csv \
--relationships=ffsc_kedro_pipeline/results/output/cities_ports_edge_dataframe.csv