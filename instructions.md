To run memgraph Docker container:

docker run -d -p 7687:7687 -p 3000:3000 -v C:/Users/paolo/Desktop/bb_graphrag/graphs:/usr/lib/memgraph/import memgraph/memgraph-platform      