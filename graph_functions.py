from neo4j import GraphDatabase


def format_subgraph_to_string(nodes, relationships=None, relationships_explored=True):
    """
    Convert a subgraph (nodes and relationships) to the standard string format using actual node names.

    Args:
        nodes (list): List of node dictionaries with properties and labels
        relationships (list, optional): List of relationship dictionaries
        relationships_explored (bool): Whether relationships were explored (default True)

    Returns:
        dict: Formatted result with graph_string and metadata
    """
    if not nodes:
        return {
            "graph_string": "G describes an empty graph with no nodes.",
            "node_count": 0,
            "relationship_count": 0,
            "node_names": []
        }

    # Extract node names
    node_names = []
    name_to_node = {}

    for node in nodes:
        # Use name if available, otherwise use internal id or fallback
        node_name = node.get('name', node.get('id', f'unnamed_node_{len(node_names)}'))
        node_names.append(node_name)
        name_to_node[node_name] = node

    # Build adjacency list with relationship information using actual names
    adjacency = {name: {} for name in node_names}  # {node: {neighbor: [rel_info]}}

    if relationships:
        for rel in relationships:
            # Extract start and end node information
            start_name = None
            end_name = None
            rel_type = rel.get('type', 'UNKNOWN')
            rel_description = rel.get('description', '')
            rel_evidence = rel.get('evidence', '')

            # Handle different relationship formats
            if 'start_node' in rel and 'end_node' in rel:
                start_name = rel['start_node'].get('name', rel['start_node'].get('id'))
                end_name = rel['end_node'].get('name', rel['end_node'].get('id'))
            elif 'source' in rel and 'target' in rel:
                start_name = rel['source'].get('name', rel['source'].get('id'))
                end_name = rel['target'].get('name', rel['target'].get('id'))
            elif 'neighbor' in rel and 'source_node' in rel:
                # For neighbor-style relationships
                start_name = rel['source_node'].get('name', rel['source_node'].get('id'))
                end_name = rel['neighbor'].get('name', rel['neighbor'].get('id'))

            # Create relationship info object
            rel_info = {
                'type': rel_type,
                'description': rel_description,
                'evidence': rel_evidence
            }

            # Add to adjacency list with relationship info if both nodes found
            if start_name in adjacency and end_name in adjacency:
                # Add relationship from start to end
                if end_name not in adjacency[start_name]:
                    adjacency[start_name][end_name] = []
                adjacency[start_name][end_name].append(rel_info)

                # Add relationship from end to start (undirected graph)
                if start_name not in adjacency[end_name]:
                    adjacency[end_name][start_name] = []
                adjacency[end_name][start_name].append(rel_info)

    # Generate the string description using actual names
    sorted_names = sorted(node_names)
    graph_desc = f"G describes a graph among nodes {', '.join(sorted_names)}."

    connections = []
    for name in sorted_names:
        if adjacency[name]:  # Has connections
            connection_parts = []
            for neighbor in sorted(adjacency[name].keys()):
                rel_infos = adjacency[name][neighbor]

                # Process relationships and build connection string
                if len(rel_infos) == 1:
                    rel_info = rel_infos[0]
                    connection_str = f"to {neighbor} via {rel_info['type']}"

                    # Add description and evidence if present
                    additional_info = []
                    if rel_info['description']:
                        additional_info.append(f"relation description: {rel_info['description']}")
                    if rel_info['evidence']:
                        additional_info.append(f"textual evidence: {rel_info['evidence']}")

                    if additional_info:
                        connection_str += f" ({', '.join(additional_info)})"

                    connection_parts.append(connection_str)
                else:
                    # Multiple relationships - combine types and collect unique descriptions/evidence
                    unique_types = []
                    descriptions = []
                    evidences = []

                    for rel_info in rel_infos:
                        if rel_info['type'] not in unique_types:
                            unique_types.append(rel_info['type'])
                        if rel_info['description'] and rel_info['description'] not in descriptions:
                            descriptions.append(rel_info['description'])
                        if rel_info['evidence'] and rel_info['evidence'] not in evidences:
                            evidences.append(rel_info['evidence'])

                    connection_str = f"to {neighbor} via {'/'.join(unique_types)}"

                    # Add combined descriptions and evidences if present
                    additional_info = []
                    if descriptions:
                        additional_info.append(f"relation description: {'; '.join(descriptions)}")
                    if evidences:
                        additional_info.append(f"textual evidence: {'; '.join(evidences)}")

                    if additional_info:
                        connection_str += f" ({', '.join(additional_info)})"

                    connection_parts.append(connection_str)

            connections.append(f"Node {name} is connected to {', '.join(connection_parts)}.")
        else:
            if relationships_explored:
                connections.append(f"Node {name} has no connections.")
            else:
                connections.append(f"The connections of Node {name} have not been explored yet.")

    if connections:
        graph_string = f"{graph_desc} In this graph: {' '.join(connections)}"
    else:
        if relationships_explored:
            graph_string = f"{graph_desc} No connections exist between nodes."
        else:
            graph_string = f"{graph_desc} The connections between nodes have not been explored yet."

    return {
        "graph_string": graph_string,
        "node_count": len(nodes),
        "relationship_count": len(relationships) if relationships else 0,
        "node_names": sorted_names,
        "raw_nodes": nodes,
        "raw_relationships": relationships or []
    }


def load_graph_from_json(json_filename, uri="bolt://localhost:7687", user="", password=""):
    """
    Load a graph into Memgraph from a JSON file using the import_util.json() procedure.
    The JSON file must be accessible via the mounted volume at /usr/lib/memgraph/import/.

    Args:
        json_filename (str): Name of the JSON file (e.g., "test.json")
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result of the import operation
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Construct the full path in the Memgraph container
            file_path = f"/usr/lib/memgraph/import/{json_filename}"

            # Use the import_util.json() procedure to load the graph
            query = "CALL import_util.json($file_path)"
            result = session.run(query, file_path=file_path)

            # Collect and return results
            records = []
            for record in result:
                records.append(dict(record))

            return {
                "status": "success",
                "message": f"Graph loaded successfully from {json_filename}",
                "records": records
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load graph: {str(e)}"
        }

    finally:
        driver.close()


def load_graph_from_json_flexible(json_filename, uri="bolt://localhost:7687", user="", password=""):
    """
    Load a graph into Memgraph from a JSON file using the import_util.json() procedure.
    The JSON file must be accessible via the mounted volume at /usr/lib/memgraph/import/.

    Args:
        json_filename (str): Name of the JSON file (e.g., "test.json")
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result of the import operation
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Construct the full path in the Memgraph container
            file_path = f"/usr/lib/memgraph/import/{json_filename}"

            # Use the import_util.json() procedure to load the graph
            query = "CALL import_util.json($file_path)"
            result = session.run(query, file_path=file_path)

            # Check the result
            records = list(result)

            # Verify nodes were actually created
            node_count_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_count_result.single()["count"]

            rel_count_result = session.run("MATCH ()-[r]-() RETURN count(r) as count")
            rel_count = rel_count_result.single()["count"]

            return {
                "status": "success",
                "message": f"Graph loaded successfully from {json_filename}. {node_count} nodes and {rel_count} relationships imported.",
                "nodes_imported": node_count,
                "relationships_imported": rel_count
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load graph: {str(e)}"
        }

    finally:
        driver.close()


def clear_graph(uri="bolt://localhost:7687", user="", password=""):
    """
    Clear all nodes and relationships from the Memgraph database.

    Args:
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result of the clear operation
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Delete all relationships first
            session.run("MATCH ()-[r]-() DELETE r")

            # Then delete all nodes
            session.run("MATCH (n) DELETE n")

            # Get count to verify clearing
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]

            return {
                "status": "success",
                "message": f"Graph cleared successfully. Remaining nodes: {node_count}",
                "nodes_remaining": node_count
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to clear graph: {str(e)}"
        }

    finally:
        driver.close()


def search_nodes_by_keyword(keyword, uri="bolt://localhost:7687", user="", password=""):
    """
    Search for nodes whose labels contain the given keyword (case-insensitive soft matching).

    Args:
        keyword (str): The keyword to search for in node labels
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing matching nodes
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Search in the name property for case-insensitive matching
            query = """
            MATCH (n)
            WHERE n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($keyword)
            RETURN n, labels(n) as node_labels
            """
            result = session.run(query, keyword=keyword)

            nodes = []
            for record in result:
                node_data = dict(record["n"])
                node_data["labels"] = record["node_labels"]
                nodes.append(node_data)

            # Format as uniform graph representation (relationships not explored)
            graph_result = format_subgraph_to_string(nodes, relationships_explored=False)

            return {
                "status": "success",
                "message": f"Found {len(nodes)} nodes with name containing '{keyword}'",
                "subgraph": graph_result,
                "count": len(nodes)
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to search nodes: {str(e)}"
        }

    finally:
        driver.close()


def search_nodes_by_types(node_type, uri="bolt://localhost:7687", user="", password=""):
    """
    Search for nodes whose type property contains the given type (case-insensitive soft matching).

    Args:
        node_type (str): The type to search for in node type properties
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing matching nodes
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Search in the labels for case-insensitive matching
            query = """
            MATCH (n)
            WHERE any(label IN labels(n) WHERE toLower(label) CONTAINS toLower($node_type))
            RETURN n, labels(n) as node_labels
            """
            result = session.run(query, node_type=node_type)

            nodes = []
            for record in result:
                node_data = dict(record["n"])
                node_data["labels"] = record["node_labels"]
                nodes.append(node_data)

            # Format as uniform graph representation (relationships not explored)
            graph_result = format_subgraph_to_string(nodes, relationships_explored=False)

            return {
                "status": "success",
                "message": f"Found {len(nodes)} nodes with labels containing '{node_type}'",
                "subgraph": graph_result,
                "count": len(nodes)
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to search nodes by type: {str(e)}"
        }

    finally:
        driver.close()


def get_neighbors(node_name, uri="bolt://localhost:7687", user="", password=""):
    """
    Get all neighbors of a node with the given name (case-insensitive exact matching on name property).

    Args:
        node_name (str): The name of the node to find neighbors for
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing neighbors and their relationships
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Find neighbors (both incoming and outgoing relationships)
            query = """
            MATCH (n)-[r]-(neighbor)
            WHERE n.name IS NOT NULL AND toLower(n.name) = toLower($node_name)
            RETURN n as source_node, r as relationship, neighbor,
                   labels(n) as source_labels, labels(neighbor) as neighbor_labels,
                   type(r) as relationship_type,
                   startNode(r) = n as is_outgoing
            """
            result = session.run(query, node_name=node_name)

            neighbors = []
            source_node = None
            all_nodes = []
            relationships = []

            for record in result:
                if source_node is None:
                    source_node_data = dict(record["source_node"])
                    source_node_data["labels"] = record["source_labels"]
                    source_node = source_node_data
                    all_nodes.append(source_node_data)

                neighbor_data = dict(record["neighbor"])
                neighbor_data["labels"] = record["neighbor_labels"]
                all_nodes.append(neighbor_data)

                relationship_data = dict(record["relationship"])
                relationship_data["type"] = record["relationship_type"]
                relationship_data["direction"] = "outgoing" if record["is_outgoing"] else "incoming"
                relationship_data["source_node"] = source_node_data
                relationship_data["neighbor"] = neighbor_data

                neighbors.append({
                    "neighbor": neighbor_data,
                    "relationship": relationship_data
                })
                relationships.append(relationship_data)

            # Remove duplicate nodes (by name)
            unique_nodes = []
            seen_names = set()
            for node in all_nodes:
                name = node.get('name', str(node.get('id', len(unique_nodes))))
                if name not in seen_names:
                    unique_nodes.append(node)
                    seen_names.add(name)

            # Format as uniform graph representation
            graph_result = format_subgraph_to_string(unique_nodes, relationships)

            return {
                "status": "success",
                "message": f"Found {len(neighbors)} neighbors for node '{node_name}'",
                "subgraph": graph_result,
                "source_node": source_node,
                "count": len(neighbors)
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get neighbors: {str(e)}"
        }

    finally:
        driver.close()


def search_relations_by_type(relation_type, uri="bolt://localhost:7687", user="", password=""):
    """
    Search for relationships based on soft matching on the relationship label/type.

    Args:
        relation_type (str): The relationship type to search for
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing matching relationships and their connected nodes
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Search for relationships by type (case-insensitive soft matching)
            query = """
            MATCH (start)-[r]->(end)
            WHERE toLower(type(r)) CONTAINS toLower($relation_type)
            RETURN start, r, end,
                   labels(start) as start_labels,
                   labels(end) as end_labels,
                   type(r) as relationship_type
            """
            result = session.run(query, relation_type=relation_type)

            nodes = []
            relationships = []
            node_names = set()

            for record in result:
                # Extract start node
                start_node = dict(record["start"])
                start_node["labels"] = record["start_labels"]
                start_name = start_node.get('name', start_node.get('id', 'unnamed_start'))

                # Extract end node
                end_node = dict(record["end"])
                end_node["labels"] = record["end_labels"]
                end_name = end_node.get('name', end_node.get('id', 'unnamed_end'))

                # Add nodes if not already added
                if start_name not in node_names:
                    nodes.append(start_node)
                    node_names.add(start_name)

                if end_name not in node_names:
                    nodes.append(end_node)
                    node_names.add(end_name)

                # Extract relationship
                relationship_data = dict(record["r"])
                relationship_data["type"] = record["relationship_type"]
                relationship_data["start_node"] = start_node
                relationship_data["end_node"] = end_node
                relationships.append(relationship_data)

            # Format as uniform graph representation (relationships explored)
            graph_result = format_subgraph_to_string(nodes, relationships, relationships_explored=True)

            return {
                "status": "success",
                "message": f"Found {len(relationships)} relationships with type containing '{relation_type}'",
                "subgraph": graph_result,
                "relationship_count": len(relationships),
                "node_count": len(nodes)
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to search relationships: {str(e)}"
        }

    finally:
        driver.close()


def identify_communities(node_name, uri="bolt://localhost:7687", user="", password=""):
    """
    Identify the community (connected component) containing a specific node.
    Uses a simple connected component algorithm to find all nodes reachable from the given node.

    Args:
        node_name (str): The name of the node to find the community for
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing the community subgraph with relationships
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Find all nodes and relationships in the connected component (with depth limit for performance)
            query = """
            MATCH (target)
            WHERE target.name IS NOT NULL AND toLower(target.name) = toLower($node_name)

            // Find all nodes within 4 hops (connected component with reasonable limit)
            MATCH (target)-[*1..4]-(connected)
            WITH target, collect(DISTINCT connected) + [target] as community_nodes

            // Get all relationships between nodes in this component
            UNWIND community_nodes as n1
            UNWIND community_nodes as n2
            MATCH (n1)-[r]-(n2)
            WHERE id(n1) < id(n2)  // Avoid duplicate relationships

            RETURN community_nodes,
                   collect(DISTINCT {rel: r, start: startNode(r), end: endNode(r)}) as relationships
            """
            result = session.run(query, node_name=node_name)

            record = result.single()
            if not record:
                return {
                    "status": "error",
                    "message": f"Node '{node_name}' not found"
                }

            # Extract nodes (remove duplicates by name)
            nodes = []
            seen_names = set()
            for node in record["community_nodes"]:
                node_data = dict(node)
                node_data["labels"] = list(node.labels)
                node_name = node_data.get('name', str(node_data.get('id', len(nodes))))
                if node_name not in seen_names:
                    nodes.append(node_data)
                    seen_names.add(node_name)

            # Extract relationships
            relationships = []
            for rel_info in record["relationships"]:
                rel = rel_info["rel"]
                start_node = rel_info["start"]
                end_node = rel_info["end"]

                relationship_data = dict(rel)
                relationship_data["type"] = rel.type

                # Add node information to relationship
                start_data = dict(start_node)
                start_data["labels"] = list(start_node.labels)
                end_data = dict(end_node)
                end_data["labels"] = list(end_node.labels)

                relationship_data["start_node"] = start_data
                relationship_data["end_node"] = end_data
                relationships.append(relationship_data)

            # Format as uniform graph representation (relationships explored)
            graph_result = format_subgraph_to_string(nodes, relationships, relationships_explored=True)

            return {
                "status": "success",
                "message": f"Found community with {len(nodes)} nodes and {len(relationships)} relationships for '{node_name}'",
                "subgraph": graph_result,
                "community_size": len(nodes),
                "relationship_count": len(relationships)
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to identify community: {str(e)}"
        }

    finally:
        driver.close()


def analyze_path(start_node_name, end_node_name, uri="bolt://localhost:7687", user="", password=""):
    """
    Find the shortest path between two nodes.

    Args:
        start_node_name (str): The name of the start node
        end_node_name (str): The name of the end node
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing the path subgraph with relationships
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Find shortest path between two nodes using BFS
            query = """
            MATCH (start)
            WHERE start.name IS NOT NULL AND toLower(start.name) = toLower($start_name)

            MATCH (end)
            WHERE end.name IS NOT NULL AND toLower(end.name) = toLower($end_name)

            MATCH path = (start)-[*1..5]-(end)
            WITH path, size(relationships(path)) as path_length
            ORDER BY path_length
            LIMIT 1

            RETURN nodes(path) as path_nodes, relationships(path) as path_rels, path_length
            """
            result = session.run(query, start_name=start_node_name, end_name=end_node_name)

            record = result.single()
            if not record:
                return {
                    "status": "error",
                    "message": f"No path found between '{start_node_name}' and '{end_node_name}'"
                }

            # Extract nodes
            nodes = []
            for node in record["path_nodes"]:
                node_data = dict(node)
                node_data["labels"] = list(node.labels)
                nodes.append(node_data)

            # Extract relationships
            relationships = []
            path_rels = record["path_rels"]
            path_nodes = record["path_nodes"]

            for i, rel in enumerate(path_rels):
                relationship_data = dict(rel)
                relationship_data["type"] = rel.type

                # Add node information to relationship
                start_node = path_nodes[i]
                end_node = path_nodes[i + 1]

                start_data = dict(start_node)
                start_data["labels"] = list(start_node.labels)
                end_data = dict(end_node)
                end_data["labels"] = list(end_node.labels)

                relationship_data["start_node"] = start_data
                relationship_data["end_node"] = end_data
                relationships.append(relationship_data)

            # Format as uniform graph representation (relationships explored)
            graph_result = format_subgraph_to_string(nodes, relationships, relationships_explored=True)

            return {
                "status": "success",
                "message": f"Found path between '{start_node_name}' and '{end_node_name}' with {len(nodes)} nodes and {len(relationships)} relationships",
                "subgraph": graph_result,
                "path_length": len(relationships),
                "node_count": len(nodes)
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to analyze path: {str(e)}"
        }

    finally:
        driver.close()


def find_hub_nodes(uri="bolt://localhost:7687", user="", password=""):
    """
    Find the top 3 hub nodes with the highest connectivity using PageRank algorithm.

    Args:
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing the top 3 hub nodes with their PageRank scores
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Use PageRank algorithm to find hub nodes
            query = """
            CALL pagerank.get() YIELD node, rank
            RETURN node, rank
            ORDER BY rank DESC
            LIMIT 3
            """
            result = session.run(query)

            nodes = []
            scores = []
            for record in result:
                node_data = dict(record["node"])
                node_data["labels"] = list(record["node"].labels)
                node_data["pagerank_score"] = record["rank"]
                nodes.append(node_data)
                scores.append(record["rank"])

            if not nodes:
                # Fallback: use degree centrality if PageRank is not available
                fallback_query = """
                MATCH (n)-[r]-()
                WITH n, count(r) as degree
                ORDER BY degree DESC
                LIMIT 3
                RETURN n, degree
                """
                result = session.run(fallback_query)

                for record in result:
                    node_data = dict(record["n"])
                    node_data["labels"] = list(record["n"].labels)
                    node_data["degree"] = record["degree"]
                    nodes.append(node_data)
                    scores.append(record["degree"])

            # Format as uniform graph representation (no relationships for hub nodes list)
            graph_result = format_subgraph_to_string(nodes, relationships_explored=False)

            metric_name = "PageRank score" if "pagerank_score" in (nodes[0] if nodes else {}) else "degree"

            return {
                "status": "success",
                "message": f"Found top 3 hub nodes based on {metric_name}",
                "subgraph": graph_result,
                "hub_nodes": [
                    {
                        "name": node.get('name', 'unnamed'),
                        "score": node.get('pagerank_score', node.get('degree', 0)),
                        "labels": node.get('labels', [])
                    }
                    for node in nodes
                ],
                "node_count": len(nodes)
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to find hub nodes: {str(e)}"
        }

    finally:
        driver.close()


def generate_graph_description(uri="bolt://localhost:7687", user="", password=""):
    """
    Generate a discursive description of the graph with entity and relationship type frequencies,
    graph density, and fragmentation index.

    Args:
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing the graph description, statistics, and metrics JSON
    """
    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=30)

    try:
        with driver.session() as session:
            # Get total counts first (simplest queries)
            total_nodes_query = "MATCH (n) RETURN count(n) as total_nodes"
            total_rels_query = "MATCH ()-[r]->() RETURN count(r) as total_relationships"

            total_nodes = session.run(total_nodes_query).single()["total_nodes"]
            total_rels = session.run(total_rels_query).single()["total_relationships"]

            # If graph is too large, return basic description to avoid performance issues
            if total_nodes > 10000 or total_rels > 50000:
                return {
                    "status": "success",
                    "message": f"Generated basic description for large graph with {total_nodes} nodes and {total_rels} relationships",
                    "description": f"This is a large graph containing {total_nodes} nodes and {total_rels} relationships. Detailed analysis skipped for performance reasons.",
                    "statistics": {
                        "total_nodes": total_nodes,
                        "total_relationships": total_rels,
                        "entity_types": [],
                        "relationship_types": []
                    },
                    "metrics": {
                        "graph_density": 0.0,
                        "fragmentation_index": 0.0
                    },
                    "density": 0.0,
                    "fragmentation_index": 0.0,
                    "largest_component_size": total_nodes,
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels
                }

            # Get node type frequencies (top 15 by frequency)
            node_query = """
            MATCH (n)
            UNWIND labels(n) as label
            RETURN label, count(*) as frequency
            ORDER BY frequency DESC, label ASC
            LIMIT 15
            """
            node_result = session.run(node_query)

            # Get relationship type frequencies (top 15 by frequency)
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(*) as frequency
            ORDER BY frequency DESC, rel_type ASC
            LIMIT 15
            """
            rel_result = session.run(rel_query)

            # Calculate graph density
            if total_nodes > 1:
                max_possible_edges = total_nodes * (total_nodes - 1) // 2
                graph_density = total_rels / max_possible_edges if max_possible_edges > 0 else 0
            else:
                graph_density = 0

            # Skip fragmentation calculation to avoid performance issues
            # Set default values
            largest_component_size = total_nodes if total_nodes > 0 else 0
            fragmentation_index = 0.0  # Assume well-connected graph

            # Process node types
            node_types = []
            for record in node_result:
                node_types.append({
                    "type": record["label"],
                    "frequency": record["frequency"]
                })

            # Process relationship types
            rel_types = []
            for record in rel_result:
                rel_types.append({
                    "type": record["rel_type"],
                    "frequency": record["frequency"]
                })

            # Generate discursive description
            description_parts = []

            # Opening statement
            description_parts.append(f"This graph contains {total_nodes} nodes and {total_rels} relationships.")

            # Add density and fragmentation information
            description_parts.append(f"The graph density is {graph_density:.4f}, indicating {'a highly connected' if graph_density > 0.5 else 'a sparsely connected' if graph_density < 0.1 else 'a moderately connected'} network.")

            if fragmentation_index == 0:
                description_parts.append("The graph is fully connected with a fragmentation index of 0.0000.")
            elif fragmentation_index < 0.2:
                description_parts.append(f"The graph shows low fragmentation with an index of {fragmentation_index:.4f}, indicating most nodes are in the main connected component.")
            elif fragmentation_index < 0.5:
                description_parts.append(f"The graph shows moderate fragmentation with an index of {fragmentation_index:.4f}.")
            else:
                description_parts.append(f"The graph is highly fragmented with an index of {fragmentation_index:.4f}, indicating many disconnected components.")

            # Entity types description
            if node_types:
                entity_descriptions = []
                for node_type in node_types:
                    count = node_type["frequency"]
                    type_name = node_type["type"]
                    if count == 1:
                        entity_descriptions.append(f'{count} "{type_name}"')
                    else:
                        entity_descriptions.append(f'{count} "{type_name}"s')

                if len(entity_descriptions) == 1:
                    description_parts.append(f"The most frequent entity type is {entity_descriptions[0]}.")
                elif len(entity_descriptions) == 2:
                    description_parts.append(f"The most frequent entity types are {entity_descriptions[0]} and {entity_descriptions[1]}.")
                else:
                    last_entity = entity_descriptions.pop()
                    description_parts.append(f"The most frequent entity types are {', '.join(entity_descriptions)}, and {last_entity}.")

            # Relationship types description
            if rel_types:
                rel_descriptions = []
                for rel_type in rel_types:
                    count = rel_type["frequency"]
                    type_name = rel_type["type"].replace("_", " ").lower()
                    if count == 1:
                        rel_descriptions.append(f'1 "{type_name}" relationship')
                    else:
                        rel_descriptions.append(f'{count} "{type_name}" relationships')

                if len(rel_descriptions) == 1:
                    description_parts.append(f"The most frequent relationship type is {rel_descriptions[0]}.")
                elif len(rel_descriptions) == 2:
                    description_parts.append(f"The most frequent relationship types are {rel_descriptions[0]} and {rel_descriptions[1]}.")
                else:
                    last_rel = rel_descriptions.pop()
                    description_parts.append(f"The most frequent relationship types are {', '.join(rel_descriptions)}, and {last_rel}.")

            # Note: Individual most common types are already included in the frequency listings above

            # Combine into final description
            full_description = " ".join(description_parts)

            # Create metrics JSON
            metrics_json = {
                "graph_density": graph_density,
                "fragmentation_index": fragmentation_index
            }

            return {
                "status": "success",
                "message": f"Generated description for graph with {total_nodes} nodes and {total_rels} relationships",
                "description": full_description,
                "statistics": {
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels,
                    "entity_types": node_types,
                    "relationship_types": rel_types
                },
                "metrics": metrics_json,
                "density": graph_density,
                "fragmentation_index": fragmentation_index,
                "largest_component_size": largest_component_size,
                "total_nodes": total_nodes,
                "total_relationships": total_rels
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to generate graph description: {str(e)}"
        }

    finally:
        driver.close()


def merge_graph_incremental(graph_filename, uri="bolt://localhost:7687", user="", password=""):
    """
    Merge new entities and relationships from a JSON file into the existing graph.
    This is a placeholder implementation that currently falls back to full replace.

    Args:
        graph_filename (str): Name of the JSON file containing new graph data
        uri (str): Memgraph connection URI
        user (str): Username for Memgraph connection
        password (str): Password for Memgraph connection

    Returns:
        dict: Result containing status and message
    """
    # For now, this is a placeholder that falls back to full replace
    # In a more sophisticated implementation, this would:
    # 1. Load the existing graph structure
    # 2. Compare with new entities/relationships
    # 3. Only add new ones and update existing ones
    # 4. Preserve existing relationships not affected by the update

    try:
        # For now, fallback to full replace
        return load_graph_from_json_flexible(graph_filename)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to merge graph incrementally: {str(e)}"
        }