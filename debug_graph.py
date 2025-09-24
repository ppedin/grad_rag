#!/usr/bin/env python3
"""
Debug script to check what's actually in the Memgraph database.
"""

from neo4j import GraphDatabase

def debug_graph():
    """Debug what's actually in the graph database."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))

    try:
        with driver.session() as session:
            print("=== Checking all nodes ===")
            result = session.run("MATCH (n) RETURN n LIMIT 10")

            for record in result:
                node = record["n"]
                print(f"Node: {dict(node)}")
                print(f"Labels: {list(node.labels)}")
                print("---")

            print("\n=== Checking node count ===")
            result = session.run("MATCH (n) RETURN count(n) as total")
            count = result.single()["total"]
            print(f"Total nodes in database: {count}")

            print("\n=== Testing Retief search ===")
            result = session.run("MATCH (n) WHERE n.name CONTAINS 'Retief' RETURN n")
            nodes = list(result)
            print(f"Nodes with 'Retief' in name: {len(nodes)}")
            for record in nodes:
                print(f"Found: {dict(record['n'])}")

    finally:
        driver.close()

if __name__ == "__main__":
    debug_graph()