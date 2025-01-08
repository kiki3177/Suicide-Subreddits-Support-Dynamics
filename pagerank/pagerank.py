import os

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, csr_matrix


linked_suicide = pd.read_csv("../posts_categorization/linked_submissions_comments/2023_depression_linked_llama_gemma_qwen.csv", dtype='str', encoding='utf-8',lineterminator='\n')

# Create nodes
nodes = pd.concat([linked_suicide['submitter_username'], linked_suicide['commenter_username']]).drop_duplicates().reset_index(drop=True)
node_to_index = {node: i for i, node in enumerate(nodes)}
num_nodes = len(nodes)

# Create edges
edges = []
for _, row in linked_suicide.iterrows():
    source_node = row['submitter_username']
    target_node = row['commenter_username']
    if source_node != target_node:  # Avoid self-loops
        edges.append((source_node, target_node))

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {len(edges)}")

# Parameters
damping_factor = 0.85
num_iterations = 100
tolerance = 1e-6

# Create sparse adjacency matrix
adj_matrix = dok_matrix((num_nodes, num_nodes), dtype=np.float32)
for source, target in edges:
    if source in node_to_index and target in node_to_index:
        adj_matrix[node_to_index[source], node_to_index[target]] = 1

# Convert to CSR format for efficient arithmetic and matrix multiplication
adj_matrix = adj_matrix.tocsr()

# Normalize sparse adjacency matrix by outgoing edges
outdegree = np.array(adj_matrix.sum(axis=1)).flatten()
outdegree[outdegree == 0] = 1  # Avoid division by zero
transition_matrix = adj_matrix.multiply(1 / outdegree[:, None])

# Initialize PageRank scores
pagerank = np.ones(num_nodes) / num_nodes

# Iterative computation
for iteration in range(num_iterations):
    new_pagerank = ((1 - damping_factor) / num_nodes) + \
                   damping_factor * (transition_matrix.T @ pagerank)

    if np.linalg.norm(new_pagerank - pagerank, ord=1) < tolerance:
        print(f"Converged after {iteration + 1} iterations")
        break

    pagerank = new_pagerank

pagerank_scores = {nodes[i]: score for i, score in enumerate(pagerank)}
print("Final PageRank Scores:")
for node, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{node}: {score:.4f}")


pagerank_df = pd.DataFrame(list(pagerank_scores.items()), columns=['node', 'pagerank'])
pagerank_df = pagerank_df.sort_values(by='pagerank', ascending=False)
pagerank_df.to_csv("pagerank_results/depression_pagerank_scores_2023.csv", index=False)
