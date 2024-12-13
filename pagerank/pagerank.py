import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, csr_matrix

# Load the dataset
linked_suicide = pd.read_csv("../data_preparation/linked_datasets_2022_2023/2022_2023_linked_suicide.csv")

# Create nodes
nodes = pd.concat([linked_suicide['author_x'], linked_suicide['author_y']]).drop_duplicates().reset_index(drop=True)
node_to_index = {node: i for i, node in enumerate(nodes)}
num_nodes = len(nodes)

# Create edges
edges = []
for _, row in linked_suicide.iterrows():
    source_node = row['author_x']
    target_node = row['author_y']
    if source_node != target_node:  # Avoid self-loops
        edges.append((source_node, target_node))

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {len(edges)}")

# Parameters
damping_factor = 0.85  # Typical value for PageRank
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

    # Check for convergence
    if np.linalg.norm(new_pagerank - pagerank, ord=1) < tolerance:
        print(f"Converged after {iteration + 1} iterations")
        break

    pagerank = new_pagerank

# Output results
pagerank_scores = {nodes[i]: score for i, score in enumerate(pagerank)}
print("Final PageRank Scores:")
for node, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{node}: {score:.4f}")

# Convert PageRank scores to a DataFrame
pagerank_df = pd.DataFrame(list(pagerank_scores.items()), columns=['Node', 'PageRank'])
pagerank_df = pagerank_df.sort_values(by='PageRank', ascending=False)

pagerank_df.to_csv("suicide_pagerank_scores.csv", index=False)
print("PageRank scores have been saved to 'suicide_pagerank_scores.csv'.")
