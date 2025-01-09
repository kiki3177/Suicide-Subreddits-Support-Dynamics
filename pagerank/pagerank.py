import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, csr_matrix


subreddit_name = "suicide"
year = 2023
linked_suicide = pd.read_csv(f"../posts_categorization/linked_submissions_comments/{year}_{subreddit_name}_linked_llama_gemma_qwen.csv", dtype='str', encoding='utf-8',lineterminator='\n')

# Create nodes
nodes = pd.concat([linked_suicide['submitter_username'], linked_suicide['commenter_username']]).reset_index(drop=True)


user_name_set = set()
for user in nodes:
    user_name_set.add(user)
user_names = sorted(list(user_name_set))


node_to_index = dict()
counter = 0
for user in user_names:
    node_to_index[user] = counter
    counter += 1

num_nodes = len(user_name_set)
print(f"Number of nodes: {num_nodes}")



# Create edges
edges = []
for _, row in linked_suicide.iterrows():
    source_node = row['submitter_username']
    target_node = row['commenter_username']
    if source_node != target_node:  # Avoid self-loops
        edges.append((source_node, target_node))
print(f"Number of edges: {len(edges)}")



# Parameters
damping_factor = 0.85
num_iterations = 20
tolerance = 1e-6



# Create sparse adjacency matrix
adj_dict = defaultdict(int)
for source, target in edges:
    adj_dict[node_to_index[source], node_to_index[target]] += 1


adj_matrix = dok_matrix((num_nodes, num_nodes), dtype=np.float32)
for (i, j), value in adj_dict.items():
    adj_matrix[i, j] = value


# Convert to CSR format for efficient arithmetic and matrix multiplication
adj_matrix = adj_matrix.tocsr()

# Normalize sparse adjacency matrix by outgoing edges
outdegree = np.array(adj_matrix.sum(axis=0)).flatten()
outdegree[outdegree == 0] = 1  # Avoid division by zero
transition_matrix = adj_matrix.multiply(1 / outdegree[:, None])

# Initialize PageRank scores
pagerank = np.ones(num_nodes) / num_nodes



# Iterative computation
for iteration in range(num_iterations):
    new_pagerank = ((1 - damping_factor) / num_nodes) + damping_factor * (transition_matrix.T  @ pagerank)

    # if np.linalg.norm(new_pagerank - pagerank, ord=1) < tolerance:
    #     print(f"Converged after {iteration + 1} iterations")
    #     break

    pagerank = new_pagerank
    print(iteration)

pagerank_scores = {user_names[i]: score for i, score in enumerate(pagerank)}
print("Final PageRank Scores:")
for node, score in sorted(pagerank_scores.items(), key=lambda x: (x[1]), reverse=True):
    print(f"{node}: {score:.4f}")


pagerank_df = pd.DataFrame(list(pagerank_scores.items()), columns=['node', 'pagerank'])
pagerank_df = pagerank_df.sort_values(by='pagerank', ascending=False)
pagerank_df.to_csv(f"pagerank_results/{subreddit_name}_pagerank_scores_{year}.csv", index=False)
