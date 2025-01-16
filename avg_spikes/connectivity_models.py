import numpy as np
import random

# Определение границ кластеров на основе размеров кластеров
def fram(cluster_sizes):
    frames = np.zeros(len(cluster_sizes))
    frames[0] = cluster_sizes[0]
    for i in range(1, len(cluster_sizes)):
        frames[i] = frames[i - 1] + cluster_sizes[i]
    return frames

# Проверка принадлежности узла к определенному кластеру
def clcheck(a, cluster_sizes):
    frames = fram(cluster_sizes)
    if a >= 0 and a < frames[0]:
        return 0
    else:
        for i in range(len(frames) - 1):
            if a >= frames[i] and a < frames[i + 1]:
                return i + 1
        return len(frames) - 1

# Генерация начальной матрицы связности SBM для внутрикластерных связей
def gensbm2(number_of_nodes, cluster_labels, p_within, cluster_sizes, directed):
    sbm = np.zeros((number_of_nodes, number_of_nodes))
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if i != j:
                if cluster_labels[i] == cluster_labels[j]:  # Проверка, находятся ли оба узла в одном кластере
                    if directed:
                        sbm[i, j] = np.random.choice([0, 1], p=[1 - p_within, p_within])
                    else:
                        edge_exists = np.random.choice([0, 1], p=[1 - p_within, p_within])
                        sbm[i, j] = sbm[j, i] = edge_exists
    return sbm

# Генерация начальных межкластерных связей
def generate_constant_inter_cluster(number_of_nodes, cluster_labels, p_between, directed):
    sbm = np.zeros((number_of_nodes, number_of_nodes))
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if i != j and cluster_labels[i] != cluster_labels[j]:  # Связи между разными кластерами
                if directed:
                    sbm[i, j] = np.random.choice([0, 1], p=[1 - p_between, p_between])
                else:
                    edge_exists = np.random.choice([0, 1], p=[1 - p_between, p_between])
                    sbm[i, j] = sbm[j, i] = edge_exists
    return sbm

# Функция для обновления внутрикластерных связей
def update_intra_cluster_connectivity(C_intra, cluster_labels, p_within):
    num_of_clusters = len(np.unique(cluster_labels))
    for cluster_idx in range(num_of_clusters):
        cluster_indices = np.where(np.array(cluster_labels) == cluster_idx)[0]
        total_possible_connections = len(cluster_indices) * (len(cluster_indices) - 1) // 2
        current_connections = np.sum(np.triu(C_intra[np.ix_(cluster_indices, cluster_indices)], k=1))
        desired_connections = int(p_within * total_possible_connections)
        connections_to_add = desired_connections - int(current_connections)
        if connections_to_add > 0:
            possible_pairs = [(i, j) for idx_i, i in enumerate(cluster_indices) 
                              for j in cluster_indices[idx_i+1:] if C_intra[i, j] == 0]
            if len(possible_pairs) == 0:
                continue
            num_new_connections = min(connections_to_add, len(possible_pairs))
            new_connections = random.sample(possible_pairs, num_new_connections)
            for i, j in new_connections:
                C_intra[i, j] = 1
                C_intra[j, i] = 1
    return C_intra

# Функция для обновления межкластерных связей
def update_inter_cluster_connectivity(C_between, cluster_labels, p_between):
    clusters = np.unique(cluster_labels)
    num_clusters = len(clusters)
    for idx_a in range(num_clusters):
        for idx_b in range(idx_a + 1, num_clusters):
            cluster_a_indices = np.where(np.array(cluster_labels) == clusters[idx_a])[0]
            cluster_b_indices = np.where(np.array(cluster_labels) == clusters[idx_b])[0]
            total_possible_connections = len(cluster_a_indices) * len(cluster_b_indices)
            current_connections = np.sum(C_between[np.ix_(cluster_a_indices, cluster_b_indices)])
            desired_connections = int(p_between * total_possible_connections)
            connections_to_add = desired_connections - int(current_connections)
            if connections_to_add > 0:
                possible_pairs = [(i, j) for i in cluster_a_indices 
                                  for j in cluster_b_indices if C_between[i, j] == 0]
                if len(possible_pairs) == 0:
                    continue
                num_new_connections = min(connections_to_add, len(possible_pairs))
                new_connections = random.sample(possible_pairs, num_new_connections)
                for i, j in new_connections:
                    C_between[i, j] = 1
                    C_between[j, i] = 1
    return C_between
