import os
import io
import csv
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from brian2 import *
from centrality import GraphCentrality
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time
import networkx as nx
import pandas as pd

# Установка параметров вывода NumPy
np.set_printoptions(threshold=np.inf)

# -- Создание/проверка директории для сохранения результатов --
directory_path = 'results_ext_test1'
if os.path.exists(directory_path):
    print(f"Директория '{directory_path}' существует")
else:
    os.mkdir(directory_path)
    print(f"Директория '{directory_path}' создана")

# -- Инициализация параметров сети --
n_neurons = 500
num_of_clusters = 2
cluster_sizes = [250, 250]

# Основные диапазоны параметров
# p_within_values = np.arange(0.60, 0.61, 0.1)  # [0.7, 0.8, 0.9, 1.0]
p_within_values = np.arange(0.2, 0.21, 0.05)  # [0.7, 0.8, 0.9, 1.0]
p_input_values = np.arange(0.1, 0.21, 0.05)

num_tests = 3
J = 0.5
J2 = 0.5
g = 5
epsilon = 0.1
nu_ext_over_nu_thr = 4
D = 1.5 * ms
rate_tick_step = 50
t_range = [0, 1000]
rate_range = [0, 200]
time_window_size = 100  # in ms
refractory_period = 10*ms # max freq=100Hz

# Для осцилляций/стимулов
oscillation_frequencies = [10]
use_stdp_values = [False]
I0_values = [1000] # pA


measure_names = [
    "degree",
    "betweenness",
    "eigenvector",
    "pagerank",
    "percolation",
    "random"
]

# -- Список времён симуляции (в мс) --
simulation_times = [1000]


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

# -- Формируем вектор меток кластеров --
cluster_labels = []
for i in range(n_neurons):
    cluster_labels.append(clcheck(i, cluster_sizes))

def generate_sbm_with_high_centrality(
    n, cluster_sizes, p_intra, p_inter,
    target_cluster_index, proportion_high_centrality=0.2,
    centrality_type="degree", boost_factor=2
):
    """
    Generate an SBM where a proportion of nodes in a fixed cluster have high centrality.

    Parameters:
        n: Total number of nodes.
        cluster_sizes: List of sizes for each cluster.
        p_intra: Probability of intra-cluster edges.
        p_inter: Probability of inter-cluster edges.
        target_cluster_index: Index of the cluster to modify (0-based).
        proportion_high_centrality: Proportion of nodes in the target cluster to have high centrality.
        centrality_type: Type of centrality to boost ("degree", "betweenness", "eigenvector", "pagerank",
                                                     "local_clustering", "closeness", "harmonic", "percolation", "cross_clique").
        boost_factor: Factor by which to increase the centrality of selected nodes.

    Returns:
        G: Generated graph.
    """
    # Step 1: Generate the initial SBM
    G = nx.Graph()
    current_node = 0
    clusters = []

    for size in cluster_sizes:
        cluster = list(range(current_node, current_node + size))
        clusters.append(cluster)
        current_node += size

    # Устанавливаем связи внутри каждого кластера
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if np.random.rand() < p_intra:
                    G.add_edge(cluster[i], cluster[j])

    # Устанавливаем связи между кластерами
    for idx1 in range(len(clusters)):
        for idx2 in range(idx1 + 1, len(clusters)):
            for u in clusters[idx1]:
                for v in clusters[idx2]:
                    if np.random.rand() < p_inter:
                        G.add_edge(u, v)

    # for i in range(len(clusters)):
    #     for j in range(len(clusters)):
    #         for u in clusters[i]:
    #             for v in clusters[j]:
    #                 if u != v:
    #                     prob = p_intra if i == j else p_inter
    #                     if np.random.rand() < prob:
    #                         G.add_edge(u, v)

    # Step 2: Identify the target cluster
    target_cluster = clusters[target_cluster_index]
    num_high_centrality_nodes = int(proportion_high_centrality * len(target_cluster))

    # Randomly select nodes to boost centrality
    high_centrality_nodes = np.random.choice(target_cluster, size=num_high_centrality_nodes, replace=False)

    # Step 3: Boost centrality of selected nodes based on centrality type
    if centrality_type == "degree":
        # Add edges to increase degree centrality
        for node in high_centrality_nodes:
            potential_neighbors = [v for v in G.nodes if v != node and not G.has_edge(node, v)]
            num_new_edges = int(boost_factor * G.degree[node])  # Scale by boost factor
            new_neighbors = np.random.choice(potential_neighbors, size=min(num_new_edges, len(potential_neighbors)), replace=False)
            for neighbor in new_neighbors:
                G.add_edge(node, neighbor)

    elif centrality_type == "betweenness":
        # Add edges to key nodes in other clusters to increase betweenness
        for node in high_centrality_nodes:
            for other_cluster in clusters:
                if other_cluster != target_cluster:
                    for neighbor in other_cluster:
                        if not G.has_edge(node, neighbor) and np.random.rand() < boost_factor * p_inter:
                            G.add_edge(node, neighbor)

    elif centrality_type == "eigenvector":
        # Increase connectivity to high-degree nodes to boost eigenvector centrality
        all_high_degree_nodes = sorted(G.nodes, key=lambda x: G.degree[x], reverse=True)[:int(0.1 * n)]  # Top 10% high-degree nodes
        for node in high_centrality_nodes:
            for neighbor in all_high_degree_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < boost_factor * p_inter:
                    G.add_edge(node, neighbor)

    elif centrality_type == "pagerank":
        # Add edges to high-degree nodes to increase PageRank centrality
        all_high_degree_nodes = sorted(G.nodes, key=lambda x: G.degree[x], reverse=True)[:int(0.1 * n)]  # Top 10% high-degree nodes
        for node in high_centrality_nodes:
            for neighbor in all_high_degree_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < boost_factor * p_inter:
                    G.add_edge(node, neighbor)

    elif centrality_type == "local_clustering":
        # Add triangles to increase local clustering coefficient
        for node in high_centrality_nodes:
            neighbors = list(G.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not G.has_edge(neighbors[i], neighbors[j]):
                        if np.random.rand() < boost_factor:
                            G.add_edge(neighbors[i], neighbors[j])

    elif centrality_type == "closeness":
        # Connect nodes to central nodes in other clusters to reduce shortest path lengths
        central_nodes = sorted(G.nodes, key=lambda x: nx.closeness_centrality(G)[x], reverse=True)[:int(0.1 * n)]
        for node in high_centrality_nodes:
            for neighbor in central_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < boost_factor * p_inter:
                    G.add_edge(node, neighbor)

    elif centrality_type == "harmonic":
        # Connect nodes to central nodes in other clusters to improve harmonic centrality
        central_nodes = sorted(G.nodes, key=lambda x: nx.harmonic_centrality(G)[x], reverse=True)[:int(0.1 * n)]
        for node in high_centrality_nodes:
            for neighbor in central_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < boost_factor * p_inter:
                    G.add_edge(node, neighbor)

    elif centrality_type == "percolation":
        # Add edges to high-degree nodes to simulate percolation centrality
        all_high_degree_nodes = sorted(G.nodes, key=lambda x: G.degree[x], reverse=True)[:int(0.1 * n)]
        for node in high_centrality_nodes:
            for neighbor in all_high_degree_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < boost_factor * p_inter:
                    G.add_edge(node, neighbor)

    elif centrality_type == "cross_clique":
        # Add edges to nodes in different cliques to increase cross-clique centrality
        cliques = list(nx.find_cliques(G))
        for node in high_centrality_nodes:
            for clique in cliques:
                if node not in clique:
                    for neighbor in clique:
                        if not G.has_edge(node, neighbor) and np.random.rand() < boost_factor * p_inter:
                            G.add_edge(node, neighbor)
    
    elif centrality_type == "random":
        # Для случайного увеличения центральности добавляются ребра ко всем потенциальным соседям
        # с вероятностью boost_factor * p_inter
        for node in high_centrality_nodes:
            potential_neighbors = [v for v in G.nodes if v != node and not G.has_edge(node, v)]
            for neighbor in potential_neighbors:
                if np.random.rand() < boost_factor * p_inter:
                    G.add_edge(node, neighbor)

    else:
        raise ValueError("Unsupported centrality type. Choose from 'degree', 'betweenness', 'eigenvector', 'pagerank', "
                         "'local_clustering', 'closeness', 'harmonic', 'percolation', 'cross_clique'.")

    return G

class GraphCentrality:
    def __init__(self, adjacency_matrix):
        """
        Инициализация объекта GraphCentrality с заданной матрицей смежности.
        
        Параметры:
        adjacency_matrix (numpy.ndarray): Квадратная матрица смежности, описывающая граф.
        """
        self.adjacency_matrix = adjacency_matrix
        self.graph = nx.from_numpy_array(adjacency_matrix)

    def calculate_betweenness_centrality(self):
        """
        Вычисление посреднической центральности (betweenness centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их посредническая центральность.
        """
        return nx.betweenness_centrality(self.graph)

    def calculate_eigenvector_centrality(self):
        """
        Вычисление векторной центральности (eigenvector centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их векторная центральность.
        """
        return nx.eigenvector_centrality(self.graph)

    def calculate_pagerank_centrality(self, alpha=0.85):
        """
        Вычисление центральности на основе PageRank.
        
        Параметры:
        alpha (float): Коэффициент затухания (damping factor) для PageRank (по умолчанию 0.85).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их PageRank-центральность.
        """
        return nx.pagerank(self.graph, alpha=alpha)

    def calculate_flow_coefficient(self):
        """
        Вычисление «коэффициента потока» (flow coefficient) для всех узлов графа.
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их «коэффициент потока».
        """
        flow_coefficient = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < 2:
                flow_coefficient[node] = 0.0
            else:
                edge_count = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if self.graph.has_edge(neighbors[i], neighbors[j]):
                            edge_count += 1
                flow_coefficient[node] = (2 * edge_count) / (len(neighbors) * (len(neighbors) - 1))
        return flow_coefficient

    def calculate_degree_centrality(self):
        """
        Вычисление степенной центральности (degree centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их степенная центральность.
        """
        return nx.degree_centrality(self.graph)

    def calculate_closeness_centrality(self):
        """
        Вычисление близостной центральности (closeness centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их близостная центральность.
        """
        return nx.closeness_centrality(self.graph)

    def calculate_harmonic_centrality(self):
        """
        Вычисление гармонической центральности (harmonic centrality).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их гармоническая центральность.
        """
        return nx.harmonic_centrality(self.graph)

    def calculate_percolation_centrality(self, attribute=None):
        """
        Вычисление перколяционной центральности (percolation centrality).
        
        Параметры:
        attribute (str или dict): Атрибут узла для моделирования «перколяции» (по умолчанию None).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их перколяционная центральность.
        """
        if attribute is None:
            attribute = {node: 1 for node in self.graph.nodes()}
        return nx.percolation_centrality(self.graph, states=attribute)

    def calculate_cross_clique_centrality(self):
        """
        Вычисление центральности «cross-clique» (количество кликовых покрытий узла).
        
        Возвращает:
        dict: Словарь, где ключи – индексы узлов, а значения – их центральность по числу клик.
        """
        cross_clique_centrality = {}
        cliques = list(nx.find_cliques(self.graph))
        for node in self.graph.nodes():
            cross_clique_centrality[node] = sum(node in clique for clique in cliques)
        return cross_clique_centrality




def plot_spikes(ax_spikes, spike_times, spike_indices, time_window_size, t_range, sim_time, oscillation_frequency, use_stdp, measure_name):
    
    ax_spikes.scatter(spike_times, spike_indices, marker='|')
    step_size = time_window_size
    total_time_ms = sim_time / ms
    time_steps = np.arange(0, total_time_ms + step_size, step_size)
    for t in time_steps:
        ax_spikes.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)

    ax_spikes.set_xlim(t_range)
    ax_spikes.set_xlabel("t [ms]")
    ax_spikes.set_ylabel("Neuron index")
    ax_spikes.set_title(f"Spike Raster Plot\nFrequency={oscillation_frequency} Hz, STDP={'On' if use_stdp else 'Off'}, Window={time_window_size} ms, {measure_name}", fontsize=16)

def plot_rates(ax_rates, N1, N2, rate_monitor, t_range):
    ax_rates.set_title(f'Number of spikes\n neurons 0-{N1}', fontsize=16)
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz, label=f'Group 1 (0-{n_neurons/2})')
    ax_rates.set_xlim(t_range)
    ax_rates.set_xlabel("t [ms]")


def plot_rates2(ax_rates2, N1, N2, rate_monitor2, t_range):
    ax_rates2.set_title(f'Number of spikes\n neurons {N1}-{N2}', fontsize=16)
    ax_rates2.plot(rate_monitor2.t / ms, rate_monitor2.rate / Hz, label=f'Group 2 ({n_neurons/2}-{n_neurons})')
    ax_rates2.set_xlim(t_range)
    ax_rates2.set_xlabel("t [ms]")    

def plot_psd(rate_monitor, N1, N2, ax_psd):
    rate = rate_monitor.rate / Hz - np.mean(rate_monitor.rate / Hz)
    from numpy.fft import rfft, rfftfreq
    N = len(rate_monitor.t)
    if N > 0:
        # Определение частоты дискретизации на основе временного шага
        dt = float((rate_monitor.t[1] - rate_monitor.t[0]) / ms)  # в мс
        sampling_rate = 1000 / dt  # преобразуем мс в Гц
        # Применяем оконную функцию (окно Ханнинга)
        window = np.hanning(N)
        rate_windowed = rate * window
        # Ограничение до нужного диапазона частот
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate_windowed)) / N 
        yn = yn[:max_point]
    
        ax_psd.set_title(f"PSD neurons 0-{N1}\n", fontsize=16)
        ax_psd.plot(x, yn, c='k', label='Function')
        ax_psd.set_xlim([0,100])
        ax_psd.set_xlabel('Hz')


def plot_psd2(rate_monitor2, N1, N2, ax_psd2):
    rate = rate_monitor2.rate / Hz - np.mean(rate_monitor2.rate / Hz)
    from numpy.fft import rfft, rfftfreq
    N = len(rate_monitor2.t)
    if N > 0:
        # Определение частоты дискретизации на основе временного шага
        dt = float((rate_monitor2.t[1] - rate_monitor2.t[0]) / ms)  # в мс
        sampling_rate = 1000 / dt  # преобразуем мс в Гц
        # Применяем оконную функцию (окно Ханнинга)
        window = np.hanning(N)
        rate_windowed = rate * window
        # Ограничение до нужного диапазона частот
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate_windowed)) / N 
        yn = yn[:max_point]
    
        ax_psd2.set_title(f"PSD neurons {N1}-{N2}\n", fontsize=16)
        ax_psd2.plot(x, yn, c='k', label='Function')
        ax_psd2.set_xlim([0,100])
        ax_psd2.set_xlabel('Hz')


     

def plot_spectrogram(rate_monitor, rate_monitor2, oscillation_frequency, use_stdp, ax_spectrogram):
    yf = rfft(rate_monitor.rate / Hz)
    N_freq = len(yf)
    xf = rfftfreq(len(rate_monitor.t), 1 / 10000)[:N_freq]
    ax_spectrogram.plot(xf, np.abs(yf), label='0-50 neurons')
    
    yf2 = rfft(rate_monitor2.rate / Hz)
    N_freq2 = len(yf2)
    xf2 = rfftfreq(len(rate_monitor2.t), 1 / 10000)[:N_freq2]
    ax_spectrogram.plot(xf2, np.abs(yf2), label='50-100 neurons')
    
    ax_spectrogram.set_xlim(0, 1000)
    ax_spectrogram.legend()
    ax_spectrogram.set_title(f'Global Frequencies\nFrequency={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}', fontsize=16)


def plot_connectivity(n_neurons, exc_synapses, inhib_synapses, connectivity2, centrality, p_in_measured, p_out_measured):
    W = np.zeros((n_neurons, n_neurons))
    W[exc_synapses.i[:], exc_synapses.j[:]] = exc_synapses.w[:]
    W[inhib_synapses.i[:], inhib_synapses.j[:]] = inhib_synapses.w[:]

    # Highlight neurons with current applied based on centrality
    highlighted_neurons = np.unique(centrality)
    for neuron in highlighted_neurons:
        connectivity2.plot([neuron], [neuron], 'ro', markersize=1, label=f'Neuron {neuron}')

    connectivity2.set_title(f'Weight Matrix,\np_in_measured={p_in_measured:.3f},\n p_out_measured={p_out_measured:.3f}', fontsize=14)
    connectivity2.matshow(W, cmap='viridis')
    connectivity2.set_xlabel('Post-synaptic neuron', fontsize=12)
    connectivity2.set_ylabel('Pre-synaptic neuron', fontsize=12)

def print_centrality(C_total, cluster_nodes, p_input, measure_name):
    """
    Вычисляет указанную меру центральности (measure_name), выводит список топ-узлов 
    по этой метрике (но только среди узлов одного кластера), а также возвращает этот список.

    Параметры:
    ----------
    C_total : numpy.ndarray
        Квадратная матрица смежности, описывающая граф.
    cluster_nodes : int или iterable
        Число узлов или список узлов (индексов), составляющих кластер, 
        по которым выбираем топ-узлы.
    p_input : float
        Доля от числа узлов кластера (len(cluster_nodes)), которую берём для формирования топ-листа.
        Если cluster_nodes - целое число, то используется именно это число для определения размера кластера.
    measure_name : str
        Название метрики центральности, которую нужно вычислить.

    Возвращает:
    ----------
    list
        Список индексов узлов, являющихся топ-узлами по заданной метрике (только внутри кластера).
    """
    if measure_name != 'random':
        submatrix = C_total[0:int(n_neurons/2), 0:int(n_neurons/2)] # внутри кластера
        graph_centrality = GraphCentrality(submatrix)

        # Сопоставление названий метрик с соответствующими методами
        measure_func_map = {
            "betweenness": graph_centrality.calculate_betweenness_centrality,
            "eigenvector": graph_centrality.calculate_eigenvector_centrality,
            "pagerank": graph_centrality.calculate_pagerank_centrality,
            "flow": graph_centrality.calculate_flow_coefficient,
            "degree": graph_centrality.calculate_degree_centrality,
            "closeness": graph_centrality.calculate_closeness_centrality,
            "harmonic": graph_centrality.calculate_harmonic_centrality,
            "percolation": graph_centrality.calculate_percolation_centrality,
            "cross_clique": graph_centrality.calculate_cross_clique_centrality
        }
            
        if measure_name not in measure_func_map:
            raise ValueError(
                f"Метрика '{measure_name}' не поддерживается. "
                f"Доступные метрики: {list(measure_func_map.keys())}."
            )

        measure_values = measure_func_map[measure_name]()
        for node in measure_values:
            measure_values[node] = round(measure_values[node], 5)

        if isinstance(cluster_nodes, int):
            cluster_list = list(range(cluster_nodes))
        else:
            cluster_list = list(cluster_nodes)

        measure_values_cluster = {
            node: measure_values[node] 
            for node in cluster_list
            if node in measure_values
        }
        top_k = int(p_input * len(cluster_list)/2)
        sorted_neurons_cluster = sorted(
            measure_values_cluster,
            key=lambda n: measure_values_cluster[n],
            reverse=True
        )
        # print("sorted_neurons_cluster", sorted_neurons_cluster)
        top_neurons = sorted_neurons_cluster[:top_k]
        # top_neurons = sorted_neurons_cluster[-top_k:]
        # print(top_neurons)
        # print(last_neurons)
    else:
        cluster1_indices = np.arange(0, int(n_neurons/2))
        num_chosen = int(p_input * len(cluster1_indices))
        top_neurons = np.random.choice(cluster1_indices, size=num_chosen, replace=False)

    return top_neurons



def save_csv_data(csv_filename, data_for_csv):
    """
    Сохранение данных в CSV
    """
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for row in data_for_csv:
            writer.writerow(row)

def save_gif(images, filename, duration=3000, loop=0):
    """
    Создание анимации из последовательности изображений
    """
    imageio.mimsave(filename, images, duration=duration, loop=loop)



def plot_3d_spike_data(
    detailed_spike_data_for_3d,
    p_within_values,
    I0_value,
    oscillation_frequency,
    use_stdp,
    current_time,
    time_window_size,
    directory_path='results_ext_test1',
    measure_name=None
):
    """
    Построение 3D-графика (Time vs p_between vs Avg Spikes)
    на основе сохранённых данных по временным окнам.
    """

    # Проверяем наличие данных для заданного I0_value
    if I0_value not in detailed_spike_data_for_3d:
        print(f"Нет данных для I0={I0_value}pA.")
        return

    for p_within in p_within_values:
        p_within_str = f"{p_within:.2f}"
        if p_within_str not in detailed_spike_data_for_3d[I0_value]:
            print(f"Нет данных для p_within={p_within_str}. Пропуск.")
            continue

        p_between_list = sorted(detailed_spike_data_for_3d[I0_value][p_within_str].keys())
        if len(p_between_list) == 0:
            print(f"Нет данных для p_within={p_within_str}. Пропуск.")
            continue

        # Получаем временные точки из первого p_between
        sample_p_between = p_between_list[0]
        time_array = detailed_spike_data_for_3d[I0_value][p_within_str][sample_p_between].get("time", np.array([]))
        
        # Исправление проверки пустоты массива
        if time_array.size == 0:
            print(f"Нет временных данных для p_within={p_within_str}, p_between={sample_p_between}. Пропуск.")
            continue

        # Проверяем, что все p_between имеют одинаковые временные окна
        consistent_time = True
        for p_btw in p_between_list:
            current_time_array = detailed_spike_data_for_3d[I0_value][p_within_str][p_btw].get("time", np.array([]))
            if current_time_array.size == 0 or len(current_time_array) != len(time_array):
                consistent_time = False
                print(f"Несоответствие временных окон для p_within={p_within_str}, p_between={p_btw}.")
                break
        if not consistent_time:
            print(f"Несоответствие временных окон для p_within={p_within_str}. Пропуск построения 3D-графика.")
            continue

        # Создаём сетки для поверхности
        Time, P_between = np.meshgrid(time_array, p_between_list)

        # Инициализируем Z
        Z = np.zeros(Time.shape)

        # Заполняем Z значениями спайков
        for i, p_btw in enumerate(p_between_list):
            spikes_arr = detailed_spike_data_for_3d[I0_value][p_within_str][p_btw].get("spikes_list", [])
            if not spikes_arr:
                Z[i, :] = 0
            else:
                # Преобразуем список в массив и проверяем длину
                spikes_arr = np.array(spikes_arr)
                if len(spikes_arr) < Z.shape[1]:
                    # Дополняем нулями, если длина меньше
                    spikes_arr = np.pad(spikes_arr, (0, Z.shape[1] - len(spikes_arr)), 'constant')
                elif len(spikes_arr) > Z.shape[1]:
                    # Обрезаем, если длина больше
                    spikes_arr = spikes_arr[:Z.shape[1]]
                Z[i, :] = spikes_arr

        # Проверка типов данных в Z
        if not np.issubdtype(Z.dtype, np.number):
            print(f"Некорректные типы данных в Z для p_within={p_within_str}. Пропуск построения графика.")
            continue

        # Проверка на финитные значения в Z
        if not np.isfinite(Z).all():
            print(f"Некоторые значения в Z не являются конечными для p_within={p_within_str}. Пропуск построения графика.")
            continue

        # Строим 3D-график
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        surf = ax_3d.plot_surface(
            Time,        # X
            P_between,   # Y
            Z,           # Z
            cmap='viridis',
            edgecolor='none'
        )

        ax_3d.set_xlabel('Time [ms]')
        ax_3d.set_ylabel('p_between')
        ax_3d.set_zlabel('Avg Spikes in 100 ms window')
        ax_3d.set_zlim(0,1)
        ax_3d.set_title(
            f'3D Surface: Time vs p_between vs Avg Spikes\n'
            f'I0={I0_value}pA, freq={oscillation_frequency}Hz, STDP={"On" if use_stdp else "Off"}, '
            f'p_within={p_within_str}_Time{current_time}ms_Bin_{time_window_size}ms_{measure_name}',
            fontsize=14
        )
        fig_3d.colorbar(surf, shrink=0.5, aspect=5)

        # Сохраняем рисунок
        fig_filename_3d = os.path.join(
            directory_path,
            f'3D_plot_I0_{I0_value}pA_freq_{oscillation_frequency}Hz_'
            f'STDP_{"On" if use_stdp else "Off"}_p_within_{p_within_str}_'
            f'Time_{current_time}ms_Bin_{time_window_size}ms_{measure_name}.png'
        )
        plt.savefig(fig_filename_3d)
        plt.close(fig_3d)
        # print(f"3D-график сохранён: {fig_filename_3d}")


def plot_pinput_between_avg_spikes_with_std(
    spike_counts_second_cluster_for_input,
    spike_counts_second_cluster,
    I0_value,
    oscillation_frequency,
    use_stdp,
    current_time,
    time_window_size,
    directory_path='results_ext_test1',
    measure_name=None
):
    """
    Строит 3D-график (p_input, p_between, avg_spikes) для каждого p_within,
    дополнительно отображая коридор (mean ± std) и сохраняет данные в CSV.
    
    spike_counts_second_cluster_for_input:
      spike_counts_second_cluster_for_input[I0_value][p_within][p_input][p_between] = mean_spikes
    spike_counts_second_cluster:
      spike_counts_second_cluster[I0_value][p_within][p_between] = [array([...]), ...],
      где список массивов содержит данные для всех повторов, а число повторов определяется как
      n_rep = (len(list_of_arrays) / число уникальных p_input).
      
    В данной реализации стандартное отклонение вычисляется для того повтора, из которого получено среднее,
    то есть для каждого p_input выбирается массив с индексом:
      index_to_use = (p_input_idx + 1) * n_rep - 1.
    """
    import os, csv
    import numpy as np
    import matplotlib.pyplot as plt

    # Вывод исходных данных
    print("test_avg_spikes", spike_counts_second_cluster)
    print("avg_tests_avg_spikes", spike_counts_second_cluster_for_input)
    
    os.makedirs(directory_path, exist_ok=True)
    
    # Сохранение данных в CSV
    avg_csv_filename = os.path.join(
        directory_path,
        f"avg_tests_avg_spikes_{measure_name if measure_name else 'default'}.csv"
    )
    with open(avg_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['I0_value', 'p_within', 'p_input', 'p_between', 'mean_spikes', 'std_spikes'])
        
        data_for_input = spike_counts_second_cluster_for_input.get(I0_value, {})
        for p_within, dict_pinput in data_for_input.items():
            # Ключи p_input заданы как строки (например, "0.10", "0.15", "0.20")
            sorted_p_inputs = sorted(dict_pinput.keys(), key=float)
            num_pinputs = len(sorted_p_inputs)
            for p_input_idx, p_input_key in enumerate(sorted_p_inputs):
                inner_dict = dict_pinput[p_input_key]
                for p_between_key in sorted(inner_dict.keys(), key=float):
                    mean_spikes = inner_dict[p_between_key]
                    stdev_value = 0.0
                    p_btw = float(p_between_key)
                    # Поиск соответствующих «сырых» данных
                    if (I0_value in spike_counts_second_cluster and 
                        p_within in spike_counts_second_cluster[I0_value]):
                        # Ищем ключ, сравнивая как float
                        for key in spike_counts_second_cluster[I0_value][p_within]:
                            if np.isclose(float(key), p_btw):
                                list_of_arrays = spike_counts_second_cluster[I0_value][p_within][key]
                                break
                        else:
                            list_of_arrays = []
                        
                        if list_of_arrays and len(list_of_arrays) >= num_pinputs:
                            n_rep = len(list_of_arrays) // num_pinputs
                            # Выбираем последний повтор для данного p_input
                            index_to_use = (p_input_idx + 1) * n_rep - 1
                            try:
                                trial_array = np.array(list_of_arrays[index_to_use], dtype=float)
                                stdev_value = np.std(trial_array)
                            except Exception as e:
                                print(f"Ошибка при вычислении std для I0={I0_value}, p_within={p_within}, "
                                      f"p_input={p_input_key}, p_between={p_btw}: {e}")
                    writer.writerow([I0_value, p_within, p_input_key, p_btw, mean_spikes, stdev_value])
    
    # Построение 3D-графика для каждого p_within
    for p_within_str, dict_pinput in spike_counts_second_cluster_for_input.get(I0_value, {}).items():
        if not dict_pinput:
            print(f"Нет данных для p_within={p_within_str}. Пропуск.")
            continue

        sorted_p_inputs = sorted(dict_pinput.keys(), key=float)
        try:
            p_input_list_float = [float(p) for p in sorted_p_inputs]
        except ValueError as e:
            print(f"Ошибка при преобразовании p_input к float для p_within={p_within_str}: {e}")
            continue

        first_p_input = sorted_p_inputs[0]
        try:
            p_between_list = sorted([float(p) for p in dict_pinput[first_p_input].keys()])
        except ValueError as e:
            print(f"Ошибка при преобразовании p_between к float для p_within={p_within_str}: {e}")
            continue

        if not p_between_list:
            print(f"Нет p_between значений для p_within={p_within_str}. Пропуск.")
            continue

        num_pinputs = len(p_input_list_float)
        Z = np.zeros((num_pinputs, len(p_between_list)))
        Z_std = np.zeros_like(Z)
        
        for i, p_inp in enumerate(p_input_list_float):
            p_inp_str = f"{p_inp:.2f}"
            for j, p_btw in enumerate(p_between_list):
                mean_spikes = dict_pinput[p_inp_str].get(p_btw, 0.0)
                try:
                    Z[i, j] = float(mean_spikes)
                except (ValueError, TypeError):
                    Z[i, j] = 0.0

                stdev_value = 0.0
                if (I0_value in spike_counts_second_cluster and 
                    p_within_str in spike_counts_second_cluster[I0_value]):
                    raw_keys = spike_counts_second_cluster[I0_value][p_within_str].keys()
                    found = False
                    for key in raw_keys:
                        if np.isclose(float(key), p_btw):
                            list_of_arrays = spike_counts_second_cluster[I0_value][p_within_str][key]
                            found = True
                            break
                    if found and list_of_arrays and len(list_of_arrays) >= num_pinputs:
                        n_rep = len(list_of_arrays) // num_pinputs
                        index_to_use = (i + 1) * n_rep - 1
                        try:
                            trial_array = np.array(list_of_arrays[index_to_use], dtype=float)
                            stdev_value = np.std(trial_array)
                        except Exception as e:
                            print(f"Ошибка при вычислении std для I0={I0_value}, p_within={p_within_str}, "
                                  f"p_input={p_inp_str}, p_between={p_btw}: {e}")
                Z_std[i, j] = stdev_value

        p_input_mesh, p_between_mesh = np.meshgrid(p_input_list_float, p_between_list, indexing='ij')
        if not np.isfinite(Z).all():
            print(f"Некоторые значения в Z не являются конечными для p_within={p_within_str}. Пропуск построения графика.")
            continue

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            p_input_mesh,
            p_between_mesh,
            Z,
            cmap='viridis',
            edgecolor='none',
            alpha=0.9
        )
        # Добавляем коридор mean ± std
        Z_plus = Z + Z_std
        Z_minus = Z - Z_std
        for z_val in [Z_plus, Z_minus]:
            ax.plot_surface(
                p_input_mesh,
                p_between_mesh,
                z_val,
                color='#404040',
                alpha=0.2,
                edgecolor='none',
                antialiased=True
            )
        ax.set_title(
            f'3D: p_input vs p_between vs avg_spikes per bin\n'
            f'I0={I0_value}pA, freq={oscillation_frequency}Hz, STDP={"On" if use_stdp else "Off"}, '
            f'p_within={p_within_str}, Time={current_time}ms, Bin={time_window_size}ms, {measure_name}',
            fontsize=14
        )
        ax.set_zlim(0, 1)
        ax.set_xlabel('p_input', fontsize=12)
        ax.set_ylabel('p_between', fontsize=12)
        ax.set_zlabel('avg_spikes per bin', fontsize=12)
        fig.colorbar(surf, shrink=0.5, aspect=5, label="Mean spikes")
        
        filename = os.path.join(
            directory_path,
            f'3D_pinput_between_spikes_STD_V0_{I0_value}_freq_{oscillation_frequency}Hz_'
            f'STDP_{"On" if use_stdp else "Off"}_p_within_{p_within_str}_'
            f'Time_{current_time}ms_Bin_{time_window_size}ms_{measure_name}.png'
        )
        plt.savefig(filename, dpi=150)
        plt.close(fig)




import random

def measure_connectivity(C, cluster_sizes):
    """
    Функция для вычисления фактических долей внутрикластерных (p_in_measured)
    и межкластерных (p_out_measured) связей по итоговой матрице C.
    """
    n_neurons = C.shape[0]
    labels = np.empty(n_neurons, dtype=int)
    start = 0
    for idx, size in enumerate(cluster_sizes):
        labels[start:start + size] = idx
        start += size

    intra_possible = 0
    intra_actual = 0
    inter_possible = 0
    inter_actual = 0

    # Перебор пар нейронов (i < j)
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if labels[i] == labels[j]:
                intra_possible += 1
                if C[i, j]:
                    intra_actual += 1
            else:
                inter_possible += 1
                if C[i, j]:
                    inter_actual += 1

    p_in_measured = intra_actual / intra_possible if intra_possible > 0 else 0
    p_out_measured = inter_actual / inter_possible if inter_possible > 0 else 0
    return p_in_measured, p_out_measured


def sim(p_within, p_between, g, nu_ext_over_nu_thr, J, J2, refractory_period, sim_time, plotting_flags,
        rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes,
        I0_value, oscillation_frequency, use_stdp, time_window_size,
        C_total_prev=None, p_within_prev=None, p_between_prev=None,
        p_input=None, measure_name=None, measure_name_prev=None, centrality=None):
    """
    Выполняет один прогон симуляции при заданных параметрах.
    Если C_total_prev=None, генерируем матрицу "с нуля",
    иначе используем прошлую (чтобы копить STDP).
    """

    start_scope()

    start_time = time.time()

    if plotting_flags is None:
        plotting_flags = {}

    n_neurons = len(cluster_labels)
    target_cluster_index = 0
    proportion_high_centrality = 0.1
    centrality_type = measure_name
    boost_factor = 2

    C_total = generate_sbm_with_high_centrality(
        n_neurons, cluster_sizes, p_within, p_between,
        target_cluster_index, proportion_high_centrality,
        centrality_type, boost_factor
    )       

    # -- Параметры LIF --
    N = n_neurons
    N1 = int(n_neurons/2)
    N2 = n_neurons
    N_E = int(N * 80 / 100)
    N_I = N - N_E
    # print("N", N)
    # print("N_E", N_E)
    # print("N_I", N_I)
    # N_E = 800
    R = 80 * Mohm
    C = 0.25 * nfarad
    tau = R*C # 20 ms
    v_threshold = -50 * mV
    v_reset = -70 * mV
    v_rest = -65 * mV
    J = J * mV
    J2 = J2 * mV
    D = 1.5 * ms
    defaultclock.dt = 0.1 * ms

    # -- Создаём группу нейронов --
    neurons = NeuronGroup(
        N,
        '''
        dv/dt = (v_rest - v + R*I_ext)/tau : volt (unless refractory)
        I_ext = I0 * sin(2 * pi * f * t + phi) : amp
        I0 : amp
        f : Hz
        phi : 1
        is_hub : integer (constant)
        ''',
        threshold="v > v_threshold",
        reset="v = v_reset",
        method="euler",
    )

    # Шаг 1. Вычисляем метрику betweenness_centrality и получаем список лучших узлов
    if p_input is None:
        p_input = 1.0

    if isinstance(C_total, np.ndarray):
        C_total_matrix = C_total
        C_total = nx.from_numpy_array(C_total)
    elif isinstance(C_total, nx.Graph):
        C_total = C_total
        C_total_matrix = nx.to_numpy_array(C_total)
        p_in_measured, p_out_measured = measure_connectivity(C_total_matrix, cluster_sizes)
        print(f"Фактическая p_in (доля связей внутри кластера): {p_in_measured:.3f}")
        print(f"Фактическая p_out (доля связей между кластерами): {p_out_measured:.3f}")
    else:
        raise ValueError("data должен быть либо numpy-массивом, либо объектом networkx.Graph")

    centrality = print_centrality(C_total_matrix, n_neurons, p_input, measure_name=measure_name)

    print(measure_name)
    print(len(centrality))
    if isinstance(centrality, np.ndarray):  # проверяем, является ли объект np.ndarray
        centrality = centrality.tolist()
    centrality = sorted(centrality)

    # centrality = filter(lambda x: x < n_cluster_neurons, centrality)
    # centrality = list(centrality)
    # print(centrality)
    # Выбираем int(p_input * 50) нейронов из 0..49, даём им синус
    # Шаг 5. Назначаем нужную модуляцию (частоту, ток, фазу) выбранным нейронам
    if centrality:
        neurons.f[centrality] = oscillation_frequency * Hz
        neurons.I0[centrality] = I0_value * pA
        neurons.phi[centrality] = 0
    else:
        print("Список выбранных нейронов пуст — пропускаем назначение модуляции.")

    input_rate = 30 * Hz
    input_group = PoissonGroup(n_neurons, rates=input_rate)
    syn_input = Synapses(input_group, neurons, on_pre='v_post += J', delay=D)
    syn_input.connect(condition='i >= 0 and j < N1', p=p_input)


    # external_poisson_input = PoissonInput(
    #     target=neurons, target_var="v", N=p_input*n_cluster_neurons, rate=10*Hz, weight=J
    # )
    # STDP или нет
    if use_stdp:
        tau_stdp = 20 * ms
        Aplus = 0.005
        Aminus = 0.005

        stdp_eqs = '''
        dApre/dt = -Apre / tau_stdp : 1 (event-driven)
        dApost/dt = -Apost / tau_stdp : 1 (event-driven)
        w : 1
        '''
        exc_synapses = Synapses(
            neurons, neurons,
            model=stdp_eqs,
            on_pre="""
                v_post += J * w
                Apre += Aplus
                w = clip(w + Apost, 0, 1)
            """,
            on_post="""
                Apost += Aminus
                w = clip(w + Apre, 0, 1)
            """,
            delay=D
        )
    else:
        exc_synapses = Synapses(neurons, neurons,
                                model="w : 1",
                                on_pre="v_post += J2 * w",
                                delay=D)
        
    inh_synapses = Synapses(neurons, neurons, 
                            model="w : 1", 
                            on_pre="v_post += -g * J * w", 
                            delay=D)


    # sources, targets = np.where(C_total > 0)
    # exc_mask = (sources < N_E)  # Возбуждающие нейроны
    # inh_mask = (sources >= N_E) # Тормозные нейроны
    # exc_synapses.connect(i=sources[exc_mask], j=targets[exc_mask])
    # inh_synapses.connect(i=sources[inh_mask], j=targets[inh_mask])
    # exc_synapses.w = C_total[sources[exc_mask], targets[exc_mask]]
    # inh_synapses.w = C_total[sources[inh_mask], targets[inh_mask]]

    # # Генерация источников и целей
    N_E_2 = int(N_E / 2)
    N_I_2 = int(N_I / 2)
    N_2 = int(N / 2)
    rows = np.arange(0, N_E_2) 
    rows2 = np.arange(N_E_2, N_E_2 + N_I_2) 
    rows3 = np.arange(N_2, N_E_2 + N_2)
    rows4 = np.arange(N_E_2 + N_2, N)  
    mask = np.isin(np.arange(C_total_matrix.shape[0]), rows)  
    mask2 = np.isin(np.arange(C_total_matrix.shape[0]), rows2)  
    mask3 = np.isin(np.arange(C_total_matrix.shape[0]), rows3) 
    mask4 = np.isin(np.arange(C_total_matrix.shape[0]), rows4)  
    sources_exc1, targets_exc1 = np.where(C_total_matrix[mask, :] > 0)
    sources_inh1, targets_inh1 = np.where(C_total_matrix[mask2, :] > 0)
    sources_exc2, targets_exc2 = np.where(C_total_matrix[mask3, :] > 0)
    sources_inh2, targets_inh2 = np.where(C_total_matrix[mask4, :] > 0)
    sources_exc1 = rows[sources_exc1]  # Преобразует 0-4 в 20-24
    sources_inh1 = rows2[sources_inh1]  # Преобразует 0-4 в 20-24
    sources_exc2 = rows3[sources_exc2]  # Преобразует 0-4 в 20-24
    sources_inh2 = rows4[sources_inh2]  # Преобразует 0-4 в 20-24
    # print("sources_exc1", sources_exc1)
    # print("targets_exc1", targets_exc1)
    # print("sources_inh1", sources_inh1)
    # print("targets_inh1", targets_inh1)
    # print("sources_exc2", sources_exc2)
    # print("targets_exc2", targets_exc2)
    # print("sources_inh2", sources_inh2)
    # print("targets_inh2", targets_inh2)
    sources_exc = np.concatenate((sources_exc1, sources_exc2))
    targets_exc = np.concatenate((targets_exc1, targets_exc2))
    sources_inh = np.concatenate((sources_inh1, sources_inh2))
    targets_inh = np.concatenate((targets_inh1, targets_inh2))
    exc_synapses.connect(i=sources_exc, j=targets_exc)
    inh_synapses.connect(i=sources_inh, j=targets_inh)
    exc_synapses.w = 1
    inh_synapses.w = 1


    # Мониторы  
    spike_monitor = SpikeMonitor(neurons)

    rate_monitor = None
    rate_monitor2 = None
    if plotting_flags.get('rates', False) or plotting_flags.get('psd', False) or plotting_flags.get('spectrogram', False):
        rate_monitor = PopulationRateMonitor(neurons[:int(n_neurons/2)])
    if plotting_flags.get('rates2', False) or plotting_flags.get('psd2', False) or plotting_flags.get('spectrogram', False):
        rate_monitor2 = PopulationRateMonitor(neurons[int(n_neurons/2):])

    trace = None
    if plotting_flags.get('trace', False):
        trace = StateMonitor(neurons, 'v', record=True)

    # print(f"Количество нейронов: {N}")
    # print(f"Количество возбуждающих синапсов: {exc_synapses.N}")
    # print(f"Количество тормозных синапсов: {inh_synapses.N}")

    # Запуск
    run(sim_time, profile=True)

    # Профилирование
    # print(profiling_summary(show=5))
    end_time = time.time()
    duration = end_time - start_time
    # print(f"Testing completed in {duration:.2f} seconds.")

    # Анализ спайков
    spike_times = spike_monitor.t / ms
    spike_indices = spike_monitor.i

    bins = np.arange(0, int(sim_time/ms) + 1, time_window_size)
    time_window_centers = (bins[:-1] + bins[1:]) / 2

    avg_neuron_spikes_cluster2_list = []
    start_cluster_neuron = n_neurons/2
    end_cluster_neuron = n_neurons

    for i in range(len(bins) - 1):
        start_t = bins[i]
        end_t = bins[i + 1]

        mask = (
            (spike_indices >= start_cluster_neuron) & (spike_indices < end_cluster_neuron) &
            (spike_times > start_t) & (spike_times < end_t)
        )
        filtered_spike_indices = spike_indices[mask]

        group2_spikes = len(filtered_spike_indices)
        avg_spikes = group2_spikes / (end_cluster_neuron - start_cluster_neuron)
        avg_neuron_spikes_cluster2_list.append(avg_spikes)

    # Убираем первое окно, если нужно
    avg_neuron_spikes_cluster2_list = avg_neuron_spikes_cluster2_list[1:]

    # Построение графиков
    if plotting_flags.get('spikes', False):
        ax_spikes = plotting_flags['ax_spikes']
        plot_spikes(ax_spikes, spike_times, spike_indices, time_window_size, t_range, sim_time,
                    oscillation_frequency, use_stdp, measure_name)

    if plotting_flags.get('rates', False):
        ax_rates = plotting_flags['ax_rates']
        plot_rates(ax_rates, N1, N2, rate_monitor, t_range)

    if plotting_flags.get('rates2', False):
        ax_rates2 = plotting_flags['ax_rates2']
        plot_rates2(ax_rates2, N1, N2, rate_monitor2, t_range)


    if plotting_flags.get('psd', False):
        ax_psd = plotting_flags['ax_psd']
        plot_psd(rate_monitor, N1, N2, ax_psd)

    if plotting_flags.get('psd2', False):
        ax_psd2 = plotting_flags['ax_psd2']
        plot_psd2(rate_monitor2, N1, N2, ax_psd2)

    if plotting_flags.get('spectrogram', False):
        ax_spectrogram = plotting_flags['ax_spectrogram']
        plot_spectrogram(rate_monitor, rate_monitor2, oscillation_frequency, use_stdp, ax_spectrogram)

    if plotting_flags.get('connectivity2', False):
        connectivity2 = plotting_flags['connectivity2']
        plot_connectivity(n_neurons, exc_synapses, inh_synapses, connectivity2, centrality, p_in_measured, p_out_measured)

    return avg_neuron_spikes_cluster2_list, time_window_centers, C_total, spike_indices, centrality, p_in_measured, p_out_measured

# Флаги для построения графиков
do_plot_spikes = True
do_plot_avg_spikes = False
do_plot_rates = True
do_plot_rates2 = True
do_plot_connectivity2 = True
do_plot_psd = True
do_plot_psd2 = True
do_plot_spectrogram = False

# -- Основной цикл по времени симуляции --
for current_time in simulation_times:
    sim_time = current_time * ms
    t_range = [0, current_time]

    # Структуры для «старых» диаграмм
    spike_counts_second_cluster = {}
    detailed_spike_data_for_3d = {}
    # Структура для новой зависимости p_input vs p_between vs avg_spikes
    spike_counts_second_cluster_for_input = {}

    # Циклы по частоте, V0, STDP
    for oscillation_frequency in oscillation_frequencies:
        for I0_value in I0_values:
            for use_stdp in use_stdp_values:
                print(f"\nRunning simulations for I0 = {I0_value} pA, Frequency = {oscillation_frequency} Hz, "
                      f"STDP = {'On' if use_stdp else 'Off'}, Time={current_time} ms")

                # Инициализация словарей с строковыми ключами
                spike_counts_second_cluster[I0_value] = {
                    f"{p_within:.2f}": {} for p_within in p_within_values
                }
                detailed_spike_data_for_3d[I0_value] = {
                    f"{p_within:.2f}": {} for p_within in p_within_values
                }
                spike_counts_second_cluster_for_input[I0_value] = {
                    f"{p_within:.2f}": {} for p_within in p_within_values
                }

                # -- Цикл по p_within --
                for p_within in p_within_values:
                    p_within = round(p_within, 3)
                    p_within_str = f"{p_within:.2f}"
                    print(f"\n--- p_within = {p_within_str} ---")

                    # Для нового p_within — подготовим ячейки в spike_counts_second_cluster_for_input
                    for p_input in p_input_values:
                        p_input = round(p_input, 2)
                        p_input_str = f"{p_input:.2f}"
                        measure_name_prev = None
                        C_total_prev = None
                        for measure_name in measure_names:
                            centrality = None
                            p_within_prev = None
                            p_between_prev = None
                            spike_counts_second_cluster_for_input[I0_value][p_within_str][p_input_str] = {}
                            # Список для хранения кадров (GIF), если нужно
                            images = []
                            p_between_values = np.arange(0.05, p_within-0.01, 0.05)
                            # -- Цикл по p_between --
                            for p_between in p_between_values:
                                p_between = round(p_between, 3)

                                print(f"\nProcessing p_between = {p_between}")

                                # При необходимости готовим фигуру для GIF
                                if any([
                                    plot_spikes, plot_rates, plot_rates2, plot_psd, plot_psd2, plot_spectrogram
                                ]):
                                    fig = plt.figure(figsize=(12, 10))
                                    fig.suptitle(
                                        f'I0={I0_value} pA, p_input={p_input:.2f},'
                                        f'p_within={p_within_str}, p_between={p_between}, '
                                        f'g={g}, epsilon={epsilon}, nu_ext_over_nu_thr={nu_ext_over_nu_thr}, '
                                        f'D={D}, Time={current_time} ms'
                                    )
                                    gs = fig.add_gridspec(ncols=4, nrows=3)
                                else:
                                    fig = None

                                plotting_flags = {
                                    'spikes': do_plot_spikes,
                                    'avg_spikes': do_plot_avg_spikes,
                                    'rates': do_plot_rates,
                                    'rates2': do_plot_rates2,
                                    'connectivity2': do_plot_connectivity2,
                                    'psd': do_plot_psd,
                                    'psd2': do_plot_psd2,
                                    'spectrogram': do_plot_spectrogram,
                                }
                                if fig is not None:
                                    if plot_spikes:
                                        plotting_flags['ax_spikes'] = fig.add_subplot(gs[0, :])
                                    if plot_rates:
                                        plotting_flags['ax_rates'] = fig.add_subplot(gs[1, 1])
                                    if plot_rates2:
                                        plotting_flags['ax_rates2'] = fig.add_subplot(gs[1, 2])
                                    if plot_connectivity:
                                        plotting_flags['connectivity2'] = fig.add_subplot(gs[1, 3])
                                    if plot_psd:
                                        plotting_flags['ax_psd'] = fig.add_subplot(gs[2, 1])
                                    if plot_psd2:
                                        plotting_flags['ax_psd2'] = fig.add_subplot(gs[2, 2])
                                    if plot_spectrogram:
                                        plotting_flags['ax_spectrogram'] = fig.add_subplot(gs[2, 3])
                                avg_window_avg_neuron_spikes_cluster2_tests = []
                                # -- Запускаем несколько тестов (num_tests) --
                                for test_num in range(num_tests):
                                    print(f'\nI0={I0_value} pA, p_within={p_within_str}, '
                                        f'p_input={p_input:.2f}, p_between={p_between}, '
                                        f'tест={test_num + 1}, Time={current_time} ms')

                                    # Очищаем оси на повторных прогонах
                                    if test_num > 0 and fig is not None:
                                        if plot_spikes:
                                            plotting_flags['ax_spikes'].clear()
                                        if plot_rates:
                                            plotting_flags['ax_rates'].clear()
                                        if plot_rates2:
                                            plotting_flags['ax_rates2'].clear()
                                        if plot_psd:
                                            plotting_flags['ax_psd'].clear()
                                        if plot_psd2:
                                            plotting_flags['ax_psd2'].clear()
                                        if plot_spectrogram:
                                            plotting_flags['ax_spectrogram'].clear()
                                        if plot_connectivity:
                                            plotting_flags['connectivity2'].clear()

                                    # -- Вызов функции симуляции --
                                    (avg_neuron_spikes_cluster2_list,
                                    time_window_centers,
                                    C_total,
                                    spike_indices, centrality, p_in_measured, p_out_measured) = sim(
                                        p_within,
                                        p_between,
                                        g,
                                        nu_ext_over_nu_thr,
                                        J,
                                        J2,
                                        refractory_period,
                                        sim_time,
                                        plotting_flags,
                                        rate_tick_step,
                                        t_range,
                                        rate_range,
                                        cluster_labels,
                                        cluster_sizes,
                                        I0_value,
                                        oscillation_frequency,
                                        use_stdp,
                                        time_window_size,
                                        C_total_prev,
                                        p_within_prev,
                                        p_between_prev,
                                        p_input=p_input,
                                        measure_name=measure_name,
                                        measure_name_prev = measure_name_prev,
                                        centrality=centrality
                                    )

                                    # Считаем среднее по окнам для 2-го кластера
                                    if avg_neuron_spikes_cluster2_list is not None and len(avg_neuron_spikes_cluster2_list) > 0:
                                        avg_window_val = np.mean(avg_neuron_spikes_cluster2_list)
                                    else:
                                        avg_window_val = 0
                                    avg_window_avg_neuron_spikes_cluster2_tests.append(avg_window_val)

                                    # print("avg_window_avg_neuron_spikes_cluster2_tests", avg_window_avg_neuron_spikes_cluster2_tests)

                                    # Если данных по времени нет, проверим и присвоим пустой массив, чтобы избежать None
                                    if time_window_centers is None:
                                        time_window_centers = np.array([])

                                    if test_num == 0:
                                        spikes_for_3d_time = avg_neuron_spikes_cluster2_list if avg_neuron_spikes_cluster2_list is not None else []
                                        time_for_3d_time = time_window_centers if time_window_centers is not None else np.array([])

                                    # -- Обновляем "prev"-переменные для STDP в рамках одной комбинации (p_within, p_input) --
                                    # **Важно:** Обновляем после сравнения
                                    C_total_prev = C_total.copy()
                                    p_within_prev = p_within
                                    p_between_prev = p_between
                                    centrality_prev = centrality
                                    measure_name_prev = measure_name

                                detailed_spike_data_for_3d[I0_value][p_within_str][p_between] = {
                                    "time": time_for_3d_time if time_for_3d_time is not None else np.array([]),
                                    "spikes_list": spikes_for_3d_time if spikes_for_3d_time is not None else []
                                }
                                avg_window_avg_neuron_spikes_cluster2_tests = np.array(avg_window_avg_neuron_spikes_cluster2_tests)
                                mean_spikes = np.mean(avg_window_avg_neuron_spikes_cluster2_tests)

                                # (1) Сохраняем в старую структуру (если нужно)
                                if p_between not in spike_counts_second_cluster[I0_value][p_within_str]:
                                    spike_counts_second_cluster[I0_value][p_within_str][p_between] = []
                                spike_counts_second_cluster[I0_value][p_within_str][p_between].append(
                                    avg_window_avg_neuron_spikes_cluster2_tests
                                )

                                # (2) Записываем в новую структуру (p_input vs p_between)
                                spike_counts_second_cluster_for_input[I0_value][p_within_str][p_input_str][p_between] = float(mean_spikes)
                                # print('spike_counts_second_cluster_for_input',spike_counts_second_cluster_for_input)

                                # Сохраняем кадр в GIF
                                if fig is not None:
                                    plt.tight_layout()
                                    buf = io.BytesIO()
                                    plt.savefig(buf, format='png')
                                    buf.seek(0)
                                    images.append(imageio.imread(buf))
                                    plt.close(fig)

                            # Конец цикла по p_between
                            if images:
                                gif_filename = (
                                    f'{directory_path}/gif_exc_I0_{I0_value}freq_{oscillation_frequency}_'
                                    f'STDP_{"On" if use_stdp else "Off"}_p_within_{p_within_str}_'
                                    f'p_input_{p_input_str}_Time_{current_time}ms_Bin_{time_window_size}ms_{measure_name}.gif'
                                )
                                imageio.mimsave(gif_filename, images, duration=2000, loop=0)  # duration в секундах

                            # Конец циклов по p_within и p_input

                            # print(f"\nСимуляции для I0 = {I0_value} pA, Time={current_time} ms завершены.")

                            # -- Строим «старые» графики --
                            plot_3d_spike_data(
                                detailed_spike_data_for_3d,
                                p_within_values,
                                I0_value,
                                oscillation_frequency,
                                use_stdp,
                                current_time,
                                time_window_size,
                                directory_path=directory_path,
                                measure_name=measure_name
                            )

                            plot_pinput_between_avg_spikes_with_std(
                                spike_counts_second_cluster_for_input,
                                spike_counts_second_cluster,
                                I0_value,
                                oscillation_frequency,
                                use_stdp,
                                current_time,
                                time_window_size,
                                directory_path=directory_path,
                                measure_name=measure_name
                            )
                            plt.show()
# Конец основного цикла по времени симуляции

print("\nГенерация графиков завершена.")
