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
from scipy.signal import resample

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
p_within_values = np.arange(0.2, 0.21, 0.05)  # [0.7, 0.8, 0.9, 1.0]
p_input_values = np.arange(0.1, 0.21, 0.05)

num_tests = 10
J = 0.8
J2 = 0.8
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
    "random",
    "betweenness",
    "eigenvector",
    "pagerank",
    "percolation",
]


# -- Время симуляции (в мс) --
simulation_times = [5000]


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
        self.adjacency_matrix = adjacency_matrix
        self.graph = nx.from_numpy_array(adjacency_matrix)

    def calculate_betweenness_centrality(self):
        return nx.betweenness_centrality(self.graph)

    def calculate_eigenvector_centrality(self):
        return nx.eigenvector_centrality(self.graph)

    def calculate_pagerank_centrality(self, alpha=0.85):
        return nx.pagerank(self.graph, alpha=alpha)

    def calculate_flow_coefficient(self):
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
        return nx.degree_centrality(self.graph)

    def calculate_closeness_centrality(self):
        return nx.closeness_centrality(self.graph)

    def calculate_harmonic_centrality(self):
        return nx.harmonic_centrality(self.graph)

    def calculate_percolation_centrality(self, attribute=None):
        if attribute is None:
            attribute = {node: 1 for node in self.graph.nodes()}
        return nx.percolation_centrality(self.graph, states=attribute)

    def calculate_cross_clique_centrality(self):
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

    ax_spikes.set_xlim([0,2000])
    ax_spikes.set_xlabel("t [ms]")
    ax_spikes.set_ylabel("Neuron index")
    ax_spikes.set_title(f"Spike Raster Plot\nFrequency={oscillation_frequency} Hz, STDP={'On' if use_stdp else 'Off'}, Window={time_window_size} ms, {measure_name}", fontsize=16)
def plot_rates(ax_rates, N1, N2, rate_monitor, t_range):
    ax_rates.set_title(f'Num. of spikes\n neurons\n 0-{N1}', fontsize=16)
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz, label=f'Group 1 (0-{n_neurons/2})')
    ax_rates.set_xlim(t_range)
    ax_rates.set_ylim([0,500])
    ax_rates.set_xlabel("t [ms]")


def plot_rates2(ax_rates2, N1, N2, rate_monitor2, t_range):
    ax_rates2.set_title(f'Num. of spikes\n neurons\n {N1}-{N2}', fontsize=16)
    ax_rates2.plot(rate_monitor2.t / ms, rate_monitor2.rate / Hz, label=f'Group 2 ({n_neurons/2}-{n_neurons})')
    ax_rates2.set_xlim(t_range)
    ax_rates2.set_ylim([0,500])
    ax_rates2.set_xlabel("t [ms]")    

def plot_psd(rate_monitor, N1, N2, ax_psd):
    rate = rate_monitor.rate / Hz - np.mean(rate_monitor.rate / Hz)
    from numpy.fft import rfft, rfftfreq
    N = len(rate_monitor.t)
    if N > 0:
        # Определение частоты дискретизации на основе временного шага
        dt = float((rate_monitor.t[1] - rate_monitor.t[0]) / ms)
        sampling_rate = 1000 / dt
        window = np.hanning(N)
        rate_windowed = rate * window
        # Ограничение до нужного диапазона частот
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate_windowed)) / N 
        yn = yn[:max_point]
    
        ax_psd.set_title(f"PSD neurons\n 0-{N1}", fontsize=16)
        ax_psd.plot(x, yn, c='k', label='Function')
        ax_psd.set_xlim([0,50])
        ax_psd.set_xlabel('Hz')


def plot_psd2(rate_monitor2, N1, N2, ax_psd2):
    rate = rate_monitor2.rate / Hz - np.mean(rate_monitor2.rate / Hz)
    from numpy.fft import rfft, rfftfreq
    N = len(rate_monitor2.t)
    if N > 0:
        # Определение частоты дискретизации на основе временного шага
        dt = float((rate_monitor2.t[1] - rate_monitor2.t[0]) / ms) 
        sampling_rate = 1000 / dt 
        window = np.hanning(N)
        rate_windowed = rate * window
        # Ограничение до нужного диапазона частот
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate_windowed)) / N 
        yn = yn[:max_point]
    
        ax_psd2.set_title(f"PSD neurons\n {N1}-{N2}", fontsize=16)
        ax_psd2.plot(x, yn, c='k', label='Function')
        ax_psd2.set_xlim([0,50])
        ax_psd2.set_xlabel('Hz')

def plot_spectrogram(rate_monitor, rate_monitor2, N1, N2, ax_spectrogram):
    dt = float((rate_monitor.t[1] - rate_monitor.t[0]) / second)  # dt в секундах
    N_freq = len(rate_monitor.t)
    xf = rfftfreq(len(rate_monitor.t), d=dt)[:N_freq]
    ax_spectrogram.plot(xf, np.abs(rfft(rate_monitor.rate / Hz)), label=f'{0}-{N1} neurons')
    
    dt2 = float((rate_monitor2.t[1] - rate_monitor2.t[0]) / second)
    N_freq2 = len(rate_monitor2.t)
    xf2 = rfftfreq(len(rate_monitor2.t), d=dt2)[:N_freq2]
    ax_spectrogram.plot(xf2, np.abs(rfft(rate_monitor2.rate / Hz)), label=f'{N1}-{N2} neurons')
    
    ax_spectrogram.set_xlim(0, 50)
    ax_spectrogram.legend()
    ax_spectrogram.set_title(f'Global\n Frequencies', fontsize=16)


# def plot_connectivity(n_neurons, S_intra1, S_intra2, S_12, S_21, connectivity2, centrality, p_in_measured, p_out_measured, percent_central):
def plot_connectivity(n_neurons, exc_synapses, inh_synapses, ax_connectivity2, centrality, p_in_measured, p_out_measured, percent_central):
    W = np.zeros((n_neurons, n_neurons))
    W[exc_synapses.i[:], exc_synapses.j[:]] = exc_synapses.w[:]
    W[inh_synapses.i[:], inh_synapses.j[:]] = inh_synapses.w[:]
    # W[S_intra1.i[:], S_intra1.j[:]] = S_intra1.w[:]
    # W[S_intra2.i[:], S_intra2.j[:]] = S_intra2.w[:]
    # W[S_12.i[:], S_12.j[:]] = S_12.w[:]
    # W[S_21.i[:], S_21.j[:]] = S_21.w[:]

    # Highlight neurons with current applied based on centrality
    highlighted_neurons = np.unique(centrality)
    for neuron in highlighted_neurons:
        ax_connectivity2.plot([neuron], [neuron], 'ro', markersize=1, label=f'Neuron {neuron}')

    ax_connectivity2.set_title(f'Weight Matrix,\np_in_measured={p_in_measured:.3f},\n p_out_measured={p_out_measured:.3f},\n p_central={percent_central}', fontsize=14)
    ax_connectivity2.matshow(W, cmap='viridis')
    ax_connectivity2.set_xlabel('Post-synaptic neuron', fontsize=12)
    ax_connectivity2.set_ylabel('Pre-synaptic neuron', fontsize=12)

def print_centrality(C_total, N1, cluster_nodes, p_input, measure_name, direct_1_2=True):
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
        Доля от числа узлов кластера, которая берётся для формирования топ-листа.
        Если cluster_nodes - целое число, то используется именно это число для определения размера кластера.
    measure_name : str
        Название метрики центральности, которую нужно вычислить.
    direct_1_2 : bool, optional
        Если True, то анализ проводится для первой половины узлов кластера, иначе – для узлов с индексами 250-499.

    Возвращает:
    ----------
    list
        Список индексов узлов, являющихся топ-узлами по заданной метрике (только внутри выбранного кластера).
    """
    if measure_name != 'random':
        graph_centrality = GraphCentrality(C_total)
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

        # Определяем список узлов для анализа в зависимости от значения direct_1_2
        if not direct_1_2:
            # Если direct_1_2 == False, выбираем нейроны с индексами 250-499
            print(cluster_nodes)
            cluster_list = list(range(N1, cluster_nodes))
        else:
            if isinstance(cluster_nodes, int):
                cluster_list = list(range(cluster_nodes))
            else:
                cluster_list = list(cluster_nodes)
            # Выбираем первую половину узлов кластера (как в исходной логике)
            cluster_list = cluster_list[:int(len(cluster_list)/2)]
        
        # Формируем словарь значений метрики для выбранных узлов
        measure_values_cluster = {
            node: measure_values[node] for node in cluster_list if node in measure_values
        }
        # Определяем количество топ-узлов для выборки
        top_k = int(p_input * len(cluster_list))
        sorted_neurons_cluster = sorted(
            measure_values_cluster,
            key=lambda n: measure_values_cluster[n],
            reverse=True
        )
        top_neurons = sorted_neurons_cluster[:top_k]
        # print(measure_values_cluster)
        # print(sorted(top_neurons))
        # plot_centrality_by_neuron_number(measure_values_cluster, top_neurons, top_percent=p_input*100)
    else:
        # Обработка случайной выборки
        if not direct_1_2:
            cluster_indices = np.arange(N1, cluster_nodes)
        else:
            # Здесь предполагается, что общее число нейронов задано переменной n_neurons,
            # а для direct_1_2 == True выбираются нейроны первой половины
            cluster_indices = np.arange(0, int(n_neurons/2))
        num_chosen = int(p_input * len(cluster_indices))
        top_neurons = np.random.choice(cluster_indices, size=num_chosen, replace=False)

    return top_neurons


def plot_centrality_by_neuron_number(measure_values_cluster, top_neurons, top_percent=10):
    """
    Строит график зависимости значения метрики центральности от номера нейрона.
    
    Параметры:
    -----------
    measure_values_cluster : dict
         Словарь, где ключ – номер нейрона, а значение – метрика центральности.
    top_neurons : list
         Список номеров нейронов, отобранных как топ по заданной метрике.
    top_percent : float, optional
         Процент для выделения порогового значения (по умолчанию 10%).
    """
    # Сортируем нейроны по их номерам для корректного отображения по оси x
    neurons = sorted(measure_values_cluster.keys())
    values = [measure_values_cluster[n] for n in neurons]
    
    plt.figure(figsize=(12, 6))
    # Линейный график, показывающий зависимость значения метрики от номера нейрона
    plt.plot(neurons, values, 'bo-', label='Значение метрики')
    plt.xlabel('Номер нейрона')
    plt.ylabel('Значение метрики центральности')
    plt.title('Зависимость значения метрики центральности от номера нейрона')
    plt.grid(True)
    
    # Определим пороговое значение метрики для топ-нейронов.
    # Для этого сначала сортируем значения метрики по возрастанию.
    sorted_values = sorted(values)
    n = len(sorted_values)
    threshold_index = int(np.ceil((1 - top_percent/100) * n))
    threshold_value = sorted_values[threshold_index] if threshold_index < n else sorted_values[-1]
    
    # Отмечаем горизонтальной пунктирной линией пороговое значение метрики
    plt.axhline(threshold_value, color='green', linestyle='--', linewidth=2,
                label=f'Пороговая метрика (топ {top_percent}%) = {threshold_value:.3f}')
    
    # Выделяем на графике нейроны, входящие в топ (их номера берём из top_neurons)
    top_values = [measure_values_cluster[n] for n in top_neurons]
    plt.scatter(top_neurons, top_values, color='red', zorder=5, label='Топ-нейроны')
    
    plt.legend()
    plt.show()

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
        ax_3d.set_zlim(0,2)
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
    Функция строит 3D-график зависимости p_input, p_between и avg_spikes для каждого значения p_within.
    Для каждого узла сетки отображаются вертикальные error bar, соответствующие интервалу [среднее - std, среднее + std].
    Данные дополнительно сохраняются в CSV.
    
    spike_counts_second_cluster_for_input:
      spike_counts_second_cluster_for_input[I0_value][p_within][p_input][p_between] = mean_spikes
    spike_counts_second_cluster:
      spike_counts_second_cluster[I0_value][p_within][p_between] = [array([...]), ...],
      где список массивов содержит данные для всех повторов, а число повторов определяется как
      n_rep = (len(list_of_arrays) / число уникальных p_input).
      
    В данной реализации для каждого p_input выбирается массив с индексом:
      index_to_use = (p_input_idx + 1) * n_rep - 1,
    по которому вычисляется стандартное отклонение.
    """
    import os, csv
    import numpy as np
    import matplotlib.pyplot as plt

    # Создаем каталог для сохранения результатов
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
            sorted_p_inputs = sorted(dict_pinput.keys(), key=float)
            num_pinputs = len(sorted_p_inputs)
            for p_input_idx, p_input_key in enumerate(sorted_p_inputs):
                inner_dict = dict_pinput[p_input_key]
                for p_between_key in sorted(inner_dict.keys(), key=float):
                    mean_spikes = inner_dict[p_between_key]
                    stdev_value = 0.0
                    p_btw = float(p_between_key)
                    if (I0_value in spike_counts_second_cluster and 
                        p_within in spike_counts_second_cluster[I0_value]):
                        for key in spike_counts_second_cluster[I0_value][p_within]:
                            if np.isclose(float(key), p_btw):
                                list_of_arrays = spike_counts_second_cluster[I0_value][p_within][key]
                                break
                        else:
                            list_of_arrays = []
                        
                        if list_of_arrays and len(list_of_arrays) >= num_pinputs:
                            n_rep = len(list_of_arrays) // num_pinputs
                            index_to_use = (p_input_idx + 1) * n_rep - 1
                            try:
                                trial_array = np.array(list_of_arrays[index_to_use], dtype=float)
                                stdev_value = np.std(trial_array)
                            except Exception as e:
                                print(f"Ошибка при вычислении std для I0={I0_value}, p_within={p_within}, "
                                      f"p_input={p_input_key}, p_between={p_btw}: {e}")
                    writer.writerow([I0_value, p_within, p_input_key, p_btw, mean_spikes, stdev_value])
    
    # Построение 3D-графика для каждого значения p_within
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

        # Извлекаем список значений p_between из первого ключа p_input
        first_p_input = sorted_p_inputs[0]
        try:
            p_between_list = sorted([float(p) for p in dict_pinput[first_p_input].keys()])
        except ValueError as e:
            print(f"Ошибка при преобразовании p_between к float для p_within={p_within_str}: {e}")
            continue

        if not p_between_list:
            print(f"Нет p_between значений для p_within={p_within_str}. Пропуск.")
            continue

        # Если по одной из осей меньше двух уникальных значений, построение поверхности невозможно
        if len(p_input_list_float) < 2 or len(p_between_list) < 2:
            print(f"Недостаточно данных для построения 3D поверхности для p_within={p_within_str}.")
            continue

        num_pinputs = len(p_input_list_float)
        Z = np.zeros((num_pinputs, len(p_between_list)))
        Z_std = np.zeros_like(Z)
        
        # Формирование матриц средних значений и стандартных отклонений
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

        # Создаем сетку для значений p_input и p_between
        p_input_mesh, p_between_mesh = np.meshgrid(p_input_list_float, p_between_list, indexing='ij')
        if not np.isfinite(Z).all():
            print(f"Некоторые значения в Z не являются конечными для p_within={p_within_str}. Пропуск построения графика.")
            continue

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Установка прозрачности поверхности (alpha) для улучшения видимости error bars
        surf = ax.plot_surface(p_input_mesh, p_between_mesh, Z, cmap='jet', edgecolor='none', 
                               zorder=1, vmin=0, vmax=1, alpha=0.7)
        
        # Отрисовка error bars поверх поверхности
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                x_val = p_input_mesh[i, j]
                y_val = p_between_mesh[i, j]
                z_val = Z[i, j]
                err = Z_std[i, j]
                # Отрисовка вертикальной линии, представляющей интервал [z - err, z + err]
                ax.plot([x_val, x_val], [y_val, y_val], [z_val - err, z_val + err], zorder=2,
                        c='k', linewidth=1.5)
        
        ax.set_title(
            f'3D: p_input vs p_between vs avg_spikes per bin\n'
            f'I0={I0_value}pA, freq={oscillation_frequency}Hz, STDP={"On" if use_stdp else "Off"}, '
            f'p_within={p_within_str}, Time={current_time}ms, Bin={time_window_size}ms, {measure_name}',
            fontsize=14
        )
        ax.set_zlim(0, 2)
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


def plot_granger(time_series_v, ax_granger, ax_dtf, ax_pdc):
    # Параметры мультитаперного спектрального анализа.
    # Увеличиваем длительность временного окна для более чёткого частотного разрешения.

    # n_time_samples = time_series_s.shape[1]
    # print('n_tine', n_time_samples)
    # t_full = np.arange(n_time_samples) / 10

    # fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # # ax[0].plot(t_full, time_series[trial_idx, :, 0], 'o', label="x1")
    # ax[0].plot(t_full, time_series_s[trial_idx, :, 0], label="0-249")
    # ax[0].set_ylabel("Амплитуда")
    # ax[0].legend()

    # # ax[1].plot(t_full, time_series[trial_idx, :, 1], 'o', label="x2", color="orange")
    # ax[1].plot(t_full, time_series_s[trial_idx, :, 1], label="250-499", color="orange")
    # ax[1].set_xlabel("Время (сек)")
    # ax[1].set_ylabel("Амплитуда")
    # ax[1].legend()

    time_halfbandwidth_product = 5
    time_window_duration = 3
    time_window_step = 0.1

    print("Начало счета Multitaper")
    from spectral_connectivity import Multitaper, Connectivity
    print(time_series_v.shape)
    m = Multitaper(
        time_series_v,
        sampling_frequency=100,
        time_halfbandwidth_product=time_halfbandwidth_product,
        start_time=0,
        time_window_duration=time_window_duration,
        time_window_step=time_window_step,
    )
    # Рассчитываем объект Connectivity
    c = Connectivity.from_multitaper(m)

    # =============================================================================
    # 3. Расчет мер направленной связи
    # =============================================================================
    granger = c.pairwise_spectral_granger_prediction()
    dtf = c.directed_transfer_function()
    pdc = c.partial_directed_coherence()

    # 4.1. Спектральная Грейнджерова причинность
    ax_granger[0].pcolormesh(c.time, c.frequencies, granger[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_granger[0].set_title("GC: x1 → x2")
    ax_granger[0].set_ylabel("Frequency (Hz)")
    ax_granger[1].pcolormesh(c.time, c.frequencies, granger[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_granger[1].set_title("GC: x2 → x1")
    ax_granger[1].set_xlabel("Time (s)")
    ax_granger[1].set_ylabel("Frequency (Hz)")

    # 4.2. directed transfer function
    ax_dtf[0].pcolormesh(c.time, c.frequencies, dtf[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_dtf[0].set_title("DTF: x1 → x2")
    ax_dtf[0].set_ylabel("Frequency (Hz)")
    ax_dtf[1].pcolormesh(c.time, c.frequencies, dtf[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_dtf[1].set_title("DTF: x2 → x1")
    ax_dtf[1].set_xlabel("Time (s)")
    ax_dtf[1].set_ylabel("Frequency (Hz)")

    ax_pdc[0].pcolormesh(c.time, c.frequencies, pdc[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_pdc[0].set_title("PDC: x1 → x2")
    ax_pdc[0].set_ylabel("Frequency (Hz)")
    ax_pdc[1].pcolormesh(c.time, c.frequencies, pdc[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_pdc[1].set_title("PDC: x2 → x1")
    ax_pdc[1].set_xlabel("Time (s)")
    ax_pdc[1].set_ylabel("Frequency (Hz)")


def sim(p_within, p_between, nu_ext_over_nu_thr, J, J2, refractory_period, sim_time, plotting_flags,
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
    proportion_high_centrality = p_input
    centrality_type = measure_name
    boost_factor = 3

    C_total = generate_sbm_with_high_centrality(
        n_neurons, cluster_sizes, p_within, p_between,
        target_cluster_index, proportion_high_centrality,
        centrality_type, boost_factor,
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
    defaultclock.dt = 0.01 * second
    phi = 0
    f = 10 * Hz
    # -- Создаём группу нейронов --
    neurons = NeuronGroup(
        N,
        '''
        dv/dt = (v_rest - v + R*I_ext)/tau : volt
        I_ext = I0 * sin(2 * pi * f * t + phi) : amp
        I0 : amp
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


    centrality = print_centrality(C_total_matrix, N1, n_neurons, p_input, measure_name=measure_name, direct_1_2=True)



    if isinstance(centrality, np.ndarray):
        centrality = centrality.tolist()
    centrality  = sorted(list(centrality))

    # # Назначение параметров модуляции для соответствующих временных интервалов:
    # if centrality:
    neurons.I0[centrality] = I0_value * pA

    percent_central = len(centrality) * 100 / N1

    # p_intra1 = p_within
    # p_intra2 = p_within
    # p_12     = p_between
    # p_21     = 0.02

    # # Веса соединения
    # w_intra1 = 1
    # w_intra2 = 1
    # w_12     = 1
    # w_21     = 1

    # n_half = n_neurons // 2


    input_rate = 1 * Hz
    input_group = PoissonGroup(n_neurons, rates=input_rate)
    syn_input = Synapses(input_group, neurons, on_pre='v_post += J')
    syn_input.connect()

    # S_intra1 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_intra1.connect(
    #     condition='i <= n_half and j <= n_half',
    #     p=p_intra1
    # )
    # S_intra1.w = w_intra1

    # # 2) Синапсы внутри 2-го кластера
    # S_intra2 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_intra2.connect(
    #     condition='i >= n_half and j >= n_half',
    #     p=p_intra2
    # )
    # S_intra2.w = w_intra2

    # # 3) Синапсы из 1-го кластера во 2-й
    # S_12 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_12.connect(
    #     condition='i < n_half and j >= n_half',
    #     p=p_12
    # )
    # S_12.w = w_12

    # # 4) Синапсы из 2-го кластера в 1-й
    # S_21 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_21.connect(
    #     condition='i >= n_half and j < n_half',
    #     p=p_21
    # )
    # S_21.w = w_21


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
                            on_pre="v_post += -J2 * w", 
                            delay=D)
    

    # Генерация источников и целей
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
    sources_exc = np.concatenate((sources_exc1, sources_exc2))
    targets_exc = np.concatenate((targets_exc1, targets_exc2))
    sources_inh = np.concatenate((sources_inh1, sources_inh2))
    targets_inh = np.concatenate((targets_inh1, targets_inh2))
    exc_synapses.connect(i=sources_exc, j=targets_exc)
    inh_synapses.connect(i=sources_inh, j=targets_inh)


    for idx in range(len(exc_synapses.i)):
        pre_neuron = exc_synapses.i[idx]
        post_neuron = exc_synapses.j[idx]
        if pre_neuron < N1 and post_neuron < n_neurons:
            exc_synapses.w[idx] = 1  # для синапсов внутри первого кластера
        else:
            exc_synapses.w[idx] = 1  # для остальных связей
    
    for idx in range(len(inh_synapses.i)):
        pre_neuron = inh_synapses.i[idx]
        post_neuron = inh_synapses.j[idx]
        if pre_neuron < N1 and post_neuron < n_neurons:
            inh_synapses.w[idx] = 1  # для синапсов внутри первого кластера
        else:
            inh_synapses.w[idx] = 1  # для остальных связей

    # Мониторы  
    spike_monitor = SpikeMonitor(neurons)

    rate_monitor = None
    rate_monitor2 = None
    if plotting_flags.get('rates', False) or plotting_flags.get('psd', False) or plotting_flags.get('spectrogram', False):
        rate_monitor = PopulationRateMonitor(neurons[:int(n_neurons/2)])
    if plotting_flags.get('rates2', False) or plotting_flags.get('psd2', False) or plotting_flags.get('spectrogram', False):
        rate_monitor2 = PopulationRateMonitor(neurons[int(n_neurons/2):])

   
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

    trace_times = trace.t / ms
    
    mask1 = spike_indices < N1
    mask2 = spike_indices >= N1
    spike_times1 = spike_times[mask1]
    spike_times2 = spike_times[mask2]

    x1 = trace.v[:n_neurons//2, :] / mV  # (форма: n_neurons//2, 1000)
    x2 = trace.v[n_neurons//2:, :] / mV  # (форма: n_neurons//2, 1000)

    trial0 = x1.T  # (1000, n_neurons//2)
    trial1 = x2.T  # (1000, n_neurons//2

    time_series_v = np.stack((trial0, trial1), axis=-1)

    # fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # # ax[0].plot(t_full, time_series[trial_idx, :, 0], 'o', label="x1")
    # ax[0].plot(trace_times, x1[0], label="0-249")
    # ax[0].set_ylabel("Амплитуда")
    # ax[0].legend()

    # # ax[1].plot(t_full, time_series[trial_idx, :, 1], 'o', label="x2", color="orange")
    # ax[1].plot(trace_times, x2[0], label="250-499", color="orange")
    # ax[1].set_xlabel("Время (сек)")
    # ax[1].set_ylabel("Амплитуда")
    # ax[1].legend()

    # plt.show()

    # # Определяем параметры дискретизации времени:
    # bin_width = 1.0  # можно задать, например, 1 единицу времени
    # t_min = spike_times.min()
    # t_max = spike_times.max()
    # time_bins = np.arange(t_min, t_max + bin_width, bin_width)
    # n_time_bins = len(time_bins) - 1

    # # Инициализация матриц для каждой группы:
    # tensor_group1 = np.zeros((N1, n_time_bins))
    # tensor_group2 = np.zeros((N1, n_time_bins))

    # # Группа 1: нейроны с индексами 0-249
    # group1_neurons = spike_indices[mask1]
    # group1_times = spike_times[mask1]
    # for neuron in range(N1):
    #     # Извлекаем спайки для данного нейрона:
    #     neuron_spikes = group1_times[group1_neurons == neuron]
    #     # Формируем гистограмму по времени:
    #     counts, _ = np.histogram(neuron_spikes, bins=time_bins)
    #     tensor_group1[neuron, :] = counts

    # # Группа 2: нейроны с индексами 250-499, приводим индексы к диапазону 0-249
    # group2_neurons = spike_indices[mask2] - N1
    # group2_times = spike_times[mask2]
    # for neuron in range(N1):
    #     neuron_spikes = group2_times[group2_neurons == neuron]
    #     counts, _ = np.histogram(neuron_spikes, bins=time_bins)
    #     tensor_group2[neuron, :] = counts


    # time_series_s = np.stack((tensor_group1, tensor_group2), axis=-1)

    print("Форма 3D тензора:", time_series_v.shape)

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
    if plotting_flags.get('granger', False):
        ax_granger = plotting_flags['ax_granger']
        ax_dtf = plotting_flags['ax_dtf']
        ax_pdc = plotting_flags['ax_pdc']
        plot_granger(time_series_v, ax_granger, ax_dtf, ax_pdc)

        
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
        plot_spectrogram(rate_monitor, rate_monitor2, N1, N2, ax_spectrogram)

    if plotting_flags.get('connectivity2', False):
        ax_connectivity2 = plotting_flags['ax_connectivity2']
        plot_connectivity(n_neurons, exc_synapses, inh_synapses, ax_connectivity2, centrality, p_in_measured, p_out_measured, percent_central)
        # plot_connectivity(n_neurons, S_intra1, S_intra2, S_12, S_21, connectivity2, centrality, p_in_measured, p_out_measured, percent_central)

    return avg_neuron_spikes_cluster2_list, time_window_centers, C_total, spike_indices, centrality, p_in_measured, p_out_measured

# Флаги для построения графиков
do_plot_granger = True
do_plot_spikes = True
do_plot_avg_spikes = False
do_plot_rates = True
do_plot_rates2 = True
do_plot_connectivity2 = True
do_plot_psd = True
do_plot_psd2 = True
do_plot_spectrogram = True

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
                      f"STDP = {'On' if use_stdp else 'Off'}, Time=2000 ms")

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
                                    plot_granger, plot_spikes, plot_rates, plot_rates2, plot_psd, plot_psd2, plot_spectrogram
                                ]):
                                    fig = plt.figure(figsize=(14, 12))
                                    fig.suptitle(
                                        f'I0={I0_value} pA, p_input={p_input:.2f},'
                                        f'p_within={p_within_str}, p_between={p_between}, '
                                        f'J={J}, J2={J2}, epsilon={epsilon}, nu_ext_over_nu_thr={nu_ext_over_nu_thr}, '
                                        f'D={D}, Time=2000 ms'
                                    )
                                    gs = fig.add_gridspec(ncols=6, nrows=3)
                                else:
                                    fig = None

                                plotting_flags = {
                                    'granger': do_plot_granger,
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
                                    if plot_granger:
                                        # Создаем два subplot для Granger (например, в двух строках одного столбца)
                                        ax_granger_1 = fig.add_subplot(gs[1, 3])
                                        ax_granger_2 = fig.add_subplot(gs[2, 3])
                                        plotting_flags['ax_granger'] = [ax_granger_1, ax_granger_2]
                                        ax_dtf_1 = fig.add_subplot(gs[1, 4])
                                        ax_dtf_2 = fig.add_subplot(gs[2, 4])
                                        plotting_flags['ax_dtf'] = [ax_dtf_1, ax_dtf_2]
                                        ax_pdc_1 = fig.add_subplot(gs[1, 5])
                                        ax_pdc_2 = fig.add_subplot(gs[2, 5])
                                        plotting_flags['ax_pdc'] = [ax_pdc_1, ax_pdc_2]
                                    if plot_spikes:
                                        plotting_flags['ax_spikes'] = fig.add_subplot(gs[0, :])
                                    if plot_rates:
                                        plotting_flags['ax_rates'] = fig.add_subplot(gs[1, 0])
                                    if plot_rates2:
                                        plotting_flags['ax_rates2'] = fig.add_subplot(gs[1, 1])
                                    if plot_connectivity:
                                        plotting_flags['ax_connectivity2'] = fig.add_subplot(gs[1, 2])
                                    if plot_psd:
                                        plotting_flags['ax_psd'] = fig.add_subplot(gs[2, 0])
                                    if plot_psd2:
                                        plotting_flags['ax_psd2'] = fig.add_subplot(gs[2, 1])
                                    if plot_spectrogram:
                                        plotting_flags['ax_spectrogram'] = fig.add_subplot(gs[2, 2])
                                avg_window_avg_neuron_spikes_cluster2_tests = []
                                # -- Запускаем несколько тестов (num_tests) --
                                for test_num in range(num_tests):
                                    print(f'\nI0={I0_value} pA, p_within={p_within_str}, '
                                        f'p_input={p_input:.2f}, p_between={p_between}, '
                                        f'tест={test_num + 1}, Time=2000 ms')
                                        # Если это не последний тест, отключаем построение Granger-графика
                                    if test_num < num_tests - 1:
                                        plotting_flags['granger'] = False
                                        plotting_flags['spikes'] = False  
                                        plotting_flags['rates'] = False 
                                        plotting_flags['rates2'] = False  
                                        plotting_flags['connectivity2'] = False  
                                        plotting_flags['psd'] = False 
                                        plotting_flags['psd2'] = False  
                                        plotting_flags['spectrogram'] = False 
                                    else:
                                        plotting_flags['granger'] = do_plot_granger  
                                        plotting_flags['spikes'] = do_plot_spikes 
                                        plotting_flags['rates'] = do_plot_rates 
                                        plotting_flags['rates2'] = do_plot_rates2
                                        plotting_flags['connectivity2'] = do_plot_connectivity2
                                        plotting_flags['psd'] = do_plot_psd
                                        plotting_flags['psd2'] = do_plot_psd2
                                        plotting_flags['spectrogram'] = do_plot_spectrogram

                                    # Очищаем оси на повторных прогонах
                                    if test_num > 0 and fig is not None:
                                        if do_plot_granger:
                                            for ax in plotting_flags.get('ax_granger', []):
                                                ax.clear()
                                            for ax in plotting_flags.get('ax_dtf', []):
                                                ax.clear()
                                            for ax in plotting_flags.get('ax_pdc', []):
                                                ax.clear()
                                        if do_plot_spikes and 'ax_spikes' in plotting_flags:
                                            plotting_flags['ax_spikes'].clear()
                                        if do_plot_rates and 'ax_rates' in plotting_flags:
                                            plotting_flags['ax_rates'].clear()
                                        if do_plot_rates2 and 'ax_rates2' in plotting_flags:
                                            plotting_flags['ax_rates2'].clear()
                                        if do_plot_psd and 'ax_psd' in plotting_flags:
                                            plotting_flags['ax_psd'].clear()
                                        if do_plot_psd2 and 'ax_psd2' in plotting_flags:
                                            plotting_flags['ax_psd2'].clear()
                                        if do_plot_spectrogram and 'ax_spectrogram' in plotting_flags:
                                            plotting_flags['ax_spectrogram'].clear()
                                        if do_plot_connectivity2 and 'ax_connectivity2' in plotting_flags:
                                            plotting_flags['ax_connectivity2'].clear()

                                    # -- Вызов функции симуляции --
                                    (avg_neuron_spikes_cluster2_list,
                                    time_window_centers,
                                    C_total,
                                    spike_indices, centrality, p_in_measured, p_out_measured) = sim(
                                        p_within,
                                        p_between,
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
# Конец основного цикла по времени симуляции

print("\nГенерация графиков завершена.")