from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import imageio
import io
import os
import csv
from mpl_toolkits.mplot3d import Axes3D

# Установка параметров вывода NumPy для отображения всех элементов массивов
np.set_printoptions(threshold=np.inf)

directory_path = 'results_ext_inh_test1'
if os.path.exists(directory_path):
    print(f"Директория '{directory_path}' существует")
else:
    os.mkdir(directory_path)
    print(f"Директория '{directory_path}' создана")

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

# Основная функция симуляции нейронной сети
def sim(p_within, p_between, g, nu_ext_over_nu_thr, J, sim_time, plotting_flags,
        rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes, V0_value,
        oscillation_frequency, use_stdp, time_window_size, C_total_prev=None,
        p_within_prev=None, p_between_prev=None):
    start_scope()
    
    n_neurons = len(cluster_labels)
    
    if C_total_prev is None:
        C_intra = gensbm2(n_neurons, cluster_labels, 0.01, cluster_sizes, directed=False)
        C_between = generate_constant_inter_cluster(n_neurons, cluster_labels, 0.01, directed=False)
    else:
        C_intra = C_total_prev.copy()
        C_between = C_total_prev.copy()
    
    C_intra = update_intra_cluster_connectivity(C_intra, cluster_labels, p_within)
    C_between = update_inter_cluster_connectivity(C_between, cluster_labels, p_between)
    
    C_total = np.maximum(C_intra, C_between)
    
    N = len(C_total)
    N_E = N / 2
    tau = 20 * ms    
    v_threshold = -50 * mV 
    v_reset = -70*mV      
    v_rest = -65*mV    
    J = J * mV    
    D = 1.5 * ms     

    defaultclock.dt = 0.1 * ms  
    
    neurons = NeuronGroup(
        N,
        '''
        dv/dt = (v_rest-v+v_ext)/tau : volt (unless refractory)
        v_ext = v0 * sin(2 * pi * f * t + phi) : volt
        v0 : volt
        f : Hz
        phi : 1
        ''',
        threshold="v > v_threshold",
        reset="v = v_reset",
        method="euler",
    )

    neurons.f = 0 * Hz
    neurons.v0 = 0 * volt
    neurons.phi = 0
    
    # Первые 50 нейронов осциллируют
    neurons.f[0:50] = oscillation_frequency * Hz
    neurons.v0[0:50] = V0_value * mV
    neurons.phi[0:50] = 0
    
    if use_stdp:
        tau_stdp = 20 * ms
        Aplus = 0.005
        Aminus = 0.005

        stdp_eqs = '''
        dApre/dt = -Apre / tau_stdp : 1 (event-driven)
        dApost/dt = -Apost / tau_stdp : 1 (event-driven)
        w : 1
        '''
        ############# Возбуждающие и тормозные
        exc_synapses = Synapses(neurons, neurons,
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
                                delay=D)
        inhib_synapses = Synapses(neurons, neurons, 
                              model="w : 1", 
                              on_pre="v_post += -g * J * w", 
                              delay=D)
    else:
        exc_synapses = Synapses(neurons, neurons,
                                model="w : 1",
                                on_pre="v_post += J * w",
                                delay=D)
        inhib_synapses = Synapses(neurons, neurons, 
                                model="w : 1", 
                                on_pre="v_post += -g * J * w", 
                                delay=D)


    exc_synapses.connect(condition='i != j and i < N_E')
    inhib_synapses.connect(condition='i != j and i >= N_E')


    sources, targets = C_total.nonzero()
    exc_synapses.w[:, :] = 0  # Обнуляем все веса
    inhib_synapses.w[:, :] = 0  # Обнуляем все веса
    # Устанавливаем веса в соответствии с начальной матрицей связности
    for idx in range(len(sources)):
        i = sources[idx]
        j = targets[idx]
        if i < N_E:
            # Возбуждающий синапс
            syn_idx = np.where((exc_synapses.i == i) & (exc_synapses.j == j))[0]
            if len(syn_idx) > 0:
                exc_synapses.w[syn_idx[0]] = 1
        else:
            # Тормозный синапс
            syn_idx = np.where((inhib_synapses.i == i) & (inhib_synapses.j == j))[0]
            if len(syn_idx) > 0:
                inhib_synapses.w[syn_idx[0]] = 1

    spike_monitor = SpikeMonitor(neurons[:100])
    
    rate_monitor = None
    if plotting_flags.get('rates', False) or plotting_flags.get('psd', False) or plotting_flags.get('spectrogram', False):
        rate_monitor = PopulationRateMonitor(neurons[:50])
    
    rate_monitor2 = None
    if plotting_flags.get('rates2', False) or plotting_flags.get('psd2', False) or plotting_flags.get('spectrogram', False):
        rate_monitor2 = PopulationRateMonitor(neurons[50:])
    
    trace = None
    if plotting_flags.get('trace', False):
        trace = StateMonitor(neurons, 'v', record=True)
    
    run(sim_time)
    
    spike_times = spike_monitor.t / ms
    spike_indices = spike_monitor.i

    # Устанавливаем интервалы времени вручную: от 0 до 1000 мс с шагом 100 мс
    # Предполагается sim_time = 1000 ms
    bins = np.arange(0, int(sim_time/ms) + 1, time_window_size) # Для time_window_size=100, bins=[0,100,200,...,1000]
    time_window_centers = (bins[:-1] + bins[1:]) / 2

    avg_neuron_spikes_cluster2_list = []
    start_cluster_neuron = 50
    end_cluster_neuron = 100
    neuron_spike_counts_per_window = {neuron: [] for neuron in range(start_cluster_neuron, end_cluster_neuron)}

    for i in range(len(bins) - 1):
        start_time = bins[i]
        end_time = bins[i + 1]
        
        selected_neurons_mask = (
            (spike_indices >= start_cluster_neuron) &
            (spike_indices < end_cluster_neuron) &
            (spike_times > start_time) &
            (spike_times < end_time)
        )
        filtered_spike_indices = spike_indices[selected_neurons_mask]
        
        group2_spikes = len(filtered_spike_indices)
        avg_neuron_spikes_cluster2 = group2_spikes / (end_cluster_neuron - start_cluster_neuron)
        avg_neuron_spikes_cluster2_list.append(avg_neuron_spikes_cluster2)

        for neuron in neuron_spike_counts_per_window.keys():
            neuron_spike_counts = np.sum(filtered_spike_indices == neuron)
            neuron_spike_counts_per_window[neuron].append(neuron_spike_counts)

    avg_neuron_spikes_cluster2_list = avg_neuron_spikes_cluster2_list[1:] 

    if plotting_flags.get('spikes', False):
        ax_spikes = plotting_flags['ax_spikes']
        ax_spikes.scatter(spike_times, spike_indices, marker='|')
        
        step_size = time_window_size
        total_time_ms = sim_time / ms
        time_steps = np.arange(0, total_time_ms + step_size, step_size)
        for t in time_steps:
            ax_spikes.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)
        
        ax_spikes.set_xlim(t_range)
        ax_spikes.set_xlabel("t [ms]")
        ax_spikes.set_ylabel("Neuron index")
        ax_spikes.set_title(f"Spike Raster Plot\nFrequency={oscillation_frequency} Hz, STDP={'On' if use_stdp else 'Off'}, Window={time_window_size} ms", fontsize=16)
   
    if plotting_flags.get('rates', False) and rate_monitor is not None:
        ax_rates = plotting_flags['ax_rates']
        ax_rates.set_title(f'Spike Rate neu.0-50\nFrequency={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}', fontsize=16)
        ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz, label='Group 1 (0-50)')
        ax_rates.set_xlim(t_range)
        ax_rates.set_xlabel("t [ms]")
        ax_rates.set_yticks(
            np.arange(
                rate_range[0], rate_range[1] + rate_tick_step, rate_tick_step
            )
        )
    
    if plotting_flags.get('rates2', False) and rate_monitor2 is not None:
        ax_rates2 = plotting_flags['ax_rates2']
        ax_rates2.set_title(f'Spike Rate neu.50-100\nFrequency={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}', fontsize=16)
        ax_rates2.plot(rate_monitor2.t / ms, rate_monitor2.rate / Hz, label='Group 2 (50-100)')
        ax_rates2.set_xlim(t_range)
        ax_rates2.set_xlabel("t [ms]")
    
    if plotting_flags.get('trace', False) and trace is not None:
        ax_trace = plotting_flags['ax_trace']
        time_points_trace = trace.t / ms
        offset = 10
        for neuron_index in range(0, 1, 1):
            ax_trace.plot(time_points_trace, trace.v[neuron_index] / mV + neuron_index * offset, label=f'Neuron {neuron_index}')
        
        ax_trace.set_xlabel("t [ms]")
        ax_trace.set_ylabel("Membrane potential [mV] + offset")
        ax_trace.legend(loc='upper right', ncol=2, fontsize='small')
        ax_trace.set_title(f'Membrane Potentials\nFrequency={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}', fontsize=16)
    
    if plotting_flags.get('psd', False) and rate_monitor is not None:
        ax_psd = plotting_flags['ax_psd']
        rate = rate_monitor.rate / Hz - np.mean(rate_monitor.rate / Hz)
        from numpy.fft import rfft, rfftfreq
        
        N = len(rate_monitor.t)
        if N > 0:
            sampling_rate = 10000
            max_point = int(N * 300 / sampling_rate)
            x = rfftfreq(N, d=1 / sampling_rate)
            x = x[:max_point]
            yn = 2 * np.abs(rfft(rate)) / N
            yn = yn[:max_point]
        
            ax_psd.set_title(f"PSD neu.0-50\nFrequency={oscillation_frequency} Hz, STDP={'On' if use_stdp else 'Off'}", fontsize=16)
            ax_psd.plot(x, yn, c='k', label='Function')
    
    if plotting_flags.get('psd2', False) and rate_monitor2 is not None:
        ax_psd2 = plotting_flags['ax_psd2']
        rate = rate_monitor2.rate / Hz - np.mean(rate_monitor2.rate / Hz)
        
        N = len(rate_monitor2.t)
        if N > 0:
            sampling_rate = 10000
            max_point = int(N * 300 / sampling_rate)
            x = rfftfreq(N, d=1 / sampling_rate)
            x = x[:max_point]
            yn = 2 * np.abs(rfft(rate)) / N
            yn = yn[:max_point]
        
            ax_psd2.set_title(f"PSD neu.50-100\nFrequency={oscillation_frequency} Hz, STDP={'On' if use_stdp else 'Off'}", fontsize=16)
            ax_psd2.plot(x, yn, c='k', label='Function')

    if plotting_flags.get('spectrogram', False) and rate_monitor is not None and rate_monitor2 is not None:
        ax_spectrogram = plotting_flags['ax_spectrogram']
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
    
    if plotting_flags.get('connectivity2', False):
        connectivity2 = plotting_flags['connectivity2']
        W = np.zeros((n_neurons, n_neurons))
        W[exc_synapses.i[:], exc_synapses.j[:]] = exc_synapses.w[:]
        connectivity2.set_title(f'Weight Matrix\nSTDP={"On" if use_stdp else "Off"}', fontsize=16)
        connectivity2.matshow(W, cmap='viridis')

    return avg_neuron_spikes_cluster2_list, time_window_centers, C_total, spike_indices


# Инициализация параметров сети
n_neurons = 100
num_of_clusters = 2
cluster_sizes = [50, 50]
p_within_values = np.arange(0.7, 1.01, 0.1)
num_tests = 1
J = 0.1
g = 6
epsilon = 0.1
nu_ext_over_nu_thr = 4
D = 1.5 * ms
rate_tick_step = 50
t_range = [0, 5000]
rate_range = [0, 200]
oscillation_frequencies = [10]  
use_stdp_values = [True]  
V0_values = [100]
time_window_size = 100  # in ms

plot_spikes = True
plot_avg_spikes = False
plot_rates = False
plot_rates2 = False
plot_connectivity2 = True
plot_trace = True
plot_psd = False
plot_psd2 = False
plot_spectrogram = False

cluster_labels = []
for i in range(n_neurons):
    cluster_labels.append(clcheck(i, cluster_sizes))

simulation_times = [5000]  # в миллисекундах

for current_time in simulation_times:
    sim_time = current_time * ms
    t_range = [0, current_time]

    spike_counts_second_cluster = {}
    # Для 3D графика:
    # detailed_spike_data_for_3d = {V0_value: {p_within: {p_between: {"time": ..., "spikes_list": ...}}}}
    detailed_spike_data_for_3d = {}

    for oscillation_frequency in oscillation_frequencies:
        for V0_value in V0_values:
            for use_stdp in use_stdp_values:
                print(f"Running simulations for V0 = {V0_value} mV, Frequency = {oscillation_frequency} Hz, STDP = {'On' if use_stdp else 'Off'}, Time={current_time} ms")
                spike_counts_second_cluster[V0_value] = {p_within: {} for p_within in p_within_values}
                detailed_spike_data_for_3d[V0_value] = {p_within: {} for p_within in p_within_values}

                for p_within_idx, p_within in enumerate(p_within_values):
                    images = []
                    p_between_values = np.arange(0.05, p_within - 0.04, 0.1)
                    C_total_prev = None
                    p_within_prev = None
                    p_between_prev = None
                    for p_between in p_between_values:
                        avg_window_avg_neuron_spikes_cluster2_tests = []
                        
                        # Переменные для 3D графика
                        spikes_for_3d_time = None
                        time_for_3d_time = None

                        if any([plot_spikes, plot_avg_spikes, plot_rates, plot_rates2, plot_connectivity2, plot_trace, plot_psd, plot_psd2, plot_spectrogram]):
                            fig = plt.figure(figsize=(12, 10))
                            fig.suptitle(f'V0={V0_value} мВ, p_within={round(p_within, 2)}, p_between={round(p_between, 2)}, g={g}, epsilon={epsilon}, nu_ext_over_nu_thr={nu_ext_over_nu_thr}, D={D}, Time={current_time} ms')
                            gs = fig.add_gridspec(ncols=4, nrows=3)
                        else:
                            fig = None

                        plotting_flags = {
                            'spikes': plot_spikes,
                            'avg_spikes': plot_avg_spikes,
                            'rates': plot_rates,
                            'rates2': plot_rates2,
                            'connectivity2': plot_connectivity2,
                            'trace': plot_trace,
                            'psd': plot_psd,
                            'psd2': plot_psd2,
                            'spectrogram': plot_spectrogram,
                        }

                        if plot_spikes:
                            plotting_flags['ax_spikes'] = fig.add_subplot(gs[0, :])
                        if plot_trace:
                            plotting_flags['ax_trace'] = fig.add_subplot(gs[1, 0])
                        if plot_rates:
                            plotting_flags['ax_rates'] = fig.add_subplot(gs[1, 1])
                        if plot_rates2:
                            plotting_flags['ax_rates2'] = fig.add_subplot(gs[1, 2])
                        if plot_connectivity2:
                            plotting_flags['connectivity2'] = fig.add_subplot(gs[1, 3])
                        if plot_avg_spikes:
                            plotting_flags['ax_avg_spikes'] = fig.add_subplot(gs[2, 0])
                        if plot_psd:
                            plotting_flags['ax_psd'] = fig.add_subplot(gs[2, 1])
                        if plot_psd2:
                            plotting_flags['ax_psd2'] = fig.add_subplot(gs[2, 2])
                        if plot_spectrogram:
                            plotting_flags['ax_spectrogram'] = fig.add_subplot(gs[2, 3])

                        for test_num in range(num_tests):
                            print(f'V0={V0_value} мВ, p_within={round(p_within, 2)}, p_between={round(p_between, 2)}, тест={test_num + 1}, Time={current_time} ms')
                            if test_num > 0 and fig is not None:
                                if plot_spikes:
                                    plotting_flags['ax_spikes'].clear()
                                if plot_rates:
                                    plotting_flags['ax_rates'].clear()
                                if plot_rates2:
                                    plotting_flags['ax_rates2'].clear()
                                if plot_trace:
                                    plotting_flags['ax_trace'].clear()
                                if plot_psd:
                                    plotting_flags['ax_psd'].clear()
                                if plot_psd2:
                                    plotting_flags['ax_psd2'].clear()
                                if plot_spectrogram:
                                    plotting_flags['ax_spectrogram'].clear()

                            avg_neuron_spikes_cluster2_list, time_window_centers, C_total, spike_indices = sim(
                                round(p_within, 2), round(p_between, 2), g, nu_ext_over_nu_thr, J, sim_time, 
                                plotting_flags, rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes, V0_value,
                                oscillation_frequency, use_stdp, time_window_size,
                                C_total_prev, p_within_prev, p_between_prev
                            )  

                            # Если данных по времени нет, проверим и присвоим пустой массив, чтобы избежать None
                            if time_window_centers is None:
                                time_window_centers = np.array([])

                            if test_num == 0:
                                spikes_for_3d_time = avg_neuron_spikes_cluster2_list if avg_neuron_spikes_cluster2_list is not None else []
                                time_for_3d_time = time_window_centers if time_window_centers is not None else np.array([])

                            avg_window_avg_neuron_spikes_cluster2_list = np.mean(avg_neuron_spikes_cluster2_list) if len(avg_neuron_spikes_cluster2_list) > 0 else 0
                            avg_window_avg_neuron_spikes_cluster2_tests.append(avg_window_avg_neuron_spikes_cluster2_list)

                            C_total_prev = C_total.copy()
                            p_within_prev = p_within
                            p_between_prev = p_between

                            cluster_indices_1 = np.where(np.array(cluster_labels) == 0)[0]
                            total_possible_connections_1 = len(cluster_indices_1) * (len(cluster_indices_1) - 1) // 2
                            actual_connections_1 = np.sum(np.triu(C_total[np.ix_(cluster_indices_1, cluster_indices_1)], k=1))
                            mean_connectivity_within_cluster1 = actual_connections_1 / total_possible_connections_1 if total_possible_connections_1 > 0 else 0
                            print(f"Средняя связность внутри первого кластера (0-50): {round(mean_connectivity_within_cluster1, 2)}")

                            cluster_indices_2 = np.where(np.array(cluster_labels) == 1)[0]
                            total_possible_connections_between = len(cluster_indices_1) * len(cluster_indices_2)
                            actual_connections_between = np.sum(C_total[np.ix_(cluster_indices_1, cluster_indices_2)])
                            mean_connectivity_between_clusters = actual_connections_between / total_possible_connections_between if total_possible_connections_between > 0 else 0
                            print(f"Средняя связность между кластерами: {round(mean_connectivity_between_clusters, 2)}")

                        avg_window_avg_neuron_spikes_cluster2_tests = np.array(avg_window_avg_neuron_spikes_cluster2_tests)

                        if p_between not in spike_counts_second_cluster[V0_value][p_within]:
                            spike_counts_second_cluster[V0_value][p_within][p_between] = []
                        spike_counts_second_cluster[V0_value][p_within][p_between].append(avg_window_avg_neuron_spikes_cluster2_tests)

                        detailed_spike_data_for_3d[V0_value][p_within][p_between] = {
                            "time": time_for_3d_time if time_for_3d_time is not None else np.array([]),
                            "spikes_list": spikes_for_3d_time if spikes_for_3d_time is not None else []
                        }

                        if fig is not None:
                            plt.tight_layout()
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            images.append(imageio.imread(buf))
                            plt.close(fig)

                    if images:
                        imageio.mimsave(f'results_ext_inh_test1/gif_exc_V0_{V0_value}freq_{oscillation_frequency}_STDP_{"On" if use_stdp else "Off"}_p_within_{p_within:.2f}_Time_{current_time}ms_Bin_{time_window_size}ms.gif', images, duration=3000, loop=0)

                print(f"Симуляции для V0 = {V0_value} мВ, Time={current_time} ms завершены.")

                # Построение графика зависимости среднего числа спайков во втором кластере
                fig_spike_dependency, ax_spike_dep = plt.subplots(figsize=(10, 6))
                data_for_csv = []
                data_for_csv.append(["p_within", "p_between", "avg_spike_counts", "std_spike_counts"])

                for p_within in p_within_values:
                    p_between_list = sorted(spike_counts_second_cluster[V0_value][p_within].keys())
                    avg_spike_counts = []
                    std_spike_counts = []
                    for p_between in p_between_list:
                        spike_counts = spike_counts_second_cluster[V0_value][p_within][p_between]
                        avg_spike = np.mean(spike_counts) if len(spike_counts) > 0 else 0
                        std_spike = np.std(spike_counts) if len(spike_counts) > 1 else 0
                        avg_spike_counts.append(avg_spike)
                        std_spike_counts.append(std_spike)
                        data_for_csv.append([p_within, p_between, avg_spike, std_spike])
                    ax_spike_dep.errorbar(p_between_list, avg_spike_counts, yerr=std_spike_counts, marker='o', label=f'p_within={p_within:.1f}')
                
                ax_spike_dep.set_xlabel('p_between', fontsize=14)
                ax_spike_dep.set_ylabel('Среднее число спайков во втором кластере', fontsize=14)
                ax_spike_dep.set_title(f'Зависимость среднего числа спайков во втором кластере от p_within и p_between\next_V0_{V0_value}mV_freq_{oscillation_frequency}Hz_STDP_{"On" if use_stdp else "Off"}_Time_{current_time}ms_Bin_{time_window_size}ms', fontsize=16)
                ax_spike_dep.legend(title='p_within', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=12)
                
                ax_spike_dep.tick_params(axis='both', which='major', labelsize=12)
                
                plt.tight_layout()
                
                plt.savefig(f'results_ext_inh_test1/avg_spike_dependency_ext_V0_{V0_value}mV_freq_{oscillation_frequency}Hz_STDP_{"On" if use_stdp else "Off"}_Time_{current_time}ms_Bin_{time_window_size}ms.png')
                plt.close(fig_spike_dependency)

                # Сохраняем данные в CSV
                csv_filename = f'results_ext_inh_test1/avg_spike_dependency_ext_data_V0_{V0_value}mV_freq_{oscillation_frequency}Hz_STDP_{"On" if use_stdp else "Off"}_Time_{current_time}ms_Bin_{time_window_size}ms.csv'
                with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    for row in data_for_csv:
                        writer.writerow(row)

                # Построение 3D графика
                # Проверим, есть ли данные для построения
                for p_within in p_within_values:
                    p_between_list = sorted(detailed_spike_data_for_3d[V0_value][p_within].keys())
                    if len(p_between_list) > 0:
                        # Берём время из любого p_between, предположим первого
                        sample_p_between = p_between_list[0]
                        time_array = detailed_spike_data_for_3d[V0_value][p_within][sample_p_between]["time"]
                        if time_array is None or len(time_array) == 0:
                            # Нет временных точек, пропускаем 3D график
                            continue
                        time_array = time_array[1:]
                        
                        Time, P_between = np.meshgrid(time_array, p_between_list)
                        max_time_len = max(
                            len(detailed_spike_data_for_3d[V0_value][p_within][p_btw]["time"])
                            for p_btw in p_between_list
                            if "time" in detailed_spike_data_for_3d[V0_value][p_within][p_btw]
                        )
                        
                        Z = np.zeros((len(p_between_list), max_time_len-1))
                        for i, p_btw in enumerate(p_between_list):
                            spikes_arr = detailed_spike_data_for_3d[V0_value][p_within][p_btw]["spikes_list"]
                            print('spike_arr',spikes_arr)
                            if spikes_arr is None or len(spikes_arr) == 0:
                                Z[i, :] = 0
                            else:
                                padded_spikes = np.zeros(max_time_len-1)
                                padded_spikes[:len(spikes_arr)] = spikes_arr
                                Z[i, :] = padded_spikes

                        print('Time',Time)
                        print('p_between', p_between)
                        print('Z',Z)
                        fig_3d = plt.figure(figsize=(10, 8))
                        ax_3d = fig_3d.add_subplot(111, projection='3d')
                        surf = ax_3d.plot_surface(Time, P_between, Z, cmap='viridis')
                        ax_3d.set_xlabel('Time [ms]')
                        ax_3d.set_ylabel('p_between')
                        ax_3d.set_zlim(0, 1.5)  # Устанавливаем диапазон оси z
                        ax_3d.set_zlabel('Avg Spikes in 100 ms window')
                        ax_3d.set_title(f'3D Surface: Time vs p_between vs Avg Spikes\nV0={V0_value}mV, freq={oscillation_frequency}Hz, STDP={"On" if use_stdp else "Off"}, p_within={round(p_within, 2)}', fontsize=14)
                        fig_3d.colorbar(surf, shrink=0.5, aspect=5)
                        plt.savefig(f'results_ext_inh_test1/3D_plot_V0_{V0_value}mV_freq_{oscillation_frequency}Hz_STDP_{"On" if use_stdp else "Off"}_p_within_{round(p_within, 2)}_Time_{current_time}ms_Bin_{time_window_size}ms.png')
                        # plt.close(fig_3d)
                        plt.show()
print("Генерация графиков завершена.")
