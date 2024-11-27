from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import imageio
import io

# Установка параметров вывода NumPy для отображения всех элементов массивов
np.set_printoptions(threshold=np.inf)

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

# Функция для обновления внутрикластерных связей на основе вероятности p_within
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

# Функция для обновления межкластерных связей на основе вероятности p_between
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
        rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes, I0_value, C_total_prev=None, p_within_prev=None, p_between_prev=None):
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
    theta = 20 * mV
    V_r = 10 * mV
    tau_rp = 2 * ms
    
    J = J * mV
    D = 1.5 * ms
    
    C_E = p_within * N_E
    C_ext = int(C_E)
    nu_thr = theta / (J * C_E * tau)
    
    defaultclock.dt = 0.1 * ms
    
    neurons = NeuronGroup(N,
        '''
        dv/dt = (-v + I_ext)/tau : volt (unless refractory)
        I_ext = I0 * sin(2 * pi * f * t + phi) : volt
        I0 : volt
        f : Hz
        phi : 1
        ''',
        threshold="v > theta",
        reset="v = V_r",
        refractory=tau_rp,
        method="euler",
    )
    
    neurons.f = 0 * Hz
    neurons.I0 = 0 * volt
    neurons.phi = 0
    
    neurons.f[0:50] = 10 * Hz
    neurons.I0[0:50] = I0_value * mV  # Изменяем амплитуду внешнего тока
    neurons.phi[0:50] = 0
    
    stdp_eqs = '''
    dApre/dt = -Apre / tau_stdp : 1 (event-driven)
    dApost/dt = -Apost / tau_stdp : 1 (event-driven)
    w : 1
    '''
    
    tau_stdp = 20 * ms
    Aplus = 0.005
    Aminus = 0.005
    
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
    # exc_synapses = Synapses(neurons, neurons,
    #                         model='w : 1',
    #                         on_pre='v += J * w',
    #                         delay=D)
    
    exc_synapses.connect(condition='i != j')
    
    sources, targets = C_total.nonzero()
    exc_synapses.w[:, :] = 0
    for idx in range(len(sources)):
        i = sources[idx]
        j = targets[idx]
        syn_idx = np.where((exc_synapses.i == i) & (exc_synapses.j == j))[0]
        if len(syn_idx) > 0:
            exc_synapses.w[syn_idx[0]] = 1
    
    nu_ext_group1 = nu_ext_over_nu_thr * nu_thr * 0.2
    
    external_poisson_input_1 = PoissonInput(
        target=neurons,
        target_var="v",
        N=C_ext,
        rate=nu_ext_group1,
        weight=J
    )
    
    # Создание мониторов в зависимости от флагов
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

    bin_size = 100
    max_time = int(np.max(spike_times)) + bin_size
    bins = np.arange(0, max_time, bin_size)
    
    avg_spikes_cluster2_list = []

    spike_times_list = []
    spike_indices_list = []
    if plotting_flags.get('spikes', False):
        spike_times_list = []
        spike_indices_list = []
    
    for i in range(len(bins) - 1):
        start_time = bins[i]
        end_time = bins[i + 1]
        mask = (spike_times >= start_time) & (spike_times < end_time)
        window_spike_times = spike_times[mask]
        window_spike_indices = spike_indices[mask]
    
        group2_mask = (window_spike_indices >= 50) & (window_spike_indices < 100)
    
        group2_spikes = np.sum(group2_mask)

        avg_spikes_cluster2 = group2_spikes / 50
        avg_spikes_cluster2_list.append(avg_spikes_cluster2)
    
        if plotting_flags.get('spikes', False):
            spike_times_list.extend(window_spike_times)
            spike_indices_list.extend(window_spike_indices)
    
    time_window_centers = (bins[:-1] + bins[1:]) / 2
    
    if plotting_flags.get('spikes', False):
        spike_times_array = np.array(spike_times_list)
        spike_indices_array = np.array(spike_indices_list)
    
    # Построение графиков только если соответствующий флаг установлен
    if plotting_flags.get('spikes', False):
        ax_spikes = plotting_flags['ax_spikes']
        ax_spikes.scatter(spike_times_array, spike_indices_array, marker='|')
        
        # Добавление вертикальных линий каждые 100 мс
        step_size = 100
        total_time_ms = sim_time / ms
        time_steps = np.arange(0, total_time_ms + step_size, step_size)
        for t in time_steps:
            ax_spikes.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)
        
        ax_spikes.set_xlim(t_range)
        ax_spikes.set_xlabel("t [ms]")
        ax_spikes.set_ylabel("Номер нейрона")
        ax_spikes.set_title("Спайковая растровая диаграмма", fontsize=16)
    
    if plotting_flags.get('rates', False):
        ax_rates = plotting_flags['ax_rates']
        ax_rates.set_title('Частота спайков\n neu.0-50', fontsize=16)
        ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz, label='Группа 1 (0-50)')
        ax_rates.set_xlim(t_range)
        ax_rates.set_xlabel("t [ms]")
        ax_rates.set_yticks(
            np.arange(
                rate_range[0], rate_range[1] + rate_tick_step, rate_tick_step
            )
        )
    
    if plotting_flags.get('rates2', False):
        ax_rates2 = plotting_flags['ax_rates2']
        ax_rates2.set_title('Частота спайков\n neu.50-100', fontsize=16)
        ax_rates2.plot(rate_monitor2.t / ms, rate_monitor2.rate / Hz, label='Группа 2 (50-100)')
        ax_rates2.set_xlim(t_range)
        ax_rates2.set_xlabel("t [ms]")
    
    if plotting_flags.get('trace', False):
        ax_trace = plotting_flags['ax_trace']
        time_points_trace = trace.t / ms
        offset = 10
        
        for neuron_index in range(0, n_neurons, 10):
            ax_trace.plot(time_points_trace, trace.v[neuron_index] / mV + neuron_index * offset, label=f'Neuron {neuron_index}')
        
        ax_trace.set_xlabel("t [ms]")
        ax_trace.set_ylabel("Мембранный потенциал [mV] + сдвиг")
        ax_trace.legend(loc='upper right', ncol=2, fontsize='small')
    
    if plotting_flags.get('psd', False):
        ax_psd = plotting_flags['ax_psd']
        rate = rate_monitor.rate / Hz - np.mean(rate_monitor.rate / Hz)
        from numpy.fft import rfft, rfftfreq
        
        N = len(rate_monitor.t)
        sampling_rate = 10000
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate)) / N
        yn = yn[:max_point]
        
        ax_psd.set_title("PSD neu.0-50", fontsize=16)
        ax_psd.plot(x, yn, c='k', label='Функция')
    
    if plotting_flags.get('psd2', False):
        ax_psd2 = plotting_flags['ax_psd2']
        rate = rate_monitor2.rate / Hz - np.mean(rate_monitor2.rate / Hz)
        
        N = len(rate_monitor2.t)
        sampling_rate = 10000
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate)) / N
        yn = yn[:max_point]
        
        ax_psd2.set_title("PSD neu.50-100", fontsize=16)
        ax_psd2.plot(x, yn, c='k', label='Функция')

    if plotting_flags.get('spectrogram', False):
        ax_spectrogram = plotting_flags['ax_spectrogram']
        yf = rfft(rate_monitor.rate / Hz)
        N_freq = len(yf)
        xf = rfftfreq(len(rate_monitor.t), 1 / 10000)[:N_freq]
        ax_spectrogram.plot(xf, np.abs(yf), label='0-50 нейронов')
        
        yf2 = rfft(rate_monitor2.rate / Hz)
        N_freq2 = len(yf2)
        xf2 = rfftfreq(len(rate_monitor2.t), 1 / 10000)[:N_freq2]
        ax_spectrogram.plot(xf2, np.abs(yf2), label='50-100 нейронов')
        
        ax_spectrogram.set_xlim(0, 1000)
        ax_spectrogram.legend()
        ax_spectrogram.set_title('Глобальные частоты\n нейронов 1 и 2 кластера', fontsize=16)
    
    if plotting_flags.get('connectivity2', False):
        connectivity2 = plotting_flags['connectivity2']
        W = np.zeros((n_neurons, n_neurons))
        W[exc_synapses.i[:], exc_synapses.j[:]] = exc_synapses.w[:]
        connectivity2.set_title('Матрица весов', fontsize=16)
        connectivity2.matshow(W, cmap='viridis')
    plt.show()
    return avg_spikes_cluster2_list, time_window_centers, C_total, spike_indices

# Параметры симуляции
p_within_values = np.arange(0.1, 1.05, 0.1)
num_tests = 1 # Увеличено для получения стандартного отклонения
J = 0.1
g = 6
time = 5000
sim_time = time * ms
epsilon = 0.1
nu_ext_over_nu_thr = 4
D = 1.5 * ms
rate_tick_step = 50
t_range = [0, time]
rate_range = [0, 200]

# Флаги для контроля отрисовки графиков
plot_spikes = True
plot_avg_spikes = False
plot_rates = False
plot_rates2 = False
plot_connectivity2 = False
plot_trace = False
plot_psd = False
plot_psd2 = False
plot_spectrogram = False

# Инициализация параметров сети
n_neurons = 100
num_of_clusters = 2
cluster_sizes = [50, 50]

# Генерация меток кластеров для каждого нейрона
cluster_labels = []
for i in range(n_neurons):
    cluster_labels.append(clcheck(i, cluster_sizes))

I0_values = np.arange(90, 100, 40)  # От 10 мВ до 100 мВ с шагом 10 мВ
spike_counts_second_cluster = {}  # Словарь для хранения результатов

# Основной цикл симуляции по I0_values
for I0_value in I0_values:
    print(f"Запуск симуляций для I0 = {I0_value} мВ")
    spike_counts_second_cluster[I0_value] = {p_within: {} for p_within in p_within_values}

    for p_within_idx, p_within in enumerate(p_within_values):
        images = []
        p_between_values = np.arange(0.05, p_within - 0.04, 0.05)
        C_total_prev = None
        p_within_prev = None
        p_between_prev = None
        for p_between in p_between_values:
            avg_spikes_cluster2_tests = []
            # Создаем фигуру и сетку подграфиков только если хотя бы один график должен быть отрисован
            if any([plot_spikes, plot_avg_spikes, plot_rates, plot_rates2, plot_connectivity2, plot_trace, plot_psd, plot_psd2, plot_spectrogram]):
                fig = plt.figure(figsize=(12, 10))
                fig.suptitle(f'I0={I0_value} мВ, p_within={round(p_within, 2)}, p_between={round(p_between, 2)}, g={g}, epsilon={epsilon}, nu_ext_over_nu_thr={nu_ext_over_nu_thr}, D={D}')
                gs = fig.add_gridspec(ncols=4, nrows=3)
            else:
                fig = None  # Фигура не создается

            # Инициализация словаря для передачи в функцию sim
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

            # Создаем подграфики только если соответствующий флаг установлен
            if plot_spikes:
                plotting_flags['ax_spikes'] = fig.add_subplot(gs[0, :])
            if plot_avg_spikes:
                plotting_flags['ax_avg_spikes'] = fig.add_subplot(gs[1, 0])
            if plot_rates:
                plotting_flags['ax_rates'] = fig.add_subplot(gs[1, 1])
            if plot_rates2:
                plotting_flags['ax_rates2'] = fig.add_subplot(gs[1, 2])
            if plot_connectivity2:
                plotting_flags['connectivity2'] = fig.add_subplot(gs[1, 3])
            if plot_trace:
                plotting_flags['ax_trace'] = fig.add_subplot(gs[2, 0])
            if plot_psd:
                plotting_flags['ax_psd'] = fig.add_subplot(gs[2, 1])
            if plot_psd2:
                plotting_flags['ax_psd2'] = fig.add_subplot(gs[2, 2])
            if plot_spectrogram:
                plotting_flags['ax_spectrogram'] = fig.add_subplot(gs[2, 3])

            for test_num in range(num_tests):
                print(f'I0={I0_value} мВ, p_within={round(p_within, 2)}, p_between={round(p_between, 2)}, тест={test_num + 1}')
                # Очищаем оси только если они существуют
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

                avg_spikes_cluster2_list, time_window_centers, C_total, spike_indices = sim(
                    round(p_within, 2), round(p_between, 2), g, nu_ext_over_nu_thr, J, sim_time, 
                    plotting_flags, rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes, I0_value,
                    C_total_prev, p_within_prev, p_between_prev
                )
                avg_spikes_cluster2_tests.append(avg_spikes_cluster2_list)
    
                C_total_prev = C_total.copy()
                p_within_prev = p_within
                p_between_prev = p_between
    
                cluster_indices_1 = np.where(np.array(cluster_labels) == 0)[0]
                total_possible_connections_1 = len(cluster_indices_1) * (len(cluster_indices_1) - 1) // 2
                actual_connections_1 = np.sum(np.triu(C_total[np.ix_(cluster_indices_1, cluster_indices_1)], k=1))
                mean_connectivity_within_cluster1 = actual_connections_1 / total_possible_connections_1
                print(f"Средняя связность внутри первого кластера (нейроны 0-50): {round(mean_connectivity_within_cluster1, 2)}")
    
                cluster_indices_2 = np.where(np.array(cluster_labels) == 1)[0]
                total_possible_connections_between = len(cluster_indices_1) * len(cluster_indices_2)
                actual_connections_between = np.sum(C_total[np.ix_(cluster_indices_1, cluster_indices_2)])
                mean_connectivity_between_clusters = actual_connections_between / total_possible_connections_between
                print(f"Средняя связность между первым и вторым кластерами: {round(mean_connectivity_between_clusters, 2)}")
    

            avg_spikes_cluster2_tests = np.array(avg_spikes_cluster2_tests)
            mean_avg_spikes = np.mean(avg_spikes_cluster2_tests, axis=0)
            std_avg_spikes = np.std(avg_spikes_cluster2_tests, axis=0)

            if p_between not in spike_counts_second_cluster[I0_value][p_within]:
                spike_counts_second_cluster[I0_value][p_within][p_between] = []
            spike_counts_second_cluster[I0_value][p_within][p_between].append(mean_avg_spikes)
            
            if plot_avg_spikes:
                ax_avg_spikes = plotting_flags['ax_avg_spikes']
                ax_avg_spikes.plot(time_window_centers, mean_avg_spikes, label='Среднее число спайков\n во 2 кластере')
                ax_avg_spikes.fill_between(time_window_centers, mean_avg_spikes - std_avg_spikes,
                                     mean_avg_spikes + std_avg_spikes, alpha=0.3, label='Стандартное отклонение')
                ax_avg_spikes.set_xlabel('Время [ms]')
                ax_avg_spikes.set_ylabel('Среднее число спайков на нейрон')
                ax_avg_spikes.legend()
                ax_avg_spikes.set_title('Среднее число спайков\n во 2 кластере', fontsize=16)
    
            if fig is not None:
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                images.append(imageio.imread(buf))
                plt.close(fig)
            else:
                # Если фигура не создавалась, пропускаем сохранение изображения
                pass
    
        # Сохранение GIF-файла с уникальным именем
        if images:
            imageio.mimsave(f'gif_exc_I0_{I0_value}_p_within_{p_within:.2f}.gif', images, duration=3000, loop=0)
    
    print(f"Симуляции для I0 = {I0_value} мВ завершены.")

    # Построение графика зависимости среднего числа спайков во втором кластере от p_within и p_between
    fig_spike_dependency, ax_spike_dep = plt.subplots(figsize=(10, 6))
    
    for p_within in p_within_values:
        p_between_list = sorted(spike_counts_second_cluster[I0_value][p_within].keys())
        avg_spike_counts = []
        std_spike_counts = []
        for p_between in p_between_list:
            spike_counts = spike_counts_second_cluster[I0_value][p_within][p_between]
            avg_spike = np.mean(spike_counts)
            std_spike = np.std(spike_counts)
            avg_spike_counts.append(avg_spike)
            std_spike_counts.append(std_spike)
        ax_spike_dep.errorbar(p_between_list, avg_spike_counts, yerr=std_spike_counts, marker='o', label=f'p_within={p_within:.1f}')
    
    ax_spike_dep.set_xlabel('p_between', fontsize=14)
    ax_spike_dep.set_ylabel('Среднее число спайков во втором кластере', fontsize=14)
    ax_spike_dep.set_title(f'Зависимость среднего числа спайков во втором кластере от p_within и p_between\nI0={I0_value} мВ', fontsize=16)
    ax_spike_dep.legend(title='p_within', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=12)
    
    ax_spike_dep.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    plt.savefig(f'average_spike_dependency_in_2_cluster_ext_I0_{I0_value}mV.png')
    plt.close(fig_spike_dependency)
    
print("Генерация графиков завершена.")
