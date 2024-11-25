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
                        # Для направленных связей выбираем наличие связи с вероятностью p_within
                        sbm[i, j] = np.random.choice([0, 1], p=[1 - p_within, p_within])
                    else:
                        # Для ненаправленных связей устанавливаем симметричную матрицу
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
        # Получаем индексы нейронов в текущем кластере
        cluster_indices = np.where(np.array(cluster_labels) == cluster_idx)[0]
        total_possible_connections = len(cluster_indices) * (len(cluster_indices) - 1) // 2  # Максимальное количество связей
        current_connections = np.sum(np.triu(C_intra[np.ix_(cluster_indices, cluster_indices)], k=1))  # Текущее количество связей
        # np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1) -> [[1,2,3],[4,5,6],[0,8,9],[0,0,12]] - нули ниже 1 диагонали
        # a[np.ix_([1,3],[2,5])] -> [[a[1,2] a[1,5]], [a[3,2] a[3,5]]]
        desired_connections = int(p_within * total_possible_connections)  # Желаемое количество связей
        connections_to_add = desired_connections - int(current_connections)
        if connections_to_add > 0:
            # Находим возможные пары для добавления связей
            possible_pairs = [(i, j) for idx_i, i in enumerate(cluster_indices) 
                              for j in cluster_indices[idx_i+1:] if C_intra[i, j] == 0]
            if len(possible_pairs) == 0:
                continue  # Если нет доступных пар, пропускаем
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
            # Получаем индексы нейронов в двух разных кластерах
            cluster_a_indices = np.where(np.array(cluster_labels) == clusters[idx_a])[0]
            cluster_b_indices = np.where(np.array(cluster_labels) == clusters[idx_b])[0]
            total_possible_connections = len(cluster_a_indices) * len(cluster_b_indices)  # Максимальное количество межкластерных связей
            current_connections = np.sum(C_between[np.ix_(cluster_a_indices, cluster_b_indices)])  # Текущее количество связей
            # a[np.ix_([1,3],[2,5])] -> [[a[1,2] a[1,5]], [a[3,2] a[3,5]]]
            desired_connections = int(p_between * total_possible_connections)  # Желаемое количество связей
            connections_to_add = desired_connections - int(current_connections)
            if connections_to_add > 0:
                # Находим возможные пары для добавления связей
                possible_pairs = [(i, j) for i in cluster_a_indices 
                                  for j in cluster_b_indices if C_between[i, j] == 0]
                if len(possible_pairs) == 0:
                    continue  # Если нет доступных пар, пропускаем
                num_new_connections = min(connections_to_add, len(possible_pairs))
                new_connections = random.sample(possible_pairs, num_new_connections)
                for i, j in new_connections:
                    C_between[i, j] = 1
                    C_between[j, i] = 1
    return C_between

# Основная функция симуляции нейронной сети
def sim(p_within, p_between, g, nu_ext_over_nu_thr, J, sim_time, ax_spikes, ax_rates, ax_rates2, ax_trace,
        rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes, C_total_prev=None, p_within_prev=None, p_between_prev=None):
    start_scope()  # Начало нового контекста симуляции
    
    n_neurons = len(cluster_labels)  # Общее количество нейронов
    
    # Инициализация или обновление матриц связности
    if C_total_prev is None:
        # Первоначальная генерация внутрикластерных и межкластерных связей с низкими вероятностями
        C_intra = gensbm2(n_neurons, cluster_labels, 0.01, cluster_sizes, directed=False)
        C_between = generate_constant_inter_cluster(n_neurons, cluster_labels, 0.01, directed=False)
    else:
        # Копирование предыдущих матриц связности
        C_intra = C_total_prev.copy()
        C_between = C_total_prev.copy()
    
    # Обновление внутрикластерных связей
    C_intra = update_intra_cluster_connectivity(C_intra, cluster_labels, p_within)
    # Обновление межкластерных связей
    C_between = update_inter_cluster_connectivity(C_between, cluster_labels, p_between)
    
    # Объединение матриц связности для получения полной матрицы связности
    C_total = np.maximum(C_intra, C_between)
    
    N = len(C_total)  # Общее количество нейронов
    N_E = N / 2  # 50% из них возбуждающие нейроны
    
    # Определение параметров нейронной модели
    tau = 20 * ms  # Временная константа
    theta = 20 * mV  # Порог активации
    V_r = 10 * mV  # Потенциал сброса
    tau_rp = 2 * ms  # Временная константа рефрактерного периода
    
    J = J * mV  # Вес синапса
    D = 1.5 * ms  # Задержка синапса
    
    C_E = p_within * N_E  # Количество внешних связей
    C_ext = int(C_E)
    nu_thr = theta / (J * C_E * tau)  # Пороговая частота
    
    defaultclock.dt = 0.1 * ms  # Шаг интегрирования
    
    # Определение группы нейронов с их динамикой
    neurons = NeuronGroup(N,
        '''
        dv/dt = (-v + I_ext)/tau : volt (unless refractory)
        I_ext = I0 * sin(2 * pi * f * t + phi) : volt
        I0 : volt
        f : Hz
        phi : 1  # Фаза в радианах
        ''',
        threshold="v > theta",  # Условие спайка
        reset="v = V_r",  # Сброс потенциала после спайка
        refractory=tau_rp,  # Рефрактерный период
        method="euler",  # Метод интегрирования
    )
    
    # Установка начальных параметров для всех нейронов
    neurons.f = 0 * Hz
    neurons.I0 = 0 * volt
    neurons.phi = 0
    
    # Установка внешнего тока и частоты для первой группы нейронов (0-50)
    neurons.f[0:50] = 10 * Hz  # Частота внешнего сигнала
    neurons.I0[0:50] = 10 * mV  # Амплитуда внешнего тока
    neurons.phi[0:50] = 0  # Фаза внешнего сигнала
    
    # Определение уравнений STDP (Spike-Timing-Dependent Plasticity) для синапсов
    stdp_eqs = '''
    dApre/dt = -Apre / tau_stdp : 1 (event-driven)
    dApost/dt = -Apost / tau_stdp : 1 (event-driven)
    w : 1
    '''
    
    tau_stdp = 20 * ms  # Временная константа STDP
    Aplus = 0.005  # Увеличение синаптического веса
    Aminus = 0.005  # Уменьшение синаптического веса
    
    # Создание всех возможных возбуждающих синапсов между нейронами
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
    
    # Подключение всех возможных возбуждающих синапсов, исключая автосинапсы
    exc_synapses.connect(condition='i != j')
    
    # Обновление синаптических весов в соответствии с матрицей связности C_total
    sources, targets = C_total.nonzero()  # Получение индексов существующих связей
    exc_synapses.w[:, :] = 0  # Обнуление всех весов
    for idx in range(len(sources)):
        i = sources[idx]
        j = targets[idx]
        # Поиск соответствующего синапса и установка веса
        syn_idx = np.where((exc_synapses.i == i) & (exc_synapses.j == j))[0]
        if len(syn_idx) > 0:
            exc_synapses.w[syn_idx[0]] = 1  # Установка веса на 1 для существующей связи
    
    # Определение параметров внешнего пуассоновского ввода для первой группы нейронов
    nu_ext_group1 = nu_ext_over_nu_thr * nu_thr * 0.2
    
    external_poisson_input_1 = PoissonInput(
        target=neurons,  # Целевая группа нейронов
        target_var="v",  # Переменная, к которой применяется ввод
        N=C_ext,  # Количество входов
        rate=nu_ext_group1,  # Частота пуассоновского ввода
        weight=J  # Вес каждого входного импульса
    )
    
    # Создание мониторов для отслеживания активности нейронов
    rate_monitor = PopulationRateMonitor(neurons[:50])  # Мониторинг скорости для первых 50 нейронов
    rate_monitor2 = PopulationRateMonitor(neurons[50:])  # Мониторинг скорости для нейронов 50-100
    spike_monitor = SpikeMonitor(neurons[:100])  # Мониторинг спайков всех 100 нейронов
    trace = StateMonitor(neurons, 'v', record=True)  # Мониторинг мембранного потенциала
    
    run(sim_time)  # Запуск симуляции
    
    # Анализ спайков для последующего анализа
    spike_times = spike_monitor.t / ms  # Время спайков в мс
    spike_indices = spike_monitor.i  # Индексы нейронов, которые спайкнули

    bin_size = 100  # Размер окна в мс
    max_time = int(np.max(spike_times)) + bin_size
    bins = np.arange(0, max_time, bin_size) # кол-во окон

    spike_times_list = []
    spike_indices_list = []

    avg_spikes_cluster2_list = []  # Список средних чисел спайков во втором кластере

    for i in range(len(bins) - 1):
        start_time = bins[i]
        end_time = bins[i + 1]
        mask = (spike_times >= start_time) & (spike_times < end_time)
        window_spike_times = spike_times[mask]
        window_spike_indices = spike_indices[mask]

        # Определение, какие спайки принадлежат второй группе нейронов
        group2_mask = (window_spike_indices >= 50) & (window_spike_indices < 100)

        group2_spikes = np.sum(group2_mask)  # Количество спайков во второй группе

        avg_spikes_cluster2 = group2_spikes / 50  # Среднее число спайков на нейрон во втором кластере
        avg_spikes_cluster2_list.append(avg_spikes_cluster2)

        spike_times_list.extend(window_spike_times)
        spike_indices_list.extend(window_spike_indices)

    time_window_centers = (bins[:-1] + bins[1:]) / 2  # Центры окон времени

    spike_times_array = np.array(spike_times_list)
    spike_indices_array = np.array(spike_indices_list)

    # Построение графика спайковой растровой диаграммы
    ax_spikes.scatter(spike_times_array, spike_indices_array, marker='|')

    # Добавление вертикальных линий каждые 100 мс для визуального разделения окон
    step_size = 100  # Шаг в мс
    total_time_ms = sim_time / ms  # Общее время симуляции в мс
    time_steps = np.arange(0, total_time_ms + step_size, step_size)
    for t in time_steps:
        ax_spikes.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)

    # Настройка осей и заголовка для растровой диаграммы
    ax_spikes.set_xlim(t_range)
    ax_spikes.set_xlabel("t [ms]")
    ax_spikes.set_ylabel("Номер нейрона")
    ax_spikes.set_title("Спайковая растровая диаграмма", fontsize=16)

    ax_rates.set_title('Частота спайков\n neu.0-50', fontsize=16)
    ax_rates2.set_title('Частота спайков\n neu.50-100', fontsize=16)
    # Построение графиков скоростей популяций нейронов
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz, label='Группа 1 (0-50)')
    ax_rates2.plot(rate_monitor2.t / ms, rate_monitor2.rate / Hz, label='Группа 2 (50-100)')
    ax_spikes.set_xlim(t_range)
    ax_rates.set_xlim(t_range)
    ax_rates.set_xlabel("t [ms]")

    # Построение мембранного потенциала для выборочных нейронов
    time_points_trace = trace.t / ms
    offset = 10  # Сдвиг для визуализации нескольких сигналов на одном графике

    for neuron_index in range(0, n_neurons, 10):
        ax_trace.plot(time_points_trace, trace.v[neuron_index] / mV + neuron_index * offset, label=f'Neuron {neuron_index}')

    # Настройка осей и легенды для графика мембранного потенциала
    ax_trace.set_xlabel("t [ms]")
    ax_trace.set_ylabel("Мембранный потенциал [mV] + сдвиг")
    ax_trace.legend(loc='upper right', ncol=2, fontsize='small')

    # Настройка делений оси Y для графиков скоростей
    ax_rates.set_yticks(
        np.arange(
            rate_range[0], rate_range[1] + rate_tick_step, rate_tick_step
        )
    )

    # Построение PSD (Power Spectral Density) и спектрограммы для первой группы нейронов
    rate = rate_monitor.rate / Hz - np.mean(rate_monitor.rate / Hz)  # Центрирование данных
    from numpy.fft import rfft, rfftfreq

    N = len(rate_monitor.t)  # Количество выборок
    sampling_rate = 10000  # Частота дискретизации (1/defaultclock.dt)
    max_point = int(N * 300 / sampling_rate)  # Максимальное количество точек для PSD
    x = rfftfreq(N, d=1 / sampling_rate)  # Частоты для PSD
    x = x[:max_point]
    yn = 2 * np.abs(rfft(rate)) / N  # Амплитудный спектр
    yn = yn[:max_point]

    ax_psd.set_title("PSD neu.0-50", fontsize=16)
    ax_psd.plot(x, yn, c='k', label='Функция')

    # Построение спектрограммы для первой группы нейронов
    yf = rfft(rate_monitor.rate / Hz)
    N_freq = len(yf)  # Количество частотных компонент
    xf = rfftfreq(len(rate_monitor.t), 1 / 10000)[:N_freq]  # Частоты для спектрограммы

    ax_spectrogram.plot(xf, np.abs(yf), label='0-50 нейронов')
    ax_spectrogram.set_xlim(0, 1000)  # Ограничение частот до 1000 Гц
    ax_spectrogram.legend()

    # Построение PSD и спектрограммы для второй группы нейронов
    rate = rate_monitor2.rate / Hz - np.mean(rate_monitor2.rate / Hz)  # Центрирование данных

    N = len(rate_monitor2.t)  # Количество выборок
    sampling_rate = 10000  # Частота дискретизации
    max_point = int(N * 300 / sampling_rate)
    x = rfftfreq(N, d=1 / sampling_rate)
    x = x[:max_point]
    yn = 2 * np.abs(rfft(rate)) / N
    yn = yn[:max_point]

    ax_psd2.set_title("PSD neu.50-100", fontsize=16)
    ax_psd2.plot(x, yn, c='k', label='Функция')

    yf = rfft(rate_monitor2.rate / Hz)
    N_freq2 = len(yf)  # Количество частотных компонент
    xf = rfftfreq(len(rate_monitor2.t), 1 / 10000)[:N_freq2]  # Частоты для спектрограммы

    ax_spectrogram.set_title('Глобальные частоты\n нейронов 1 и 2 кластера', fontsize=16)
    ax_spectrogram.plot(xf, np.abs(yf), label='50-100 нейронов')
    ax_spectrogram.legend()
    ax_spectrogram.set_xlim(0, 1000)

    # Визуализация матрицы связности
    # connectivity.matshow(C_total, cmap='viridis')
    W = np.zeros((n_neurons, n_neurons))
    W[exc_synapses.i[:], exc_synapses.j[:]] = exc_synapses.w[:]  # Создание матрицы весов
    connectivity2.set_title('Матрица весов', fontsize=16)
    connectivity2.matshow(W, cmap='viridis')  # Визуализация матрицы весов
    plt.subplots_adjust()  # Автоматическая настройка отступов между подграфиками

    # Возврат информации о среднем числе спайков, матрице связности и индексов спайков для дальнейшего анализа
    return avg_spikes_cluster2_list, time_window_centers, C_total, spike_indices

# Параметры симуляции
p_within_values = np.arange(0.1, 1.05, 0.1)  # Диапазон значений вероятности внутрикластерных связей
num_tests = 1  # Количество тестов для каждого значения связности
J = 0.1  # Вес синапса
g = 6  # Коэффициент торможения
time = 3000  # Время симуляции в мс
sim_time = time * ms  # Время симуляции с единицей измерения
epsilon = 0.1  # Дополнительный параметр (может быть использован в дальнейшем)
nu_ext_over_nu_thr = 4  # Отношение внешней частоты к пороговой
D = 1.5 * ms  # Задержка синапса
rate_tick_step = 50  # Шаг делений оси Y для графиков скоростей
t_range = [0, time]  # Диапазон времени для графиков
rate_range = [0, 200]  # Диапазон скоростей для графиков

# Инициализация параметров сети
n_neurons = 100  # Общее количество нейронов
num_of_clusters = 2  # Количество кластеров
cluster_sizes = [50, 50]  # Размеры кластеров

# Генерация меток кластеров для каждого нейрона
cluster_labels = []
for i in range(n_neurons):
    cluster_labels.append(clcheck(i, cluster_sizes))

# Инициализация структуры данных для хранения среднего числа спайков во втором кластере
# Структура: {p_within: {p_between: [спайки во втором кластере для каждого теста]}}
spike_counts_second_cluster = {p_within: {} for p_within in p_within_values}


# Основной цикл симуляции по p_within
for p_within_idx, p_within in enumerate(p_within_values):
    images = []  # Список для хранения изображений для GIF
    p_between_values = np.arange(0.05, p_within - 0.04, 0.05)  # Диапазон значений вероятности межкластерных связей
    # Инициализация предыдущих значений матриц связности и вероятностей
    C_total_prev = None
    p_within_prev = None
    p_between_prev = None
    for p_between in p_between_values:
        avg_spikes_cluster2_tests = []  # Инициализация списка для хранения результатов
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(f'p_within={round(p_within, 2)}, p_between={round(p_between, 2)}, g={g}, epsilon={epsilon}, nu_ext_over_nu_thr={nu_ext_over_nu_thr}, D={D}')
    
        # Создание сетки подграфиков
        gs = fig.add_gridspec(ncols=4, nrows=3)
        ax_spikes = fig.add_subplot(gs[0, :])  # Растровая диаграмма спайков
        ax_avg_spikes = fig.add_subplot(gs[1, 0])  # График среднего числа спайков во втором кластере
        ax_rates = fig.add_subplot(gs[1, 1])  # График скорости первой группы нейронов
        ax_rates2 = fig.add_subplot(gs[1, 2])  # График скорости второй группы нейронов
        connectivity2 = fig.add_subplot(gs[1, 3])  # Визуализация матрицы весов синапсов
        ax_trace = fig.add_subplot(gs[2, 0])  # График мембранного потенциала
        ax_psd = fig.add_subplot(gs[2, 1])  # PSD для первой группы нейронов
        ax_psd2 = fig.add_subplot(gs[2, 2])  # PSD для второй группы нейронов
        ax_spectrogram = fig.add_subplot(gs[2, 3])  # Спектрограмма

        for test_num in range(num_tests):
            print(f'p_within={round(p_within, 2)}, p_between={round(p_between, 2)}, test={test_num + 1}')
            # Перед запуском каждого теста очищаем оси, кроме оси среднего числа спайков
            if test_num > 0:
                ax_spikes.clear()
                ax_rates.clear()
                ax_rates2.clear()
                ax_trace.clear()
                ax_psd.clear()
                ax_psd2.clear()
                ax_spectrogram.clear()

            # Запуск симуляции
            avg_spikes_cluster2_list, time_window_centers, C_total, spike_indices = sim(
                round(p_within, 2), round(p_between, 2), g, nu_ext_over_nu_thr, J, sim_time, 
                ax_spikes, ax_rates, ax_rates2, ax_trace,
                rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes, 
                C_total_prev, p_within_prev, p_between_prev
            )
            avg_spikes_cluster2_tests.append(avg_spikes_cluster2_list)

            # Обновление предыдущих значений для следующего теста
            C_total_prev = C_total.copy()
            p_within_prev = p_within
            p_between_prev = p_between

            # Вычисление средней связности внутри первого кластера (нейроны 0-50)
            cluster_indices_1 = np.where(np.array(cluster_labels) == 0)[0]
            total_possible_connections_1 = len(cluster_indices_1) * (len(cluster_indices_1) - 1) // 2
            actual_connections_1 = np.sum(np.triu(C_total[np.ix_(cluster_indices_1, cluster_indices_1)], k=1))
            mean_connectivity_within_cluster1 = actual_connections_1 / total_possible_connections_1
            print(f"Средняя связность внутри первого кластера (нейроны 0-50): {round(mean_connectivity_within_cluster1, 2)}")

            # Вычисление средней связности между первым и вторым кластерами (нейроны 0-50 и 50-100)
            cluster_indices_2 = np.where(np.array(cluster_labels) == 1)[0]
            total_possible_connections_between = len(cluster_indices_1) * len(cluster_indices_2)
            actual_connections_between = np.sum(C_total[np.ix_(cluster_indices_1, cluster_indices_2)])
            mean_connectivity_between_clusters = actual_connections_between / total_possible_connections_between
            print(f"Средняя связность между первым и вторым кластерами: {round(mean_connectivity_between_clusters, 2)}")


            # Фильтрация индексов в диапазоне 50-100
            filtered_spikes = spike_indices[(spike_indices >= 50) & (spike_indices <= 100)]
            # Подсчет количества спайков для каждого уникального нейрона
            unique_neurons, counts = np.unique(filtered_spikes, return_counts=True)
            # Вычисление среднего числа спайков на нейрон
            average_spikes = np.mean(counts)
            print("Спайки в диапазоне 50-100:", np.sort(filtered_spikes))
            print("Среднее число спайков на нейрон:", average_spikes)

            if p_between not in spike_counts_second_cluster[p_within]:
                spike_counts_second_cluster[p_within][p_between] = []
            # # Добавляем количество спайков для текущего теста
            spike_counts_second_cluster[p_within][p_between].append(average_spikes)
    
        avg_spikes_cluster2_tests = np.array(avg_spikes_cluster2_tests)
        mean_avg_spikes = np.mean(avg_spikes_cluster2_tests, axis=0)
        std_avg_spikes = np.std(avg_spikes_cluster2_tests, axis=0)

        # Построение графика среднего числа спайков во втором кластере
        ax_avg_spikes.plot(time_window_centers, mean_avg_spikes, label='Среднее число спайков\n во 2 кластере')
        ax_avg_spikes.fill_between(time_window_centers, mean_avg_spikes - std_avg_spikes,
                             mean_avg_spikes + std_avg_spikes, alpha=0.3, label='Стандартное отклонение')
        ax_avg_spikes.set_xlabel('Время [ms]')
        ax_avg_spikes.set_ylabel('Среднее число спайков на нейрон')
        ax_avg_spikes.legend()
        ax_avg_spikes.set_title('Среднее число спайков\n во 2 кластере', fontsize=16)

        plt.tight_layout()  # Автоматическая настройка расположения подграфиков
        # Сохранение текущей фигуры в буфер для дальнейшего сохранения в GIF
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))  # Добавление изображения в список
        plt.close(fig)  # Закрытие фигуры для освобождения памяти

    # Сохранение списка изображений в GIF-файл с длительностью 3 секунды для каждого кадра
    imageio.mimsave(f'gif_exc_p_within_{p_within:.2f}.gif', images, duration=3000, loop=0)

print("Генерация GIF-файлов завершена.")

# Создадим отдельную фигуру для этих графиков
fig_spike_dependency, ax_spike_dep = plt.subplots(figsize=(10, 6))

# Проход по всем значениям p_within
for p_within in p_within_values:
    p_between_list = sorted(spike_counts_second_cluster[p_within].keys())
    avg_spike_counts = []
    for p_between in p_between_list:
        # Вычисляем среднее число спайков во втором кластере для данной комбинации p_within и p_between
        avg_spike = spike_counts_second_cluster[p_within][p_between]
        avg_spike_counts.append(avg_spike)
    # Построение кривой для текущего p_within
    ax_spike_dep.plot(p_between_list, avg_spike_counts, marker='o', label=f'p_within={p_within:.1f}')

# Настройка осей и заголовка
ax_spike_dep.set_xlabel('p_between', fontsize=14)  # Увеличен размер шрифта
ax_spike_dep.set_ylabel('Среднее число спайков во втором кластере', fontsize=14)  # Увеличен размер шрифта
ax_spike_dep.set_title('Зависимость среднего числа спайков во втором кластере от p_within и p_between', fontsize=16)  # Увеличен размер шрифта
ax_spike_dep.legend(title='p_within', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=12)  # Легенда с увеличенным шрифтом

# Увеличение размеров делений осей
ax_spike_dep.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# Сохранение графика зависимости спайков во втором кластере
plt.savefig('average_spike_dependency_in_2_cluster_ext.png')
plt.show()
