import matplotlib.pyplot as plt
import numpy as np
import csv
import io
import imageio
from brian2 import *
from centrality import GraphCentrality
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_spikes(ax_spikes, spike_times, spike_indices, time_window_size, t_range, sim_time, oscillation_frequency, use_stdp, measure_name):
    ax_spikes.scatter(spike_times, spike_indices, marker='|')

    # Highlight neurons with current applied based on centrality
    # highlighted_neurons = np.unique(centrality)
    # for neuron in highlighted_neurons:
    #     ax_spikes.scatter(spike_times[spike_indices == neuron], spike_indices[spike_indices == neuron], 
    #                       color='red', s=5, label=f'Neuron {neuron}')

    step_size = time_window_size
    total_time_ms = sim_time / ms
    time_steps = np.arange(0, total_time_ms + step_size, step_size)
    for t in time_steps:
        ax_spikes.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)

    ax_spikes.set_xlim(t_range)
    ax_spikes.set_xlabel("t [ms]")
    ax_spikes.set_ylabel("Neuron index")
    ax_spikes.set_title(f"Spike Raster Plot\nFrequency={oscillation_frequency} Hz, STDP={'On' if use_stdp else 'Off'}, Window={time_window_size} ms, {measure_name}", fontsize=16)

def plot_rates(ax_rates, oscillation_frequency, use_stdp, rate_monitor, t_range, rate_range, rate_tick_step):
    ax_rates.set_title(f'Spike Rate neu.0-50\nFrequency={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}', fontsize=16)
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz, label='Group 1 (0-50)')
    ax_rates.set_xlim(t_range)
    ax_rates.set_xlabel("t [ms]")
    ax_rates.set_yticks(
        np.arange(
            rate_range[0], rate_range[1] + rate_tick_step, rate_tick_step
        )
    )

def plot_rates2(ax_rates2, oscillation_frequency, use_stdp, rate_monitor2, t_range):
    ax_rates2.set_title(f'Spike Rate neu.50-100\nFrequency={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}', fontsize=16)
    ax_rates2.plot(rate_monitor2.t / ms, rate_monitor2.rate / Hz, label='Group 2 (50-100)')
    ax_rates2.set_xlim(t_range)
    ax_rates2.set_xlabel("t [ms]")

def plot_trace(ax_trace, trace, oscillation_frequency, use_stdp):
    time_points_trace = trace.t / ms
    offset = 10
    for neuron_index in range(0, 1, 1):
        ax_trace.plot(time_points_trace, trace.v[neuron_index] / mV + neuron_index * offset, label=f'Neuron {neuron_index}')
    
    ax_trace.set_xlabel("t [ms]")
    ax_trace.set_ylabel("Membrane potential [mV] + offset")
    ax_trace.legend(loc='upper right', ncol=2, fontsize='small')
    ax_trace.set_title(f'Membrane Potentials\nFrequency={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}', fontsize=16)
    

def plot_psd(rate_monitor, oscillation_frequency, use_stdp, ax_psd):
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


def plot_psd2(rate_monitor2, oscillation_frequency, use_stdp, ax_psd2):
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


def plot_connectivity(n_neurons, exc_synapses, inhib_synapses, connectivity2, use_stdp, centrality):
    W = np.zeros((n_neurons, n_neurons))
    W[exc_synapses.i[:], exc_synapses.j[:]] = exc_synapses.w[:]
    W[inhib_synapses.i[:], inhib_synapses.j[:]] = inhib_synapses.w[:]

    # Highlight neurons with current applied based on centrality
    highlighted_neurons = np.unique(centrality)
    for neuron in highlighted_neurons:
        connectivity2.plot([neuron], [neuron], 'ro', markersize=1, label=f'Neuron {neuron}')

    connectivity2.set_title(f'Weight Matrix\nSTDP={"On" if use_stdp else "Off"}', fontsize=16)
    connectivity2.matshow(W, cmap='viridis')


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
        graph_centrality = GraphCentrality(C_total)

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
        top_k = int(p_input * len(cluster_list))
        sorted_neurons_cluster = sorted(
            measure_values_cluster,
            key=lambda n: measure_values_cluster[n],
            reverse=True
        )
        top_neurons = sorted_neurons_cluster[:top_k]

    else:
        cluster1_indices = np.arange(0, cluster_nodes)
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

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_spike_dependency(
    spike_counts_second_cluster,
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
    Построение графика зависимости среднего числа спайков во втором кластере
    от p_within и p_between, а также сохранение данных в CSV.
    """
    fig_spike_dependency, ax_spike_dep = plt.subplots(figsize=(10, 6))

    # Подготовим список для CSV
    data_for_csv = []
    data_for_csv.append(["p_within", "p_between", "avg_spike_counts", "std_spike_counts"])

    # Перебираем значения p_within
    for p_within in p_within_values:
        p_within_str = f"{p_within:.2f}"
        # Собираем список p_between для данного p_within
        p_between_list = sorted(spike_counts_second_cluster[I0_value][p_within_str].keys())
        avg_spike_counts = []
        std_spike_counts = []

        # Для каждого p_between вычисляем среднее и std
        for p_between in p_between_list:
            # spike_counts — это список массивов из разных прогонов (num_tests)
            spike_counts = spike_counts_second_cluster[I0_value][p_within_str][p_between]
            if len(spike_counts) > 0:
                # Плоский список из всех массивов
                flat_spike_counts = np.concatenate(spike_counts)
                avg_spike = np.mean(flat_spike_counts)
                std_spike = np.std(flat_spike_counts) if len(flat_spike_counts) > 1 else 0
            else:
                avg_spike = 0
                std_spike = 0

            avg_spike_counts.append(avg_spike)
            std_spike_counts.append(std_spike)
            data_for_csv.append([p_within, p_between, avg_spike, std_spike])

        # Строим ошибочные столбики (errorbar)
        ax_spike_dep.errorbar(
            p_between_list,
            avg_spike_counts,
            yerr=std_spike_counts,
            marker='o',
            label=f'p_within={p_within_str}'
        )

    ax_spike_dep.set_xlabel('p_between', fontsize=14)
    ax_spike_dep.set_ylabel('Среднее число спайков во втором кластере', fontsize=14)
    ax_spike_dep.set_title(
        f'Зависимость среднего числа спайков во втором кластере от p_within и p_between\n'
        f'I0={I0_value}pA, freq={oscillation_frequency}Hz, STDP={"On" if use_stdp else "Off"}, '
        f'Time={current_time}ms, Bin={time_window_size}ms_{measure_name}',
        fontsize=16
    )
    ax_spike_dep.legend(
        title='p_within',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=12,
        title_fontsize=12
    )
    ax_spike_dep.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    # Сохраняем рисунок
    fig_filename = os.path.join(
        directory_path,
        f'avg_spike_dependency_I0_{I0_value}pA_freq_{oscillation_frequency}Hz_'
        f'STDP_{"On" if use_stdp else "Off"}_Time_{current_time}ms_Bin_{time_window_size}ms_{measure_name}.png'
    )
    plt.savefig(fig_filename)
    plt.close(fig_spike_dependency)
    # print(f"График зависимости среднего числа спайков сохранён: {fig_filename}")

    # Сохраняем данные в CSV
    csv_filename = os.path.join(
        directory_path,
        f'avg_spike_dependency_data_I0_{I0_value}pA_freq_{oscillation_frequency}Hz_'
        f'STDP_{"On" if use_stdp else "Off"}_Time_{current_time}ms_Bin_{time_window_size}ms_{measure_name}.csv'
    )
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for row in data_for_csv:
            writer.writerow(row)
    # print(f"CSV-файл со средними значениями спайков сохранён: {csv_filename}")


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
        ax_3d.set_zlim(0,30)
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


import os
import numpy as np
import matplotlib.pyplot as plt

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
    Строит 3D-график (p_input, p_between, avg_spikes) для КАЖДОГО p_within,
    дополнительно отображая коридор отклонений по Z (mean ± std).

    Параметры
    ----------
    spike_counts_second_cluster_for_input : dict
        Словарь со средними значениями спайков:
        spike_counts_second_cluster_for_input[I0_value][p_within][p_input][p_between] = mean_spikes
        
    spike_counts_second_cluster : dict
        Словарь с «сырыми» данными (или разбивкой по повторам):
        spike_counts_second_cluster[I0_value][p_within][p_between] = [array([...]), ...],
        где индекс списка соответствует индексу p_input, а внутри array(...) могут быть повторные измерения.

    I0_value : float
        Текущее значение тока, для которого строим графики.

    oscillation_frequency : float
        Частота внешней стимуляции (Гц).

    use_stdp : bool
        Флаг включения STDP.

    current_time : float
        Текущее время (мс), используемое в заголовке графиков.

    time_window_size : float
        Размер временного окна (мс).

    directory_path : str
        Папка для сохранения результатов.

    measure_name : str
        Дополнительное имя метрики (добавляется в название файла).
    """
    print("spike_counts_second_cluster_for_input", spike_counts_second_cluster_for_input)
    print("spike_counts_second_cluster", spike_counts_second_cluster)
    # Проверяем наличие данных для заданного I0_value
    data_for_v0 = spike_counts_second_cluster_for_input.get(I0_value, {})
    if not data_for_v0:
        print(f"No data found for I0={I0_value}pA.")
        return

    for p_within_str, dict_pinput in data_for_v0.items():
        if not dict_pinput:
            print(f"Нет данных для p_within={p_within_str}. Пропуск.")
            continue

        # Собираем уникальные p_input (как строки) и преобразуем к float
        p_input_list = sorted(dict_pinput.keys())
        try:
            p_input_list_float = [float(p) for p in p_input_list]
        except ValueError as e:
            print(f"Ошибка при преобразовании p_input к float для p_within={p_within_str}: {e}")
            continue

        # Определяем список p_between из первого p_input
        first_p_input_str = p_input_list[0]
        p_between_keys = dict_pinput[first_p_input_str].keys()
        try:
            p_between_list = sorted([float(p) for p in p_between_keys])
        except ValueError as e:
            print(f"Ошибка при преобразовании p_between к float для p_within={p_within_str}: {e}")
            continue

        if not p_between_list:
            print(f"Нет p_between значений для p_within={p_within_str}. Пропуск.")
            continue

        # Создаём 2D-массивы Z (средние значения) и Z_std (стандартные отклонения)
        Z = np.zeros((len(p_input_list_float), len(p_between_list)))
        Z_std = np.zeros_like(Z)

        for i, p_inp in enumerate(p_input_list_float):
            p_inp_str = f"{p_inp:.2f}"
            for j, p_btw in enumerate(p_between_list):
                mean_spikes = dict_pinput[p_inp_str].get(p_btw, 0.0)
                # Переводим в float среднее
                try:
                    Z[i, j] = float(mean_spikes)
                except (ValueError, TypeError):
                    Z[i, j] = 0.0

                # Извлекаем «сырые» данные, чтобы посчитать std
                stdev_value = 0.0
                # Проверяем, что в spike_counts_second_cluster есть нужные ключи
                # p_btw здесь float, а в словаре — может быть тоже float (или мы приводим к тому же виду).
                if (I0_value in spike_counts_second_cluster and 
                    p_within_str in spike_counts_second_cluster[I0_value] and
                    p_btw in spike_counts_second_cluster[I0_value][p_within_str]):
                    
                    # Достаем список по ключу p_btw
                    list_of_arrays = spike_counts_second_cluster[I0_value][p_within_str][p_btw]
                    
                    # Проверяем, что индекс i (для p_input) не выходит за границы
                    if 0 <= i < len(list_of_arrays):
                        # В list_of_arrays[i] лежит массив данных (например, несколько повторов)
                        data_arr = list_of_arrays[i]
                        # Если это не массив/список, обернём в np.array
                        data_arr = np.array(data_arr, dtype=float)
                        # Считаем стандартное отклонение
                        stdev_value = np.std(data_arr)
                
                Z_std[i, j] = stdev_value

        # Готовим координатные сетки для 3D
        p_input_mesh, p_between_mesh = np.meshgrid(p_input_list_float, p_between_list, indexing='ij')

        # Проверка на финитные значения в Z
        if not np.isfinite(Z).all():
            print(f"Некоторые значения в Z не являются конечными для p_within={p_within_str}. Пропуск построения графика.")
            continue

        # Создаём фигуру
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Главная поверхность (средние значения)
        surf = ax.plot_surface(
            p_input_mesh,    # X
            p_between_mesh,  # Y
            Z,               # Z (mean)
            cmap='viridis',
            edgecolor='none',
            alpha=0.9
        )
        
        # Добавляем «коридор» стандартного отклонения: mean ± std
        Z_plus = Z + Z_std
        Z_minus = Z - Z_std
        # print("Z", Z)
        # print("Z_plus", Z_plus)
        # print("Z_minus", Z_minus)
        # Чтобы не загромождать график, используем полупрозрачный однотонный цвет
        ax.plot_surface(
            p_input_mesh,
            p_between_mesh,
            Z_plus,
            color='red',
            alpha=0.3,
            edgecolor='none'
        )
        ax.plot_surface(
            p_input_mesh,
            p_between_mesh,
            Z_minus,
            color='blue',
            alpha=0.3,
            edgecolor='none'
        )

        ax.set_title(
            f'3D: p_input vs p_between vs avg_spikes\n'
            f'I0={I0_value}pA, freq={oscillation_frequency}Hz, STDP={"On" if use_stdp else "Off"}, '
            f'p_within={p_within_str}, Time={current_time}ms, Bin={time_window_size}ms, {measure_name}',
            fontsize=14
        )
        ax.set_zlim(0, 30)
        ax.set_xlabel('p_input', fontsize=12)
        ax.set_ylabel('p_between', fontsize=12)
        ax.set_zlabel('avg_spikes', fontsize=12)
        
        fig.colorbar(surf, shrink=0.5, aspect=5, label="Mean spikes")

        # Сохраняем результат
        filename = os.path.join(
            directory_path,
            f'3D_pinput_between_spikes_STD_V0_{I0_value}_freq_{oscillation_frequency}Hz_'
            f'STDP_{"On" if use_stdp else "Off"}_p_within_{p_within_str}_'
            f'Time_{current_time}ms_Bin_{time_window_size}ms_{measure_name}.png'
        )
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        # print(f"[plot_pinput_between_avg_spikes_with_std] График сохранён: {filename}")

