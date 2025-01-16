import os
import io
import csv
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from brian2 import *
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Импорт из ваших модулей
from connectivity_models import clcheck
from simulation import sim
from plotting_and_saving import (
    plot_avg_spike_dependency,
    plot_3d_spike_data,
    plot_pinput_between_avg_spikes
)

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
n_neurons = 100
num_of_clusters = 2
cluster_sizes = [50, 50]

# Основные диапазоны параметров
p_within_values = np.arange(0.7, 0.71, 0.1)  # [0.7, 0.8, 0.9, 1.0]
# Новый диапазон значений p_input (0.2, 0.4, 0.6, 0.8, 1.0)
p_input_values = np.arange(0.2, 1.01, 0.2)

num_tests = 1
J = 0.2
g = 6
epsilon = 0.1
nu_ext_over_nu_thr = 4
D = 1.5 * ms
rate_tick_step = 50
t_range = [0, 1000]
rate_range = [0, 200]
time_window_size = 100  # in ms

# Для осцилляций/стимулов
oscillation_frequencies = [10]
use_stdp_values = [True]
I0_values = [2000]

# Флаги для построения графиков
plot_spikes = True
plot_avg_spikes = False
plot_rates = False
plot_rates2 = False
plot_connectivity2 = True
plot_trace = True
plot_psd = False
plot_psd2 = False
plot_spectrogram = False

# -- Формируем вектор меток кластеров --
cluster_labels = []
for i in range(n_neurons):
    cluster_labels.append(clcheck(i, cluster_sizes))

# -- Список времён симуляции (в мс) --
simulation_times = [1000]

def check_intra_cluster_connectivity_change(C_old, C_new, cluster_labels):
    unique_clusters = set(cluster_labels)
    change_detected = False
    total_changes = 0
    for cluster_id in unique_clusters:
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        sub_old = C_old[np.ix_(indices, indices)]
        sub_new = C_new[np.ix_(indices, indices)]
        diff = sub_new - sub_old
        changed_elements = np.sum(np.abs(diff) > 1e-6)
        diff_norm = np.linalg.norm(diff)
        print(f"Кластер {cluster_id}: Норма разницы = {diff_norm:.6f}, Изменённых связей = {changed_elements}")
        if diff_norm > 1e-6:
            print(f"Внутрикластерная связность изменилась в кластере {cluster_id}!")
            change_detected = True
            total_changes += changed_elements
    print(f"Всего изменённых внутрикластерных связей: {total_changes}")
    return change_detected


def check_inter_cluster_connectivity_change(C_old, C_new, cluster_labels):
    unique_clusters = sorted(set(cluster_labels))
    change_detected = False

    # Проверяем все пары кластеров (межкластерные связи)
    for i, cluster_id_1 in enumerate(unique_clusters):
        for cluster_id_2 in unique_clusters[i+1:]:
            # Индексы нейронов в каждом кластере
            indices_1 = [idx for idx, label in enumerate(cluster_labels) if label == cluster_id_1]
            indices_2 = [idx for idx, label in enumerate(cluster_labels) if label == cluster_id_2]

            # Вырезаем межкластерные части матрицы
            sub_old = C_old[np.ix_(indices_1, indices_2)]
            sub_new = C_new[np.ix_(indices_1, indices_2)]

            # Вычисляем разницу
            diff = sub_new - sub_old
            diff_norm = np.linalg.norm(diff)

            print(f"Межкластерная связь между кластерами {cluster_id_1} и {cluster_id_2}: Норма разницы = {diff_norm:.6f}")

            if diff_norm > 1e-6:
                print(f"Межкластерная связность изменилась между кластерами {cluster_id_1} и {cluster_id_2}!")
                change_detected = True

    return change_detected

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
                    p_within = round(p_within, 2)
                    p_within_str = f"{p_within:.2f}"

                    print(f"\n--- p_within = {p_within_str} ---")

                    # Для нового p_within — подготовим ячейки в spike_counts_second_cluster_for_input
                    for p_input in p_input_values:
                        p_input = round(p_input, 2)
                        p_input_str = f"{p_input:.2f}"
                        spike_counts_second_cluster_for_input[I0_value][p_within_str][p_input_str] = {}

                        # Список для хранения кадров (GIF), если нужно
                        images = []
                        C_total_prev = None
                        p_within_prev = None
                        p_between_prev = None
                        p_between_values = np.arange(0.05, p_within - 0.04, 0.1)

                        # -- Цикл по p_between --
                        for p_between in p_between_values:
                            avg_window_avg_neuron_spikes_cluster2_tests = []
                            p_between = round(p_between, 2)

                            print(f"\nProcessing p_between = {p_between}")

                            # При необходимости готовим фигуру для GIF
                            if any([
                                plot_spikes, plot_avg_spikes, plot_rates, plot_rates2,
                                plot_connectivity2, plot_trace, plot_psd, plot_psd2, plot_spectrogram
                            ]):
                                fig = plt.figure(figsize=(12, 10))
                                fig.suptitle(
                                    f'I0={I0_value} pA, p_within={p_within_str}, '
                                    f'p_input={p_input:.1f}, p_between={p_between}, '
                                    f'g={g}, epsilon={epsilon}, nu_ext_over_nu_thr={nu_ext_over_nu_thr}, '
                                    f'D={D}, Time={current_time} ms'
                                )
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
                            if fig is not None:
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

                            # -- Запускаем несколько тестов (num_tests) --
                            for test_num in range(num_tests):
                                print(f'\nI0={I0_value} pA, p_within={p_within_str}, '
                                      f'p_input={p_input:.1f}, p_between={p_between}, '
                                      f'tест={test_num + 1}, Time={current_time} ms')

                                # Очищаем оси на повторных прогонах
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

                                # -- Вызов функции симуляции --
                                (avg_neuron_spikes_cluster2_list,
                                 time_window_centers,
                                 C_total,
                                 spike_indices) = sim(
                                    p_within,
                                    p_between,
                                    g,
                                    nu_ext_over_nu_thr,
                                    J,
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
                                    p_input=p_input
                                )

                                # Считаем среднее по окнам для 2-го кластера
                                if avg_neuron_spikes_cluster2_list is not None and len(avg_neuron_spikes_cluster2_list) > 0:
                                    avg_window_val = np.mean(avg_neuron_spikes_cluster2_list)
                                else:
                                    avg_window_val = 0
                                avg_window_avg_neuron_spikes_cluster2_tests.append(avg_window_val)

                                # -- Проверяем изменение внутрикластерной связности --
                                if C_total_prev is not None:
                                    # Вычисляем норму разницы для отладки
                                    diff_norm = np.linalg.norm(C_total - C_total_prev)
                                    print(f"Норма разницы в C_total = {diff_norm:.6f}")

                                    # -- Проверяем изменение внутрикластерной связности --
                                    changed_intra = check_intra_cluster_connectivity_change(
                                        C_total_prev, C_total, cluster_labels
                                    )
                                    if changed_intra:
                                        print("Внутрикластерная связность изменилась!")
                                    else:
                                        print("Внутрикластерная связность НЕ изменилась!")

                                    # -- Проверяем изменение межкластерной связности --
                                    changed_inter = check_inter_cluster_connectivity_change(
                                        C_total_prev, C_total, cluster_labels
                                    )
                                    if changed_inter:
                                        print("Межкластерная связность изменилась!")
                                    else:
                                        print("Межкластерная связность НЕ изменилась!")

                                else:
                                    print("C_total_prev не определён, пропуск проверки изменений.")


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

                            detailed_spike_data_for_3d[I0_value][p_within_str][p_between] = {
                                "time": time_for_3d_time if time_for_3d_time is not None else np.array([]),
                                "spikes_list": spikes_for_3d_time if spikes_for_3d_time is not None else []
                            }
                            # -- По окончании всех тестов --
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

                            # Сохраняем кадр в GIF
                            if fig is not None:
                                plt.tight_layout()
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)
                                images.append(imageio.imread(buf))
                                plt.close(fig)

                        # Конец цикла по p_between
                        # -- Если хотим общий GIF по всем p_between для данного (p_within, p_input), сохраняем --
                        if images:
                            gif_filename = (
                                f'{directory_path}/gif_exc_I0_{I0_value}freq_{oscillation_frequency}_'
                                f'STDP_{"On" if use_stdp else "Off"}_p_within_{p_within_str}_'
                                f'p_input_{p_input_str}_Time_{current_time}ms_Bin_{time_window_size}ms.gif'
                            )
                            imageio.mimsave(gif_filename, images, duration=2000, loop=0)  # duration в секундах

                # Конец циклов по p_within и p_input

                print(f"\nСимуляции для I0 = {I0_value} pA, Time={current_time} ms завершены.")

                # -- Строим «старые» графики --
                plot_avg_spike_dependency(
                    spike_counts_second_cluster,
                    p_within_values,
                    I0_value,
                    oscillation_frequency,
                    use_stdp,
                    current_time,
                    time_window_size,
                    directory_path=directory_path
                )
                plot_3d_spike_data(
                    detailed_spike_data_for_3d,
                    p_within_values,
                    I0_value,
                    oscillation_frequency,
                    use_stdp,
                    current_time,
                    time_window_size,
                    directory_path=directory_path
                )

                # *** Строим НОВЫЙ 3D-график p_input vs p_between vs avg_spikes для каждого p_within ***
                plot_pinput_between_avg_spikes(
                    spike_counts_second_cluster_for_input,
                    I0_value,
                    oscillation_frequency,
                    use_stdp,
                    current_time,
                    time_window_size,
                    directory_path=directory_path
                )

# Конец основного цикла по времени симуляции

print("\nГенерация графиков завершена.")
