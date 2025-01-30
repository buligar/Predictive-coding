import numpy as np
from brian2 import *
import time
from connectivity_models import (
    gensbm2,
    generate_constant_inter_cluster,
    update_intra_cluster_connectivity,
    update_inter_cluster_connectivity
)
from plotting_and_saving import (
    plot_spikes,
    plot_rates, plot_rates2,
    plot_trace, plot_psd, plot_psd2,
    plot_spectrogram, plot_connectivity, print_centrality
)

def sim(p_within, p_between, g, nu_ext_over_nu_thr, J, J2, refractory_period, sim_time, plotting_flags,
        rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes,
        I0_value, oscillation_frequency, use_stdp, time_window_size,
        C_total_prev=None, p_within_prev=None, p_between_prev=None,
        p_input=None, measure_name=None, centrality=None):
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

    # -- Генерация/обновление матрицы связей --
    if C_total_prev is None:
        # генерируем "пустую" структуру и обновляем
        C_intra = gensbm2(n_neurons, cluster_labels, 0.01, cluster_sizes, directed=False)
        C_between = generate_constant_inter_cluster(n_neurons, cluster_labels, 0.01, directed=False)
    else:
        # берём предыдущую матрицу и дальше "докручиваем"
        C_intra = C_total_prev.copy()
        C_between = C_total_prev.copy()

    # Обновляем матрицу в соответствии с p_within, p_between
    C_intra = update_intra_cluster_connectivity(C_intra, cluster_labels, p_within)
    C_between = update_inter_cluster_connectivity(C_between, cluster_labels, p_between)

    C_total = np.maximum(C_intra, C_between)

    
    # -- Параметры LIF --
    N = n_neurons
    N_E = int(N * 80 / 100)
    N_I = N - N_E
    print("N", N)
    print("N_E", N_E)
    print("N_I", N_I)
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
        ''',
        threshold="v > v_threshold",
        reset="v = v_reset",
        method="euler",
    )

    n_cluster_neurons = int(n_neurons/2)


    # Шаг 1. Вычисляем метрику betweenness_centrality и получаем список лучших узлов
    if p_input is None:
        p_input = 1.0

    if centrality is None:
        centrality = print_centrality(C_total, n_cluster_neurons, p_input, measure_name=measure_name)

    centrality = sorted(centrality)
    print('Топ нейронов',centrality)

    # Выбираем int(p_input * 50) нейронов из 0..49, даём им синус
    # Шаг 5. Назначаем нужную модуляцию (частоту, ток, фазу) выбранным нейронам
    if centrality:
        neurons.f[centrality] = oscillation_frequency * Hz
        neurons.I0[centrality] = I0_value * pA
        neurons.phi[centrality] = 0
    else:
        print("Список выбранных нейронов пуст — пропускаем назначение модуляции.")

    # cluster1_indices = np.arange(0, n_cluster_neurons)
    # num_chosen = int(p_input * len(cluster1_indices))
    # chosen_indices = np.random.choice(cluster1_indices, size=num_chosen, replace=False)

    # # print(chosen_indices)
    # neurons.f[chosen_indices] = oscillation_frequency * Hz
    # neurons.I0[chosen_indices] = I0_value * pA
    # neurons.phi[chosen_indices] = 0

    input_rate = 50 * Hz
    input_group = PoissonGroup(n_neurons, rates=input_rate)
    syn_input = Synapses(input_group, neurons, on_pre='v_post += J', delay=D)
    syn_input.connect(condition='i >= 0 and j < n_cluster_neurons', p=p_input)


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
    

    
    # # Генерация источников и целей
    N_E_2 = int(N_E / 2)
    N_I_2 = int(N_I / 2)
    N_2 = int(N / 2)
    rows = np.arange(0, N_E_2) 
    rows2 = np.arange(N_E_2, N_E_2 + N_I_2) 
    rows3 = np.arange(N_2, N_E_2 + N_2)
    rows4 = np.arange(N_E_2 + N_2, N)  
    mask = np.isin(np.arange(C_total.shape[0]), rows)  
    mask2 = np.isin(np.arange(C_total.shape[0]), rows2)  
    mask3 = np.isin(np.arange(C_total.shape[0]), rows3) 
    mask4 = np.isin(np.arange(C_total.shape[0]), rows4)  
    sources_exc1, targets_exc1 = np.where(C_total[mask, :] > 0)
    sources_inh1, targets_inh1 = np.where(C_total[mask2, :] > 0)
    sources_exc2, targets_exc2 = np.where(C_total[mask3, :] > 0)
    sources_inh2, targets_inh2 = np.where(C_total[mask4, :] > 0)
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
    exc_synapses.w = 2
    inh_synapses.w = 1
    
    # Мониторы  
    spike_monitor = SpikeMonitor(neurons)

    rate_monitor = None
    rate_monitor2 = None
    if plotting_flags.get('rates', False) or plotting_flags.get('psd', False) or plotting_flags.get('spectrogram', False):
        rate_monitor = PopulationRateMonitor(neurons[:n_neurons/2])
    if plotting_flags.get('rates2', False) or plotting_flags.get('psd2', False) or plotting_flags.get('spectrogram', False):
        rate_monitor2 = PopulationRateMonitor(neurons[n_neurons/2:])

    trace = None
    if plotting_flags.get('trace', False):
        trace = StateMonitor(neurons, 'v', record=True)

    print(f"Количество нейронов: {N}")
    print(f"Количество возбуждающих синапсов: {exc_synapses.N}")
    print(f"Количество тормозных синапсов: {inh_synapses.N}")

    # Запуск
    run(sim_time, profile=True)

    # Профилирование
    print(profiling_summary(show=5))
    end_time = time.time()
    duration = end_time - start_time
    print(f"Testing completed in {duration:.2f} seconds.")

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

    if plotting_flags.get('rates', False) and rate_monitor is not None:
        ax_rates = plotting_flags['ax_rates']
        plot_rates(ax_rates, oscillation_frequency, use_stdp,
                   rate_monitor, t_range, rate_range, rate_tick_step)

    if plotting_flags.get('rates2', False) and rate_monitor2 is not None:
        ax_rates2 = plotting_flags['ax_rates2']
        plot_rates2(ax_rates2, oscillation_frequency, use_stdp,
                    rate_monitor2, t_range)

    if plotting_flags.get('trace', False) and trace is not None:
        ax_trace = plotting_flags['ax_trace']
        plot_trace(ax_trace, trace, oscillation_frequency, use_stdp)

    if plotting_flags.get('psd', False) and rate_monitor is not None:
        ax_psd = plotting_flags['ax_psd']
        plot_psd(rate_monitor, oscillation_frequency, use_stdp, ax_psd)

    if plotting_flags.get('psd2', False) and rate_monitor2 is not None:
        ax_psd2 = plotting_flags['ax_psd2']
        plot_psd2(rate_monitor2, oscillation_frequency, use_stdp, ax_psd2)

    if plotting_flags.get('spectrogram', False) and rate_monitor is not None and rate_monitor2 is not None:
        ax_spectrogram = plotting_flags['ax_spectrogram']
        plot_spectrogram(rate_monitor, rate_monitor2, oscillation_frequency, use_stdp, ax_spectrogram)

    if plotting_flags.get('connectivity2', False):
        connectivity2 = plotting_flags['connectivity2']
        plot_connectivity(n_neurons, exc_synapses, inh_synapses, connectivity2, use_stdp, centrality)

    return avg_neuron_spikes_cluster2_list, time_window_centers, C_total, spike_indices, centrality

