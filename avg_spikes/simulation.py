import numpy as np
from brian2 import *
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
    plot_spectrogram, plot_connectivity
)

def sim(p_within, p_between, g, nu_ext_over_nu_thr, J, sim_time, plotting_flags,
        rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes,
        I0_value, oscillation_frequency, use_stdp, time_window_size,
        C_total_prev=None, p_within_prev=None, p_between_prev=None,
        p_input=None):
    """
    Выполняет один прогон симуляции при заданных параметрах.
    Если C_total_prev=None, генерируем матрицу "с нуля",
    иначе используем прошлую (чтобы копить STDP).
    """

    start_scope()

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
    R = 80 * Mohm
    C = 0.25 * nfarad
    tau = R*C # 20 ms
    v_threshold = -50 * mV
    v_reset = -70 * mV
    v_rest = -65 * mV
    J = J * mV
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

    # # Обнулим осцилляции везде
    # neurons.f = 0 * Hz
    # neurons.I0 = 0 * mA
    # neurons.phi = 0
    # # Если p_input не задан, пусть =1 (все нейроны 1-го кластера)
    # if p_input is None:
    #     p_input = 1.0

    # # Выбираем int(p_input * 50) нейронов из 0..49, даём им синус
    # cluster1_indices = np.arange(0, 50)
    # num_chosen = int(round(p_input * len(cluster1_indices)))
    # chosen_indices = np.random.choice(cluster1_indices, size=num_chosen, replace=False)

    # neurons.f[chosen_indices] = oscillation_frequency * Hz
    # neurons.I0[chosen_indices] = I0_value * pA
    # neurons.phi[chosen_indices] = 0

    input_rate = 20000 * Hz
    input_group = PoissonGroup(1, rates=input_rate)
    syn_input = Synapses(input_group, neurons, on_pre='v_post += J', delay=D)
    syn_input.connect(condition='i == 0 and j < 50', p=p_input)




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
        exc_synapses = Synapses(
            neurons, neurons,
            model='w : 1',
            on_pre='v += J * w',
            delay=D
        )

    exc_synapses.connect(condition='i != j')

    # Устанавливаем веса по C_total
    sources, targets = C_total.nonzero()
    exc_synapses.w[:, :] = 0
    for idx in range(len(sources)):
        i = sources[idx]
        j = targets[idx]
        syn_idx = np.where((exc_synapses.i == i) & (exc_synapses.j == j))[0]
        if len(syn_idx) > 0:
            exc_synapses.w[syn_idx[0]] = 1

    # Мониторы
    spike_monitor = SpikeMonitor(neurons[:100])

    rate_monitor = None
    rate_monitor2 = None
    if plotting_flags.get('rates', False) or plotting_flags.get('psd', False) or plotting_flags.get('spectrogram', False):
        rate_monitor = PopulationRateMonitor(neurons[:50])
    if plotting_flags.get('rates2', False) or plotting_flags.get('psd2', False) or plotting_flags.get('spectrogram', False):
        rate_monitor2 = PopulationRateMonitor(neurons[50:])

    trace = None
    if plotting_flags.get('trace', False):
        trace = StateMonitor(neurons, 'v', record=True)

    # Запуск
    run(sim_time)

    # Анализ спайков
    spike_times = spike_monitor.t / ms
    spike_indices = spike_monitor.i

    bins = np.arange(0, int(sim_time/ms) + 1, time_window_size)
    time_window_centers = (bins[:-1] + bins[1:]) / 2

    avg_neuron_spikes_cluster2_list = []
    start_cluster_neuron = 50
    end_cluster_neuron = 100

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
                    oscillation_frequency, use_stdp)

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
        plot_connectivity(n_neurons, exc_synapses, connectivity2, use_stdp)

    return avg_neuron_spikes_cluster2_list, time_window_centers, C_total, spike_indices
