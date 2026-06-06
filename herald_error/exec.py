import numpy as np
import stim
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from herald_error.circuit import make_rotated_surface_code_memory_no_repeat, count_cnot
from herald_error.herald import (
    sampling_cx_error_list,
    get_unheralded_dem,
)
from herald_error.data import SimulationConfig, Result
from herald_error.qec import count_logical_error_for_pattern

_WORKER_CIRCUIT_NOISELESS = None
_WORKER_DEM_UNHERALDED = None


def _init_worker(circuit_noiseless_text: str, dem_unheralded_text: str) -> None:
    """Initialize process-local Stim objects for multiprocessing workers.

    Args:
        circuit_noiseless_text: Text serialization of the noiseless circuit.
        dem_unheralded_text: Text serialization of the unheralded detector
            error model.
    """
    global _WORKER_CIRCUIT_NOISELESS, _WORKER_DEM_UNHERALDED
    _WORKER_CIRCUIT_NOISELESS = stim.Circuit(circuit_noiseless_text)
    _WORKER_DEM_UNHERALDED = stim.DetectorErrorModel(dem_unheralded_text)


def _count_logical_error_single_thread(
    tasks: list,
    circuit_noiseless: stim.Circuit,
    dem_unheralded: stim.DetectorErrorModel,
    use_tqdm: bool = True,
) -> tuple[int, int, int]:
    """Evaluate all sampled heralding patterns in the current process.

    Args:
        tasks: Tuples of ``(error_pattern, count, seed)`` to evaluate.
        circuit_noiseless: Base circuit without CX errors.
        dem_unheralded: Detector error model using averaged CX noise.
        use_tqdm: Whether to render a progress bar.

    Returns:
        Total shots, logical errors with heralding, and logical errors without
        heralding.
    """

    sum_sample = 0
    sum_error_with_herald = 0
    sum_error_without_herald = 0

    if use_tqdm:
        progress_bar = tqdm(tasks, total=len(tasks))
    else:
        progress_bar = tasks

    for idx, task in enumerate(progress_bar):
        error_pattern, error_pattern_count, error_pattern_seed = task
        if use_tqdm:
            progress_bar.set_postfix(
                sum_sample=sum_sample,
                w_herald=sum_error_with_herald,
                wo_herald=sum_error_without_herald,
            )

        result = count_logical_error_for_pattern(
            circuit_noiseless,
            dem_unheralded,
            error_pattern,
            error_pattern_count,
            error_pattern_seed,
        )

        num_sample, num_error_with_herald, num_error_without_herald = result
        sum_sample += num_sample
        sum_error_with_herald += num_error_with_herald
        sum_error_without_herald += num_error_without_herald

        if use_tqdm:
            progress_bar.set_postfix(
                sum_sample=sum_sample,
                w_herald=sum_error_with_herald,
                wo_herald=sum_error_without_herald,
            )
        else:
            marker = "*" if num_error_with_herald > num_error_without_herald else ""
            print(
                idx, num_sample, num_error_with_herald, num_error_without_herald, marker
            )
    return sum_sample, sum_error_with_herald, sum_error_without_herald


def _count_logical_error_for_pattern_worker(
    error_pattern: list[float],
    error_pattern_count: int,
    error_pattern_seed: int,
) -> tuple[int, int, int]:
    """Evaluate one heralding-pattern task inside a worker process.

    Args:
        error_pattern: CX error probabilities for one heralding pattern.
        error_pattern_count: Number of shots represented by the pattern.
        error_pattern_seed: Random seed for detector sampling.

    Returns:
        Shots and logical-error counts for the pattern.
    """
    return count_logical_error_for_pattern(
        _WORKER_CIRCUIT_NOISELESS,
        _WORKER_DEM_UNHERALDED,
        error_pattern,
        error_pattern_count,
        error_pattern_seed,
    )


def _count_logical_error_multi_thread(
    tasks: list,
    circuit_noiseless: stim.Circuit,
    dem_unheralded: stim.DetectorErrorModel,
    num_workers: int,
    use_tqdm: bool = True,
) -> tuple[int, int, int]:
    """Evaluate sampled heralding patterns across worker processes.

    Args:
        tasks: Tuples of ``(error_pattern, count, seed)`` to evaluate.
        circuit_noiseless: Base circuit without CX errors.
        dem_unheralded: Detector error model using averaged CX noise.
        num_workers: Number of worker processes to launch.
        use_tqdm: Whether to render a progress bar.

    Returns:
        Total shots, logical errors with heralding, and logical errors without
        heralding.
    """

    if use_tqdm:
        progress_bar = tqdm(total=len(tasks))

    sum_sample = 0
    sum_error_with_herald = 0
    sum_error_without_herald = 0

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(str(circuit_noiseless), str(dem_unheralded)),
    ) as executor:

        futures = []
        for error_pattern, error_pattern_count, error_pattern_seed in tasks:
            job = executor.submit(
                _count_logical_error_for_pattern_worker,
                error_pattern,
                error_pattern_count,
                error_pattern_seed,
            )
            futures.append(job)

        for future in as_completed(futures):
            num_sample, num_error_with_herald, num_error_without_herald = (
                future.result()
            )

            sum_sample += num_sample
            sum_error_with_herald += num_error_with_herald
            sum_error_without_herald += num_error_without_herald
            if use_tqdm:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    sum_sample=sum_sample,
                    w_herald=sum_error_with_herald,
                    wo_herald=sum_error_without_herald,
                )
            else:
                maker = "*" if num_error_with_herald > num_error_without_herald else ""
                print(
                    num_sample, num_error_with_herald, num_error_without_herald, maker
                )

    if use_tqdm:
        progress_bar.close()
    return sum_sample, sum_error_with_herald, sum_error_without_herald


def run(config: SimulationConfig, use_tqdm: bool) -> Result:
    """Run a complete heralded error simulation.

    The simulation groups identical heralding patterns, evaluates each unique
    pattern once with its multiplicity, and compares decoding with heralded
    noise against decoding with averaged unheralded noise.

    Args:
        config: Simulation parameters.
        use_tqdm: Whether to show progress bars while evaluating patterns.

    Returns:
        Logical-error counts with and without herald information.
    """

    num_workers = max(1, config.num_workers)
    random_state = np.random.RandomState(config.seed)
    seed_herald_sample = random_state.randint(0, 2**31)

    # create circuit which has no error on CX, and count CX
    circuit_noiseless = make_rotated_surface_code_memory_no_repeat(
        distance=config.distance,
        rounds=config.rounds,
        basis=config.basis,
        before_round_data_depolarization=config.before_round_data_depolarization,
        after_reset_flip_probability=config.error_after_measurement,
        before_measure_flip_probability=config.error_before_measurement,
    )
    cnot_count = count_cnot(circuit_noiseless)
    dem_unheralded = get_unheralded_dem(config, circuit_noiseless, cnot_count)

    # determine heralding patterns
    # As most patterns are expected to be "no heralding", we create counter for patterns to reduce repetition
    error_pattern_list, error_pattern_count_list = sampling_cx_error_list(
        num_sample=config.num_sample,
        count_cx=cnot_count,
        herald_rate=config.herald_rate,
        error_rate_heralded=config.error_rate_heralded,
        error_rate_unheralded=config.error_rate_unheralded,
        seed=seed_herald_sample,
    )
    error_pattern_seed = random_state.randint(0, 2**31, size=len(error_pattern_list))

    if 0:
        with open("detslice-with-ops-svg.html", "w", encoding="utf-8") as f:
            print(circuit_noiseless.diagram("detslice-with-ops-svg-html"), file=f)

    # create task
    tasks = list(zip(error_pattern_list, error_pattern_count_list, error_pattern_seed))

    if num_workers == 1:
        result = _count_logical_error_single_thread(
            tasks, circuit_noiseless, dem_unheralded, use_tqdm=use_tqdm
        )
    else:
        result = _count_logical_error_multi_thread(
            tasks, circuit_noiseless, dem_unheralded, num_workers, use_tqdm=use_tqdm
        )

    sum_sample, sum_error_with_herald, sum_error_without_herald = result
    assert sum_sample == config.num_sample

    return Result(
        simulation_config=config,
        num_error_with_herald=sum_error_with_herald,
        num_error_without_herald=sum_error_without_herald,
    )
