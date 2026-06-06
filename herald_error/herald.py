"""Heralding-pattern sampling and unheralded-noise model construction."""

import numpy as np
import stim
from herald_error.data import SimulationConfig
from herald_error.circuit import add_cx_error_to_circuit


def sampling_cx_error_list(
    num_sample: int,
    count_cx: int,
    herald_rate: float,
    error_rate_heralded: float,
    error_rate_unheralded: float,
    seed: int = None,
) -> list[float]:
    """Sample and aggregate CX error-probability patterns.

    Each sampled shot first draws a boolean heralding pattern over all CX
    locations. Identical patterns are grouped so later simulation only needs to
    evaluate each unique pattern once.

    Args:
        num_sample: Number of shots to sample.
        count_cx: Number of CX locations in the circuit.
        herald_rate: Probability that each CX location is heralded.
        error_rate_heralded: Error probability assigned to heralded CXs.
        error_rate_unheralded: Error probability assigned to unheralded CXs.
        seed: Optional random seed for sampling.

    Returns:
        Unique CX error-probability patterns and their multiplicities, sorted by
        decreasing multiplicity.
    """
    random_state = np.random.RandomState(seed)
    prob_list = random_state.random([num_sample, count_cx])
    herald_pattern_list = prob_list < herald_rate
    herald_pattern_flag_list, herald_pattern_count_list = np.unique(
        herald_pattern_list, axis=0, return_counts=True
    )

    order = np.argsort(herald_pattern_count_list)[::-1]
    herald_pattern_flag_list_sorted = herald_pattern_flag_list[order]
    herald_pattern_count_list_sorted = herald_pattern_count_list[order]
    herald_pattern_prob_list_sorted = np.where(
        herald_pattern_flag_list_sorted,
        error_rate_heralded,
        error_rate_unheralded,
    )
    assert (
        np.sum(herald_pattern_count_list_sorted) == num_sample
    ), "The sum of counts must equal num_sample."

    return herald_pattern_prob_list_sorted, herald_pattern_count_list_sorted


def get_unheralded_dem(
    config: SimulationConfig, circuit_noiseless: stim.Circuit, cnot_count: int
) -> stim.DetectorErrorModel:
    """Build a detector error model using averaged CX error probability.

    Args:
        config: Simulation parameters containing herald and error rates.
        circuit_noiseless: Base circuit without CX depolarizing errors.
        cnot_count: Number of CX locations in the circuit.

    Returns:
        Detector error model for the circuit with uniform averaged CX noise.
    """

    # calculate average error rate
    average_error_rate = (
        config.herald_rate * config.error_rate_heralded
        + (1 - config.herald_rate) * config.error_rate_unheralded
    )

    # add uniform error rates on all the CX gates
    circuit_unheralded = add_cx_error_to_circuit(
        circuit_noiseless,
        [average_error_rate] * cnot_count,
    )
    dem_unheralded = circuit_unheralded.detector_error_model(decompose_errors=False)

    return dem_unheralded
