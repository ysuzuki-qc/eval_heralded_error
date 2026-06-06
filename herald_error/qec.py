import numpy as np
import stim
import pymatching

from herald_error.circuit import add_cx_error_to_circuit


def _get_logical_error_count(
    dem: stim.DetectorErrorModel,
    detector_list: list[list[int]],
    observable_flip_list: list[list[int]],
) -> int:
    """Decode detector samples and count logical prediction failures.

    Args:
        dem: Detector error model used to construct the decoder.
        detector_list: Detector-event samples.
        observable_flip_list: Actual observable flips for the same samples.

    Returns:
        Number of samples whose decoded observable prediction differs from the
        sampled observable flip.
    """
    heralded_decoder = pymatching.Matching.from_detector_error_model(dem)
    observable_pred_list = heralded_decoder.decode_batch(detector_list)

    num_error = 0
    for pred, flip in zip(observable_pred_list, observable_flip_list):
        if not np.array_equal(pred, flip):
            num_error += 1
    return num_error


def count_logical_error_for_pattern(
    circuit_noiseless: stim.Circuit,
    dem_unheralded: stim.DetectorErrorModel,
    error_pattern: list[float],
    error_pattern_count: int,
    error_pattern_seed: int,
) -> tuple[int, int, int]:
    """Sample one CX error pattern and compare heralded/unheralded decoding.

    Args:
        circuit_noiseless: Base circuit without CX errors.
        dem_unheralded: Detector error model built from averaged CX noise.
        error_pattern: CX error probabilities for this heralding pattern.
        error_pattern_count: Number of shots to sample for this pattern.
        error_pattern_seed: Random seed used by Stim's detector sampler.

    Returns:
        The number of shots, logical errors decoded with herald information,
        and logical errors decoded with the unheralded model.
    """
    # Create heralded circuits
    circuit_heralded = add_cx_error_to_circuit(
        circuit_noiseless,
        error_pattern,
    )

    # sampling
    sampler = circuit_heralded.compile_detector_sampler(seed=error_pattern_seed)
    detector_list, observable_flip_list = sampler.sample(
        shots=error_pattern_count, separate_observables=True
    )

    # estimate error with herald signals
    dem_heralded = circuit_heralded.detector_error_model(decompose_errors=False)
    num_error_with_herald = _get_logical_error_count(
        dem_heralded, detector_list, observable_flip_list
    )

    # estimate error without herald signals
    num_error_without_herald = _get_logical_error_count(
        dem_unheralded, detector_list, observable_flip_list
    )

    return error_pattern_count, num_error_with_herald, num_error_without_herald
