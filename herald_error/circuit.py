import stim


def _flatten_stim_circuit(circuit: stim.Circuit) -> stim.Circuit:
    """Expand repeat blocks in a Stim circuit into a flat instruction sequence.

    Args:
        circuit: Circuit that may contain repeat blocks.

    Returns:
        A circuit containing only concrete circuit instructions.
    """
    flattend_circuit = stim.Circuit()

    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction):
            flattend_circuit.append(inst)

        elif isinstance(inst, stim.CircuitRepeatBlock):
            body = _flatten_stim_circuit(inst.body_copy())
            for _ in range(inst.repeat_count):
                flattend_circuit += body

        else:
            raise TypeError(f"Unexpected instruction type: {type(inst)}")

    return flattend_circuit


def make_rotated_surface_code_memory_no_repeat(
    distance: int,
    rounds: int,
    basis: str,
    before_round_data_depolarization: float,
    after_reset_flip_probability: float,
    before_measure_flip_probability: float,
) -> stim.Circuit:
    """Create a rotated surface-code memory circuit without repeat blocks.

    Args:
        distance: Code distance passed to Stim's generated circuit factory.
        rounds: Number of syndrome-extraction rounds.
        basis: Logical memory basis, either ``"x"`` or ``"z"``.
        before_round_data_depolarization: Data-qubit depolarization probability
            applied at the beginning of each round.
        after_reset_flip_probability: Reset flip probability.
        before_measure_flip_probability: Measurement flip probability.

    Returns:
        A generated Stim circuit with repeat blocks expanded.
    """
    assert basis in ["x", "z"]
    repeated = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=before_round_data_depolarization,
        after_clifford_depolarization=0,
        after_reset_flip_probability=after_reset_flip_probability,
        before_measure_flip_probability=before_measure_flip_probability,
    )
    flattened = _flatten_stim_circuit(repeated)
    return flattened


def count_cnot(circuit: stim.Circuit) -> int:
    """Count two-qubit CNOT operations represented by CX instructions.

    Args:
        circuit: Stim circuit whose CX instructions are counted.

    Returns:
        Number of CNOT pairs in all CX instructions.
    """
    count = 0
    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction):
            if inst.name == "CX":
                targets = inst.targets_copy()
                assert len(targets) % 2 == 0
                count += len(targets) // 2
    return count


def add_cx_error_to_circuit(
    circuit: stim.Circuit,
    cx_error_list: list[float],
) -> stim.Circuit:
    """Insert per-CX depolarizing errors into a circuit.

    Each CNOT pair in every CX instruction receives one ``DEPOLARIZE2``
    instruction immediately after the original CX instruction.

    Args:
        circuit: Flat Stim circuit to augment with CX noise.
        cx_error_list: Error probabilities, one value for each CNOT pair.

    Returns:
        A new circuit containing the original instructions plus inserted CX
        depolarizing errors.
    """
    new_circuit = stim.Circuit()
    idx = 0
    for inst in circuit:
        assert isinstance(inst, stim.CircuitInstruction)

        if inst.name == "CX":
            new_circuit.append(inst)
            targets = inst.targets_copy()
            assert len(targets) % 2 == 0
            for i in range(0, len(targets), 2):
                new_circuit.append(
                    "DEPOLARIZE2", targets[i: i + 2], cx_error_list[idx]
                )
                idx += 1
        elif inst.name in ["MR", "R", "M", "H"]:
            new_circuit.append(inst)
        elif inst.name in ["DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR"]:
            new_circuit.append(inst)
        elif inst.name in [
            "TICK",
            "QUBIT_COORDS",
            "DETECTOR",
            "SHIFT_COORDS",
            "OBSERVABLE_INCLUDE",
        ]:
            new_circuit.append(inst)
        else:
            raise ValueError(f"Unexpected instruction: {inst.name}")
    assert idx == len(cx_error_list)
    return new_circuit
