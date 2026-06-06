from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Parameters controlling a single heralded error simulation run.

    Attributes:
        num_sample: Number of Monte Carlo shots to simulate.
        distance: Surface-code distance.
        rounds: Number of syndrome-extraction rounds.
        basis: Logical memory basis, either ``"x"`` or ``"z"``.
        herald_rate: Probability that a CX location is heralded.
        error_rate_with_herald: CX error probability for heralded locations.
        error_rate_without_herald: CX error probability for unheralded locations.
        error_before_measurement: Measurement flip probability before measuring.
        error_after_measurement: Reset flip probability after measurement/reset.
        before_round_data_depolarization: Data-qubit depolarization probability
            before each round.
        seed: Optional random seed used to derive sampling seeds.
        num_workers: Number of worker processes used for evaluation.
    """

    num_sample: int
    distance: int
    rounds: int
    basis: str
    herald_rate: float
    error_rate_with_herald: float
    error_rate_without_herald: float
    error_before_measurement: float
    error_after_measurement: float
    before_round_data_depolarization: float
    num_workers: int = 1
    seed: int = None


@dataclass
class Result:
    """Logical-error counts produced by a simulation run.

    Attributes:
        simulation_config: Configuration used to produce the result.
        num_error_with_herald: Logical errors decoded with herald information.
        num_error_without_herald: Logical errors decoded using averaged noise.
    """

    simulation_config: SimulationConfig
    num_error_with_herald: int
    num_error_without_herald: int
