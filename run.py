import os
import pprint
import json
from dataclasses import asdict

from herald_error.exec import run
from herald_error.data import SimulationConfig


def main() -> None:
    if not os.path.exists("result"):
        os.mkdir("result")
    # create config
    p = 5e-3
    N = 100000
    basis = "z"
    p_meas = p
    p_idle = p
    p_herald = p
    p_error_with_herald = 0.5
    p_error_without_herald = 1e-3
    num_worker = 12
    seed = 42

    d_list = [3, 5, 7, 9]

    for d in d_list:
        config = SimulationConfig(
            num_sample=N,
            distance=d,
            rounds=d,
            basis=basis,
            herald_rate=p_herald,
            error_rate_with_herald=p_error_with_herald,
            error_rate_without_herald=p_error_without_herald,
            error_before_measurement=p_meas / 2,
            error_after_measurement=p_meas / 2,
            before_round_data_depolarization=p_idle,
            num_workers=num_worker,
            seed=seed,
        )

        result = run(config, use_tqdm=True)

        output = f"./result/result_{d}.json"
        with open(output, "w") as fout:
            json.dump(asdict(result), fout, indent=4)
        pprint.pprint(result)


if __name__ == "__main__":
    main()
