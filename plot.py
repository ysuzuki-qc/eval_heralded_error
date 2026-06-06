import os
import matplotlib.pyplot as plt
import glob
import json
import numpy as np
from scipy.optimize import curve_fit
from herald_error.data import Result, SimulationConfig


def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * b ** (-(x - 1) / 2)


def fit_logical_error_rate(
    distance: list[float],
    logical_error_rate: list[float],
    logical_error_rate_std: list[float],
) -> tuple[float, float]:
    if len(distance) != len(logical_error_rate):
        raise ValueError("distance and logical_error_rate must have the same length")
    if logical_error_rate_std is not None and len(distance) != len(
        logical_error_rate_std
    ):
        raise ValueError(
            "distance and logical_error_rate_std must have the same length"
        )
    if len(distance) < 2:
        raise ValueError("at least two points are required to fit")
    if any(y <= 0 for y in logical_error_rate):
        raise ValueError("logical_error_rate must contain only positive values")
    if logical_error_rate_std is not None and any(
        y_std <= 0 for y_std in logical_error_rate_std
    ):
        raise ValueError("logical_error_rate_std must contain only positive values")

    x = np.asarray(distance, dtype=float)
    y = np.asarray(logical_error_rate, dtype=float)
    y_std = (
        None
        if logical_error_rate_std is None
        else np.asarray(logical_error_rate_std, dtype=float)
    )
    if len(np.unique(x)) < 2:
        raise ValueError("distance must contain at least two distinct values")

    (a, b), _ = curve_fit(model, x, y, sigma=y_std, absolute_sigma=True)
    return float(a), float(b)


def load(path: str) -> list[Result]:
    file_list = glob.glob(f"{path}/result_*.json")
    result_list: list[Result] = []
    for file in file_list:
        with open(file) as fin:
            data = json.load(fin)
        result = Result(
            simulation_config=SimulationConfig(**data["simulation_config"]),
            num_error_with_herald=data["num_error_with_herald"],
            num_error_without_herald=data["num_error_without_herald"],
        )
        result_list.append(result)
    result_list = sorted(result_list, key=lambda x: x.simulation_config.distance)
    return result_list


def plot(result_list: list[Result]) -> None:
    distance_with_herald = []
    distance_without_herald = []
    logical_error_with_herald = []
    logical_error_without_herald = []
    logical_error_with_herald_std = []
    logical_error_without_herald_std = []
    for result in result_list:
        num_sample = result.simulation_config.num_sample
        num_round = result.simulation_config.rounds
        if result.num_error_with_herald > 0:
            distance_with_herald.append(result.simulation_config.distance)
            lp = 1 - (1 - result.num_error_with_herald / num_sample) ** (1 / num_round)
            logical_error_with_herald.append(lp)
            logical_error_with_herald_std.append(
                np.sqrt(lp * (1 - lp) / np.sqrt(num_sample))
            )

        if result.num_error_without_herald > 0:
            distance_without_herald.append(result.simulation_config.distance)
            lp = 1 - (1 - result.num_error_without_herald / num_sample) ** (
                1 / num_round
            )
            logical_error_without_herald.append(lp)
            logical_error_without_herald_std.append(
                np.sqrt(lp * (1 - lp) / np.sqrt(num_sample))
            )

    p0_with_herald, lambda_with_herald = fit_logical_error_rate(
        distance_with_herald, logical_error_with_herald, logical_error_with_herald_std
    )
    p0_without_herald, lambda_without_herald = fit_logical_error_rate(
        distance_without_herald,
        logical_error_without_herald,
        logical_error_without_herald_std,
    )

    plt.xlabel("code distance")
    plt.ylabel("logical error rate per round")
    plt.yscale("log")

    fit_d = np.array([1, 15])
    plt.xlim(1, np.max(fit_d))
    plt.ylim(np.min(model(fit_d, p0_with_herald, lambda_with_herald)), 1e-1)

    plt.plot(
        distance_with_herald,
        logical_error_with_herald,
        "*--",
        color="blue",
        label=f"with herald p0={p0_with_herald:.2e} Lam={lambda_with_herald:.2e}",
    )
    plt.plot(
        fit_d, model(fit_d, p0_with_herald, lambda_with_herald), color="blue", alpha=0.5
    )
    plt.plot(
        distance_without_herald,
        logical_error_without_herald,
        "*--",
        color="red",
        label=f"without herald p0={p0_without_herald:.2e} Lam={lambda_without_herald:.2e}",
    )
    plt.plot(
        fit_d,
        model(fit_d, p0_without_herald, lambda_without_herald),
        color="red",
        alpha=0.5,
    )
    plt.grid(which="major", color="black", linestyle="-", alpha=0.2)
    plt.grid(which="minor", color="black", linestyle="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()

    if not os.path.exists("figure"):
        os.mkdir("figure")
    plt.savefig("./figure/logical_error_rate.pdf")
    plt.savefig("./figure/logical_error_rate.png")
    plt.show()

    plt.clf()
    test_d_list = np.arange(3, 51, 2)
    num_qubit = 2 * test_d_list**2 - 1
    QuOP_with_herald = 1 / (
        model(test_d_list, p0_with_herald, lambda_with_herald) * test_d_list
    )
    QuOP_without_herald = 1 / (
        model(test_d_list, p0_without_herald, lambda_without_herald) * test_d_list
    )

    def get_pareto_frontier(x, y):
        px = np.zeros(shape=len(x) * 2)
        py = np.zeros(shape=len(y) * 2)
        px[::2] = x
        px[1:-1:2] = x[1:]
        px[-1] = x[-1]
        py[::2] = y
        py[1::2] = y
        return px, py

    x, y = get_pareto_frontier(num_qubit, QuOP_with_herald)
    plt.plot(
        x,
        y,
        color="blue",
        label=f"with herald p0={p0_with_herald:.2e} Lam={lambda_with_herald:.2e}",
    )
    x, y = get_pareto_frontier(num_qubit, QuOP_without_herald)
    plt.plot(
        x,
        y,
        color="red",
        label=f"without herald p0={p0_without_herald:.2e} Lam={lambda_without_herald:.2e}",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("qubit count per surface code")
    plt.ylabel("QuOP")
    plt.grid(which="major", color="black", linestyle="-", alpha=0.2)
    plt.grid(which="minor", color="black", linestyle="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.xlim(min(num_qubit), max(num_qubit))
    plt.ylim(1e1, 1e13)
    plt.savefig("./figure/quop.pdf")
    plt.savefig("./figure/quop.png")
    plt.show()


if __name__ == "__main__":
    result_list = load("./result")
    plot(result_list)
