import matplotlib.pyplot as plt
import glob
import json
from herald_error.data import Result, SimulationConfig


def load(path: str) -> list[Result]:
    file_list = glob.glob("./result/result_*.json")
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
    distance = []
    logical_error_with_herald = []
    logical_error_without_herald = []
    for result in result_list:
        distance.append(result.simulation_config.distance)
        num_sample = result.simulation_config.num_sample
        logical_error_with_herald.append(result.num_error_with_herald/num_sample)
        logical_error_without_herald.append(result.num_error_without_herald/num_sample)

    plt.xlabel("code distance")
    plt.ylabel("logical error rate")
    plt.yscale("log")
    plt.plot(distance, logical_error_with_herald)
    plt.plot(distance, logical_error_without_herald)
    plt.show()


if __name__ == "__main__":
    result_list = load("./result")
    plot(result_list)
