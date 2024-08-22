import matplotlib.pyplot as plt
from datetime import datetime
import scipy.optimize as opt
import numpy as np
import json
from tkinter import filedialog
from typing import Union, Tuple, Literal


def exp_function(x, a, b, c, d):
    y = c - a * np.exp(-x * b + d)
    return y


def linear_function(x, a, b):
    y = a * x + b
    return y


def calculate_r_squared(x_data, y_data, optimized_parameters, function) -> float:
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    residuals = y_data - function(x_data, *optimized_parameters)
    ss_residuals = np.sum(residuals ** 2)
    ss_total = np.sum((y_data - np.mean(y_data)) ** 2)

    r_squared = 1 - (ss_residuals / ss_total)

    return r_squared


def shorten_filepath(filepath: str, remove_extension=True) -> str:
    short = filepath[filepath.rfind("/") + 1:]
    if remove_extension:
        short = short[:short.rfind(".")]
    return short


def read_json_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"File {file_path} does not contain valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")


def write_json_file(data: dict, file_path: str = None, filename_suggestion: str = None, compact_json: bool = False
                    ) -> None:
    json_str = json.dumps(data, indent=0 if compact_json else 4)

    if file_path is None:
        file_path = filedialog.asksaveasfile(mode="w", defaultextension=".json",
                                             filetypes=[("json file", "*.json")], initialfile=filename_suggestion)
        file_path.write(json_str)
        file_path.close()
    else:
        with open(file_path, "w") as writefile:
            writefile.write(json_str)


def read_data_file(file: str = None, return_relative_time: bool = True, start_time: str = None
                   ) -> tuple[any, any, any, any]:
    if file is None:
        file = filedialog.askopenfilename(title="Select data to process",
                                          filetypes=(("text files", "*.txt"), ("all files", "*.*")))

    with open(file) as readfile:
        data = readfile.read()

    lines = data.split("\n")

    exp_time = []
    pump_flow = []
    mass = []
    permeate_flow = []

    for line in lines:
        if len(line) > 1:
            line = line.replace(",", ".")
            values = line.split("; ")
            if len(values) == 4:
                t, pf, m, fv = datetime.strptime(values[0], "%H:%M:%S"), float(values[1]), float(values[2]), float(
                    values[3])
                exp_time.append(t)
                pump_flow.append(float(pf))
                mass.append(float(m))
                permeate_flow.append(fv)

    relative_time = []

    if start_time is not None:
        start_time = datetime.strptime(start_time, "%H:%M:%S")
    else:
        start_time = exp_time[0]

    for t in exp_time:
        relative_time.append((t - start_time).total_seconds() / 60)

    permeate_flow_mlmin = []

    for val in permeate_flow:
        permeate_flow_mlmin.append(val * 16.666)

    if return_relative_time:
        return relative_time, mass, pump_flow, permeate_flow_mlmin
    else:
        return exp_time, mass, pump_flow, permeate_flow_mlmin


def filter_data(x: list[Union[float, int]], y: list[Union[float, int]], smooth_factor: int = 25,
                threshold: float = 0.03, optimize_threshold: bool = False, min_pts_preserved: float = .8,
                max_iterations: int = 100, _iteration: int = 1
                ) -> tuple[list[Union[int, float]], list[Union[int, float]], dict]:
    filter_parameters = {
        "threshold": threshold,
        "smooth_factor": smooth_factor,
        "optimize_threshold": optimize_threshold
    }

    if optimize_threshold:
        filter_parameters["minimum_pts_preserved"] = min_pts_preserved
        filter_parameters["optimization_iterations"] = _iteration

    if smooth_factor >= 3:
        smooth_y = smooth_curve(x, y, smooth_factor, plot=False)
    else:
        smooth_y = y

    filtered_x, filtered_y = _bandpass_filter(x, y, smooth_y, threshold)

    percentage_datapoints_preserved = len(filtered_x) / len(x)

    filter_parameters["pts_preserved"] = percentage_datapoints_preserved

    if percentage_datapoints_preserved < min_pts_preserved and optimize_threshold and _iteration <= max_iterations:
        print(f"Optimizing... Iteration {_iteration}")
        filtered_x, filtered_y, filter_parameters = filter_data(x, y, threshold=threshold + 0.005,
                                                                optimize_threshold=True, _iteration=_iteration + 1)
    elif _iteration >= max_iterations:
        print(f"Maximum iterations reached, aborting optimization of threshold at {round(threshold, 3)}")

    return filtered_x, filtered_y, filter_parameters


def _bandpass_filter(x: list[Union[float, int]], y: list[Union[float, int]],
                     reference_y: list[Union[float, int]], threshold: Union[float, int]
                     ) -> tuple[list[Union[int, float]], list[Union[int, float]]]:
    filtered_x, filtered_y = ([], [])
    for i, y_i in enumerate(y):
        if abs(y_i - reference_y[i]) / y_i <= threshold and y_i >= 0:
            filtered_x.append(x[i])
            filtered_y.append(y_i)

    return filtered_x, filtered_y


def smooth_curve(x: list[Union[float, int]], y: list[Union[float, int]], smoothing_factor: int, plot: bool = True,
                 ax: plt.Axes = None, mode: Literal["valid", "same", "full"] = "full", label: str = "moving average"
                 ) -> np.ndarray:
    # noinspection PyTypeChecker
    moving_average = np.convolve(y, np.ones(smoothing_factor) / smoothing_factor, mode)

    label += f" (over {smoothing_factor} points)"

    if plot and ax is not None:
        ax.plot(x, y, label=label)
    elif plot and ax is None:
        ax = plt.gca()
        ax.plot(x, y, label=label)

    return moving_average


def plot_and_process(data: tuple[list, list], parameters: dict, fit_bounds: list[float] = None, plot: bool = True,
                     ax: plt.Axes = None, plot_title: str = "Permeate Flux", data_name: str = None,
                     style_raw: str = "o", color_raw=None, style_fit: str = "-", color_fit=None,
                     plot_fit_interval: bool = True, plot_equilibrium_value: bool = True,
                     display_results_in_plot: bool = True, y_lim_upper=None) -> dict:
    if ax is None and plot:
        ax = plt.gca()

    if y_lim_upper is None:
        y_lim_upper = np.max(data[1])

    relative_time, permeate_flow_mlmin = data

    average_last_min = np.mean(permeate_flow_mlmin[-120:])
    std_last_min = np.std(permeate_flow_mlmin[-120:])

    optimization_start = parameters["fit_start"]
    optimization_end = parameters["fit_end"]

    end, start = _identify_fit_interval(optimization_end, optimization_start, relative_time)

    if plot:
        ax.plot(relative_time, permeate_flow_mlmin, style_raw, color=color_raw, label=data_name)

    try:

        # noinspection PyTupleAssignmentBalance
        optimized_parameters, pcov = opt.curve_fit(exp_function, relative_time[start:end],
                                                   permeate_flow_mlmin[start:end],
                                                   bounds=fit_bounds)
        resolved_x = np.linspace(optimization_start, optimization_end + 30, 100)

        r_squared = calculate_r_squared(relative_time[start:end], permeate_flow_mlmin[start:end],
                                        optimized_parameters, exp_function)

        equilibrium_time = (- np.log((- 0.01 * optimized_parameters[2]) / optimized_parameters[0]) +
                            optimized_parameters[3]) / optimized_parameters[1]
        equilibrium_time = round(equilibrium_time, 0)

        if plot:
            ax.plot(resolved_x, exp_function(np.asarray(resolved_x), *optimized_parameters),
                    style_fit, color=color_fit, label=f"fit for {data_name}" if data_name is not None else None)
            ax.set_title(plot_title)
            ax.set_xlabel("$t - t_0$ / min")

            if display_results_in_plot:
                ax.text(0.01, 0.95, f"$R^2$ = {round(r_squared, 4)}", transform=ax.transAxes)
                ax.text(0.01, 0.90, f"pred. $F_V$ = {round(optimized_parameters[2], 2)} mL / min",
                        transform=ax.transAxes)
                ax.text(0.01, 0.85, f"pred. eq. time = {equilibrium_time} min", transform=ax.transAxes)
                ax.text(0.01, 0.80, "$\\bar{F_V}$ = " + f" = {round(average_last_min, 2)} mL / min",
                        transform=ax.transAxes)
                ax.text(0.01, 0.75, "$\\sigma_{F_V}$ = " + f" = {round(std_last_min, 4)} mL / min",
                        transform=ax.transAxes)

            if plot_equilibrium_value:
                ax.vlines(equilibrium_time, 0, y_lim_upper, linestyles="dotted")
                ax.hlines(optimized_parameters[2], optimization_start, equilibrium_time + 10, linestyles="dotted")

        results = {
            "permeateFluxStabilized": optimized_parameters[2],
            "rSquared": round(r_squared, 4),
            "stabilizationTimeMin": equilibrium_time,
            "finalPermeateFlux": average_last_min,
            "stdPermeateFlux": std_last_min
        }

    except RuntimeError:
        print("Could not fit to function using the given data! You might try to filter the data before analysis.")
        results = {}

    if plot_fit_interval and plot:
        ax.vlines(relative_time[start], 0, y_lim_upper, linestyles="dotted", color="#fa8174")
        ax.vlines(relative_time[end], 0, y_lim_upper, linestyles="dotted", color="#fa8174")

    return results


def _identify_fit_interval(optimization_end: Union[int, float], optimization_start: Union[int, float],
                           relative_time: list[Union[int, float]]) -> tuple[int, int]:
    start = 0
    end = -1
    for t in relative_time:
        if t >= optimization_start:
            start = relative_time.index(t)
            break
    for t in relative_time:
        if t >= optimization_end:
            end = relative_time.index(t)
            break
    return end, start


def select_data_files() -> Union[Literal[""], tuple[str, ...]]:
    files = filedialog.askopenfilenames(title="Select data to process",
                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    return files


def filter_parameter_analysis(data, fit_parameters, smooth_factors=None, thresholds=None) -> None:
    if smooth_factors is None:
        smooth_factors = [0, 3, 6, 9, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.17, 0.20, 0.25, 0.5, 1.0]

    results = [[], [], [], []]
    used_thresholds = []
    used_smoothing_factors = []

    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # iterating over the different combinations of processing parameters
    for s_f in smooth_factors:
        r_squared = []
        eq_values = []
        eq_times = []
        used_t_h = []
        pts_preserved = []
        for t_h in thresholds:
            filter_result = filter_data(data[0], data[3], smooth_factor=s_f, threshold=t_h)
            pts_preserved.append(filter_result[2])
            filtered_data = filter_result[0:2]
            result = plot_and_process(filtered_data, fit_parameters, fit_bounds=[-150, 150], plot=False)
            used_t_h.append(t_h)
            r_squared.append(result["rSquared"])
            eq_values.append(result["permeateFluxStabilized"])
            eq_times.append(result["stabilizationTimeMin"])
            results[0].append(result["permeateFluxStabilized"])
            results[1].append(result["stabilizationTimeMin"])
            results[2].append(result["rSquared"])
            results[3].append(pts_preserved[-1])
            used_thresholds.append(t_h)
            used_smoothing_factors.append(s_f)

        ax.plot(used_t_h, eq_values, label=s_f)
        ax2.plot(used_t_h, r_squared, label=s_f)
        ax3.plot(used_t_h, eq_times, label=s_f)

        ax.set_title("Stablized Permeate Flux")
        ax2.set_title("$R^2$")
        ax3.set_title("Equilbration Time")

        ax.set_ylabel("Flux / $mL \; min^{-1}$")
        ax2.set_ylabel("$R^2$")
        ax3.set_ylabel("t / min")
        ax.set_xlabel("Filter threshold")
        ax2.set_xlabel("Filter threshold")
        ax3.set_xlabel("Filter threshold")

        ax4.scatter(results[0], results[2])
        ax4.set_xlabel('Stablized Permeate Flux (calculated) / $mL \; min^{-1}$')
        ax4.set_ylabel('$R^2$')

    ax.legend()
    plt.show()

    _summarize_filter_analysis(used_smoothing_factors, used_thresholds, results)

    plot_3d(used_smoothing_factors, used_thresholds, results[0],
            "Stablized Permeate Flux (calculated) / $mL \\; min^{-1}$")
    plot_3d(used_smoothing_factors, used_thresholds, results[1],
            title_x="Filter threshold", title_y="Smoothing factor", title_z="Stablization Time (calc.) / min")
    plot_3d(used_smoothing_factors, used_thresholds, results[2],
            title_x="Filter threshold", title_y="Smoothing factor", title_z="$R^2$ / -")
    plot_3d(used_smoothing_factors, used_thresholds, results[3],
            title_x="Filter threshold", title_y="Smoothing factor", title_z="Pts Preserved / -")

    plt.show()


def _summarize_filter_analysis(smooth_factors: list[int], thresholds: list[float], results: dict, r_min: float = .9,
                               pts_min: float = .8) -> None:
    r_max_index = results[2].index(max(results[2]))
    print(f"best R2 ({results[2][r_max_index]}): s = {smooth_factors[r_max_index]}, t = {thresholds[r_max_index]}")

    closest_to_pts_min = 1
    for i, pt in enumerate(results[3]):
        diff = pt - pts_min
        if 0 <= diff < closest_to_pts_min - pts_min and results[2][i] >= r_min:
            closest_to_pts_min = pt

    best_pts_index = results[3].index(closest_to_pts_min)
    print(f"max_min pts ({round(results[3][best_pts_index] * 100, 2)} %): s = {smooth_factors[best_pts_index]}, "
          f"t = {thresholds[best_pts_index]}, "
          f"R2 = {results[2][best_pts_index]}")

    mean_permeate_flux = (np.mean(results[0]), np.std(results[0]))
    mean_stabilization_time = (np.mean(results[1]), np.std(results[1]))

    print(f"Mean stablization time: {mean_stabilization_time} min")
    print(f"Mean stablized permeate flux: {mean_permeate_flux} mL/min")


def plot_3d(x: list[Union[int, float]], y: list[Union[int, float]], z: list[Union[int, float]],
            title_x: str = None, title_y: str = None, title_z: str = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(y, x, z)

    ax.set_xlabel(title_x)
    ax.set_ylabel(title_y)
    ax.set_zlabel(title_z)


def main():
    # plt.style.use("dark_background")

    data_files = select_data_files()

    main_loop = True
    while main_loop:
        param_dict = read_json_file("Parameters/permeability.json")

        for datafile in data_files:

            fig, (ax_raw, ax_processed) = plt.subplots(2)

            data = read_data_file(datafile)

            # filter_parameter_analysis(data, param_dict)

            unfiltered_result_dict = plot_and_process((data[0], data[3]), param_dict, fit_bounds=[-400, 400], ax=ax_raw)

            unfiltered_result_dict["fit_parameters"] = param_dict
            unfiltered_result_dict["data_file"] = shorten_filepath(datafile, remove_extension=False)

            data = filter_data(data[0], data[3], threshold=0.2, smooth_factor=25, optimize_threshold=False)

            result_dict = plot_and_process((data[0], data[1]), param_dict, fit_bounds=[-400, 400], ax=ax_processed,
                                           plot_title="Filtered Data", y_lim_upper=5)

            result_dict["filter_parameters"] = data[2]

            unfiltered_result_dict["filtered_results"] = result_dict
            plt.show()

            write_file = input(f"Save results of {datafile} (y/n/q)? ")

            if write_file == "y":
                write_json_file(unfiltered_result_dict, filename_suggestion=shorten_filepath(datafile))
                main_loop = False
            elif write_file == "q":
                main_loop = False


if __name__ == "__main__":
    main()
