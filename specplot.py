import json

import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import scipy.optimize as opt
import os

# This script © 2024 by Henrik Schröter is licensed under CC BY-SA 4.0
# Email: henrik.schroeter@uni-rostock.de / ORCID 0009-0008-1112-2835

# specplot v1.0


def plot_full_spectrum(file=None, title=None):
    if file is None:
        file = filedialog.askopenfilename()
    label = file[file.rfind("/"):] if file.rfind("/") != -1 else file

    wavelength_lst, absorption_lst = read_spectrum(file)

    plt.plot(wavelength_lst, absorption_lst, label=label)

    ax = plt.gca()

    ax.set_title(title)
    ax.set_ylabel("Abs / -")
    ax.set_xlabel("$\lambda$ / nm")


def read_spectrum(file, try_relative=True) -> (list[float], list[float]):
    if os.path.exists(file):
        with open(file) as readfile:
            data = readfile.read()
    elif os.path.exists(extend_to_current_path(file)) and try_relative:
        with open(extend_to_current_path(file), "r") as readfile:
            data = readfile.read()
    else:
        raise FileNotFoundError

    lines = data.split("\n")

    wavelength_lst = []
    absorption_lst = []

    for line in lines:
        if len(line) > 1:
            line = line.replace(",", ".")
            values = line.split(";")
            if len(values) == 2:
                w, a = float(values[0]), float(values[1])
                wavelength_lst.append(w)
                absorption_lst.append(a)

    return wavelength_lst, absorption_lst


def return_absorbance_for_wavelength(spectral_data: tuple[list[float], list[float]], wavelength) -> float:
    relevant_index = spectral_data[0].index(wavelength)
    if relevant_index != -1:
        return spectral_data[1][relevant_index]
    else:
        print(f"Could not find absorbance value for lambda = {wavelength} nm")
        return 0


def monochromatic_plot(selected_wavelength, files, concentrations=None, title=None):
    absorptions = []
    if concentrations is None:
        concentrations = []

    for file in files:
        spectral_data = read_spectrum(file)
        absorptions.append(return_absorbance_for_wavelength(spectral_data, selected_wavelength))
        if not concentrations:
            concentrations.append(float(input(f"Specify concentration (mol L-1) for {file}: ")))

    plt.scatter(concentrations, absorptions)

    ax = plt.gca()

    ax.set_title(title)
    ax.set_xlim([0, None])
    ax.set_ylabel("Abs / -")
    ax.set_xlabel("c / $mol \; L^{-1}$")

    return concentrations, absorptions


def fit_calibration(concentrations, absorptions, label="calibration"):
    optimized_parameters, covariance = opt.curve_fit(linear_function, concentrations, absorptions, bounds=[0, 1000])
    resolved_x = np.linspace(0, max(concentrations), 100)

    r_squared = calculate_r_squared(concentrations, absorptions, optimized_parameters, linear_function)

    label += f" [m = {round(optimized_parameters[0], 2)}" + " $L \; mol^{-1}$, $R^2$ =" + f" {round(r_squared, 4)}]"

    plt.plot(resolved_x, linear_function(np.asarray(resolved_x), *optimized_parameters), label=label)

    ax = plt.gca()
    ax.set_title("Calibration")

    return optimized_parameters, r_squared


def linear_function(x, a):
    y = x * a
    return y


def select_spectra_files() -> list[str]:
    files = filedialog.askopenfilenames(title="Select spectra to process",
                                        filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    return files


def concentrations_from_spec_files(spec_files=None, wavelength=None, conversion_factor=None,
                                   calibration_file=None) -> tuple[list[str], list[float]]:
    if spec_files is None:
        spec_files = select_spectra_files()
    concentrations = []
    sample_labels = []
    for file in spec_files:
        data = read_spectrum(file)
        if wavelength is None:
            wavelength = input("Select a wavelength: ")
            if wavelength == "plot":
                plot_full_spectrum(file, title="Select a wavelength")
                plt.show()
                wavelength = float(input("Select a wavelength: "))
            else:
                wavelength = float(wavelength)
        concentrations.append(convert_abs_to_concentration(return_absorbance_for_wavelength(data, wavelength), conversion_factor, mmol=True))
        sample_labels.append(shorten_filepath(file))

    return sample_labels, concentrations


def convert_abs_to_concentration(absorbance, conversion_factor, mmol=False) -> float:
    if mmol:
        concentration = absorbance / conversion_factor * 1000
    else:
        concentration = absorbance / conversion_factor
    return concentration


def calculate_r_squared(x_data, y_data, optimized_parameters, function) -> float:
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    residuals = y_data - function(x_data, *optimized_parameters)
    ss_residuals = np.sum(residuals ** 2)
    ss_total = np.sum((y_data - np.mean(y_data)) ** 2)

    r_squared = 1 - (ss_residuals / ss_total)

    return r_squared


def extend_to_current_path(file: str):
    return __file__[:__file__.rfind("\\")] + "\\" + file


def shorten_filepath(filepath: str) -> str:
    short = filepath[filepath.rfind("/") + 1:]
    return short


def merge_multiple(data: tuple[list[str], list[float]], erna=-6) -> tuple[list[str], list[tuple[float, float]]]:
    merged_titles = []
    mean_tuples = []
    for i, datapoint in enumerate(data[0]):
        if datapoint[:erna] in merged_titles:
            pass
        else:
            values = [data[1][i]]
            for j, label in enumerate(data[0]):
                if label[:erna] == datapoint[:erna] and label != datapoint:
                    values.append(data[1][j])
            mean = np.mean(values)
            std = np.std(values)
            merged_titles.append(datapoint[:erna])
            mean_tuples.append((mean, std))
            print(f"{merged_titles[-1]}: {mean}, {std} ({len(values)} pts.)")

    return merged_titles, mean_tuples


def generate_result_file(data: list[list], metadata=None, file=None, filetype="csv"):

    full_string = ""

    result_str = "Sample; Concentration / mol L-1; Std Dev Concentration / mol L-1\n"
    if filetype == "csv":
        metadata_json = json.dumps(metadata)
        for i, line in enumerate(data[0]):
            result_str += line + "; " + f"{data[1][i][0]:.10f}" + "; " + f"{data[1][i][1]:.10f}" + "\n"

        full_string = "---\n" + metadata_json + "\n---\n" + result_str

    elif filetype == "json":
        result_dict = {}
        for i, line in enumerate(data[0]):
            result_dict[line] = data[1][i]

        full_result = {
            "metadata": metadata,
            "samples": result_dict
        }

        full_string = json.dumps(full_result, indent=4)

    with open(file, "w") as writefile:
        writefile.write(full_string)


def example():
    """
    Demonstration of possible processing using this script. Will only work if example files are accessible.
    :return: None
    """

    files = [
        "Examples/HS_T001_K1.csv",
        "Examples/HS_T001_K2.csv",
        "Examples/HS_T001_K3.csv",
        "Examples/HS_T001_K4.csv",
        "Examples/HS_T001_K5.csv",
        "Examples/HS_T001_K6.csv",
        "Examples/HS_T001_K7.csv"
    ]

    concentrations = [0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001, 0.01]

    relevant_wavelengths = [258, 267]

    # plt.style.use("dark_background")

    for file in files:
        plot_full_spectrum(file=file)

    for wavelength in relevant_wavelengths:
        plt.vlines(wavelength, 0, 7, linestyles=":", color="#fa8174")

    plt.legend()
    plt.show()

    for entry in relevant_wavelengths:
        c, a = monochromatic_plot(entry, files, concentrations)
        fit_calibration(c, a, label=str(entry) + " nm")

    plt.legend()
    plt.show()


def main():
    """
    Define processing steps here. See example() for reference.
    :return: None
    """

    files = select_spectra_files()

    for file in files:
        plot_full_spectrum(file)
    plt.legend()
    plt.show()

    concentration_data = concentrations_from_spec_files(spec_files=files, wavelength=258, conversion_factor=298.1477)
    concentration_data = merge_multiple(concentration_data, erna=-6)

    metadata = {
        "pointsPerSample": 3,
        "wavelength_nm": 258,
        "temperature_C": 25,
        "parameterFile": "HS_T001.par",
        "conversionFactor": 298.1477,
        "calibrationExperiment": "HS_T001"
    }

    generate_result_file([concentration_data[0], concentration_data[1]], file="results.csv",
                         filetype="csv", metadata=metadata)


if __name__ == "__main__":
    main()
