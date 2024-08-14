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


def monochromatic_plot(selected_wavelength, files, concentrations=None, title=None):
    absorptions = []
    if concentrations is None:
        concentrations = []

    for file in files:
        wavelength_lst, absorption_lst = read_spectrum(file)
        relevant_index = wavelength_lst.index(selected_wavelength)
        absorptions.append(absorption_lst[relevant_index])
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


def main():

    files = [
        "HS_T001_S1.csv",
        "HS_T001_S2.csv",
        "HS_T001_S3.csv"
    ]

    for file in files:
        plot_full_spectrum(file=file)
    plt.legend()
    plt.show()

    files = [
        "HS_T001_K1.csv",
        "HS_T001_K2.csv",
        "HS_T001_K3.csv",
        "HS_T001_K4.csv",
        "HS_T001_K5.csv",
        "HS_T001_K6.csv",
        "HS_T001_K7.csv"
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


if __name__ == "__main__":
    main()
