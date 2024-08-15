import matplotlib.pyplot as plt
from datetime import datetime
import scipy.optimize as opt
import numpy as np
import json

plt.style.use("dark_background")


def exp_function(x, a, b, c, d):
    y = c - a * np.exp(-x * b + d)
    return y
  
  
def linear_function(x, a, b):
    y = a*x + b
    return y


def calculate_r_squared(x_data, y_data, optimized_parameters, function) -> float:
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    residuals = y_data - function(x_data, *optimized_parameters)
    ss_residuals = np.sum(residuals**2)
    ss_total = np.sum((y_data-np.mean(y_data))**2)

    r_squared = 1 - (ss_residuals / ss_total)

    return r_squared


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Die Datei {file_path} wurde nicht gefunden.")
    except json.JSONDecodeError:
        print(f"Die Datei {file_path} enthält kein gültiges JSON.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")


def process_data(x, y):
    moving_average = np.convolve(y, np.ones(25)/25, "full")
    
    new_x, new_y = ([], [])
    
    for i, value in enumerate(y):
        if abs(value - moving_average[i]) / value <= 0.03:
            new_x.append(x[i])
            new_y.append(value)

    return new_x, new_y
    

def process_data_t(x, y):
    new_x, new_y = ([], [])
    
    optimized_parameters, pcov = opt.curve_fit(exp_function, x, y, bounds=[-150, 150])
    
    fitted_y = exp_function(np.asarray(x), *optimized_parameters)
    
    for i, value in enumerate(y):
        if abs(value - fitted_y[i]) / optimized_parameters[2] <= 0.2:
            new_x.append(x[i])
            new_y.append(value)
    
    print(f"""Filtered out {len(x) - len(new_x)} from {len(x)} overall datapoints ({round((len(x) - len(new_x)) / len(x) * 100, 2)} %).""")
    
    return new_x, new_y


def update_plot(parameters, ax):
    
    smooth_fit = False
    
    ax.clear()
    
    with open("Examples/HS_F024_2.txt") as readfile:
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
                t, pf, m, fv = datetime.strptime(values[0], "%H:%M:%S"), float(values[1]), float(values[2]), float(values[3])
                exp_time.append(t)
                pump_flow.append(float(pf))
                mass.append(float(m))
                permeate_flow.append(fv)

    tim = []

    for t in exp_time:
        tim.append((t-exp_time[0]).total_seconds() / 60)
    
    permeate_flow_mLmin = []

    for val in permeate_flow:
        permeate_flow_mLmin.append(val*16.666)
    
    smoothing_factor = parameters["smoothing"]
    
    moving_avg_permeate_flow = np.convolve(permeate_flow_mLmin, np.ones(smoothing_factor)/smoothing_factor, "full")
    
    optimization_start = parameters["fit_start"]
    optimization_end = parameters["fit_end"]
    
    start = 0
    end = -1

    for t in tim:
        if t >= optimization_start:
            start = tim.index(t)
            break
    for t in tim:
        if t >= optimization_end:
            end = tim.index(t)
            break

    ax.plot(tim, permeate_flow_mLmin, "o", color="#8dd3c7")
    
    try:
        
        if smooth_fit:
            raise RuntimeError
        
        optimized_parameters, pcov = opt.curve_fit(exp_function, tim[start:end], permeate_flow_mLmin[start:end], bounds=[-150, 150])
        resolved_x = np.linspace(optimization_start, optimization_end + 30, 100)

        r_squared = calculate_r_squared(tim[start:end], permeate_flow_mLmin[start:end], optimized_parameters, exp_function)
    
        equilibrium_time = (- np.log((- 0.01 * optimized_parameters[2]) / optimized_parameters[0]) + optimized_parameters[3]) / optimized_parameters[1]
        equilibrium_time = round(equilibrium_time, 0)
        
        ax.plot(resolved_x, exp_function(np.asarray(resolved_x), *optimized_parameters), "-", color="#feffb3")
        
        ax.text(0.01, 0.95, f"$R^2$ = {round(r_squared, 4)}", transform=ax.transAxes)
        ax.text(0.01, 0.90, f"pred. $F_V$ = {round(optimized_parameters[2], 2)} mL / min", transform=ax.transAxes)
        ax.text(0.01, 0.85, f"pred. eq. time = {equilibrium_time} min", transform=ax.transAxes)
        
        ax.vlines(equilibrium_time, 0 , 10, linestyles="dotted")
        ax.hlines(optimized_parameters[2], optimization_start, equilibrium_time + 10, linestyles="dotted")
        
    except RuntimeError:
        try:
            optimized_parameters, pcov = opt.curve_fit(exp_function, tim[start:end], moving_avg_permeate_flow[start:end-smoothing_factor+1], bounds=[-150, 150])
            resolved_x = np.linspace(optimization_start, optimization_end + 30, 100)

            r_squared = calculate_r_squared(tim[start:end], moving_avg_permeate_flow[start:end-smoothing_factor+1], optimized_parameters, exp_function)
        
            equilibrium_time = (- np.log((- 0.01 * optimized_parameters[2]) / optimized_parameters[0]) + optimized_parameters[3]) / optimized_parameters[1]
            equilibrium_time = round(equilibrium_time, 0)
            
            ax.plot(resolved_x, exp_function(np.asarray(resolved_x), *optimized_parameters), "-", color="#feffb3")
            
            ax.text(0.01, 0.95, f"$R^2$ = {round(r_squared, 4)}", transform=ax.transAxes)
            ax.text(0.01, 0.90, f"pred. $F_V$ = {round(optimized_parameters[2], 2)} mL / min", transform=ax.transAxes)
            ax.text(0.01, 0.85, f"pred. eq. time = {equilibrium_time} min", transform=ax.transAxes)
            
            ax.vlines(equilibrium_time, 0 , 10, linestyles="dotted")
            ax.hlines(optimized_parameters[2], optimization_start, equilibrium_time + 10, linestyles="dotted")
            
            print("Performed fitting from moving average")
        except RuntimeError:
            print("Could not fit to function using the given data!")
    
    # proc_time, proc_flow = process_data_t(tim, permeate_flow_mLmin)
    
    ax.set_title("Fitting")
    ax.set_xlabel("$t - t_0$ / min")
    
    ax.plot(tim, moving_avg_permeate_flow[:-smoothing_factor+1], "-")
    # flow_ax.plot(proc_time, proc_flow, "-")
    
    ax.vlines(tim[start], 0, 10, linestyles="dotted", color="#fa8174")
    ax.vlines(tim[end], 0, 10, linestyles="dotted", color="#fa8174")       


def main():
    fig, ax = plt.subplots(1)

    param_dict = read_json_file("Parameters/permeability.json")

    update_plot(param_dict, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()

