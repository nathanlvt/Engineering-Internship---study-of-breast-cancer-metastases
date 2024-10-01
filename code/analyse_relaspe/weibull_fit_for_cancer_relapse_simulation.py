import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import os
from tqdm import tqdm # type: ignore
from scipy.optimize import curve_fit # type: ignore
from scipy.stats import kstest # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore


def sum_by_groups_of_12(df):
    # Group data by 12 months (excluding the first row)
    data_to_group = df.iloc[1:]  
    grouped_data = data_to_group.groupby(data_to_group.index // 12).sum().reset_index(drop=True)
    return grouped_data

num_paths = 500

# Load data for different relapse stages (simulated)
relapse1 = pd.read_csv(f'simulation_results_histogramme/results_{num_paths}_path_histogramme/simulation_relapse1.csv')
relapse2 = pd.read_csv(f'simulation_results_histogramme/results_{num_paths}_path_histogramme/simulation_relapse2.csv')
relapse3 = pd.read_csv(f'simulation_results_histogramme/results_{num_paths}_path_histogramme/simulation_relapse3.csv')
relapse4 = pd.read_csv(f'simulation_results_histogramme/results_{num_paths}_path_histogramme/simulation_relapse4.csv')
relapse5 = pd.read_csv(f'simulation_results_histogramme/results_{num_paths}_path_histogramme/simulation_relapse5.csv')
relapse6 = pd.read_csv(f'simulation_results_histogramme/results_{num_paths}_path_histogramme/simulation_relapse6.csv')
relapse7 = pd.read_csv(f'simulation_results_histogramme/results_{num_paths}_path_histogramme/simulation_relapse7.csv')

states = relapse1.columns.tolist()  # Extract state labels from relapse1

all_relapses = [relapse1, relapse2, relapse3, relapse4, relapse5, relapse6, relapse7]

# Reduce data by grouping it into yearly intervals
all_relapses_reduced = [sum_by_groups_of_12(relapse) for relapse in all_relapses]

# Sum all relapse data to get the total number of transitions across all relapses
data_relapse = np.add(np.add(np.add(np.add(np.add(np.add(all_relapses_reduced[0], all_relapses_reduced[1]), all_relapses_reduced[2]), all_relapses_reduced[3]), all_relapses_reduced[4]), all_relapses_reduced[5]), all_relapses_reduced[6])

# Create output directory for graphs
output_dir = f'simulation_results_histogramme/graphs_by_metastase_{num_paths}_path'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

base_colors = ['blue', 'green', 'red', 'magenta', 'yellow', 'black', 'gray']
colors = sns.color_palette(base_colors, desat=0.7)  # Set color palette for plotting

def count_for_label(all_relapses_reduced, state):
    # Count the total number of patients per state across all relapses
    count_by_relaspes = []
    for relapse in all_relapses_reduced:
        relapse_count = relapse[state].sum()
        count_by_relaspes.append(relapse_count)
    return count_by_relaspes

def safe_file_name(name):
    # Create safe file names by replacing special characters
    return name.replace('/', '_').replace(' ', '_')

def conv_data_fitting(data):
    # Expand data for fitting by repeating the index according to the number of patients
    data_fiting = []
    for index, nPatient in enumerate(data):
        data_fiting.extend(nPatient * [index])
    return data_fiting

def conv_data_relapse(all_relapses, state):
    # Extract and convert state data for all relapses
    extracted_columns = []
    extracted_columns_conv = []
    count_by_relaspes = []

    for relapse in all_relapses:
        extracted_columns.append(relapse[state].values) 

    for i in range(len(extracted_columns)):
        conv_column = conv_data_fitting(extracted_columns[i])
        extracted_columns_conv.append(conv_column) 
    
    return extracted_columns_conv

# Define the Weibull function for curve fitting
def weibull_function(x, c, loc, scale):
    y = []
    for i in range(len(x)):
        if x[i] < loc:
            y.append(0)  # If x is less than the location parameter, the probability is zero
        else:
            y.append((c / scale) * (((x[i] - loc) / scale) ** (c-1)) * np.exp(-((x[i] - loc) / scale) ** c))
    return y

# Initialize a list to store results
results = []

# Loop through each state to generate histograms and fit Weibull distributions
for state in tqdm(states[1:], 'Plotting histograms'):
    relapse_by_state = conv_data_relapse(all_relapses_reduced, state)
    data_relapse_fitting = conv_data_fitting(data_relapse[state])

    # Calculate the empirical histogram
    hist_values, bin_edges = np.histogram(data_relapse_fitting, bins=range(31), density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Calculate the center of each bin

    # Set initial parameters for Weibull fitting based on the state
    if state == 'Bones':
        p0 = [1.19, 0.0215, 6.9]
    elif state == 'Chest Wall':
        p0 = [1.12, -0.0475, 6.3]
    elif state == 'Lung/pleura':
        p0 = [1.35, -0.0557, 7.61]
    elif state == 'Liver':
        p0 = [1.2, 0.00928, 8.19]
    else:
        p0 = [1.2, 0, 7]

    # Perform Weibull fitting using curve_fit
    params, _ = curve_fit(weibull_function, bin_centers, hist_values, p0=p0, maxfev=10000)
    c, loc, scale = params

    # Calculate the fitted probability density function (PDF)
    x_fit = np.linspace(0, 30, 1000)
    pdf_curve_fit = weibull_function(x_fit, c, loc, scale)
    
    # Perform Kolmogorov-Smirnov test
    D, p_value = kstest(data_relapse_fitting, 'weibull_min', args=(c, loc, scale))

    # Calculate the MSE between the empirical histogram and the fitted Weibull distribution
    pdf_hist = weibull_function(bin_centers, c, loc, scale)
    mse = mean_squared_error(hist_values, pdf_hist)

    # Store results
    results.append([state, c, loc, scale, mse, p_value])

    count_label = count_for_label(all_relapses_reduced, state)
    label_hist = [f"Relapse {i+1}: {count_label[i]}" for i in range(7)]

    # Generate the label for the fitted Weibull function
    cdivScale = c / scale
    if loc == 0:
        label_fit = f'$f(x) = {cdivScale:.2f} \\left(\\dfrac{{x}}{{{scale:.2f}}}\\right)^{{{c-1:.2f}}} \\exp\\left(-\\left(\\dfrac{{x}}{{{scale:.2f}}}\\right)^{{{c:.2f}}}\\right), \\; x \\geq {loc:.2f}$\n     $\\,= 0, \\; x < {loc:.2f}$'
    elif loc < 0:
        label_fit = f'$f(x) = {cdivScale:.2f} \\left(\\dfrac{{x + {-loc:.2f}}}{{{scale:.2f}}}\\right)^{{{c-1:.2f}}} \\exp\\left(-\\left(\\dfrac{{x + {-loc:.2f}}}{{{scale:.2f}}}\\right)^{{{c:.2f}}}\\right), \\; x \\geq {loc:.2f}$\n     $\\,= 0, \\; x < {loc:.2f}$'
    else:
        label_fit = f'$f(x) = {cdivScale:.2f} \\left(\\dfrac{{x - {loc:.2f}}}{{{scale:.2f}}}\\right)^{{{c-1:.2f}}} \\exp\\left(-\\left(\\dfrac{{x - {loc:.2f}}}{{{scale:.2f}}}\\right)^{{{c:.2f}}}\\right), \\; x \\geq {loc:.2f}$\n     $\\,= 0, \\; x < {loc:.2f}$'

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(relapse_by_state, bins=range(31), alpha=0.7, edgecolor='black', color=colors, stacked=True, density=True, label=label_hist)
    plt.plot(x_fit, pdf_curve_fit, 'r-', label=label_fit + f'\nRMS Error = {mse:.4f}')
    plt.xlabel('Years') 
    plt.ylabel('Number of mets. (density)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{output_dir}/{safe_file_name(state)}_histogramme_with_weibull_fit.png')
    plt.close()

# Save the Weibull fitting results in a CSV file
results_df = pd.DataFrame(results, columns=['State', 'Shape (c)', 'Location (loc)', 'Scale (scale)', 'MSE', 'P-value'])
results_csv_path = os.path.join(output_dir, 'weibull_fit_results.csv')
results_df.to_csv(results_csv_path, index=False)

print(f'Results saved to {results_csv_path}')
