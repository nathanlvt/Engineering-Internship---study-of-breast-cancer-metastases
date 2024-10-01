import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import scipy.stats # type: ignore
import os

# List of CSV files containing temporal transition data
transition_files = [
    "temporal_0diagnosisTO1st.csv",
    "temporal_1stTO2nd.csv",
    "temporal_2ndTO3rd.csv",
    "temporal_3rdTO4th.csv",
    "temporal_4thTO5th.csv",
    "temporal_5thTO6th.csv",
    "temporal_6thTO7th.csv",
]

# Create a DataFrame to store the fitting results: file name, distribution parameters, MSE, and p-value
results_df = pd.DataFrame(columns=['file_name', 'params', 'mse', 'p_value'])

# Check if the output directory for the distribution results exists, otherwise create it
output_dir = 'temporal_distribution_results_exp'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each transition file to process data and fit an exponential distribution
for index, file_name in enumerate(transition_files):
    # Load the CSV data for the transition file
    data_load = pd.read_csv(f'Data/temporal_data/{file_name}')
    
    # Sum the columns (starting from the second) to get the total number of patients per month
    data_load['total'] = data_load.iloc[:, 1:].sum(axis=1)
    
    # Extract the 'total' patient data and the 'month' information
    data = data_load['total']
    month = data_load['month']

    # Prepare a list to store the expanded data for fitting
    data_fiting = []

    # Expand the data so each index repeats according to the number of patients
    for index, nPatient in enumerate(data):
        data_fiting.extend(nPatient * [index])

    # Normalize the data to get the density values
    total_values = sum(data)
    density = data / total_values  # Density of patients over time

    # Define the x-values for plotting the fitted PDF
    x_for_pdf = np.linspace(0, max(month), 1000)

    # Fit the data to an exponential distribution
    dist = scipy.stats.expon
    params = dist.fit(data_fiting)  # Fit the exponential distribution to the data
    D, p_value = scipy.stats.kstest(data_fiting, 'expon', args=params)  # Perform the Kolmogorov-Smirnov test

    loc, scale = params  # Exponential distribution has two parameters: loc and scale
    pdf = dist.pdf(x_for_pdf, loc=loc, scale=scale)  # Calculate the fitted PDF based on the parameters
    
    # Format the label for the plot
    if loc != 0:
        label = f'${loc:+.2f}+\exp(-{scale:.2f}t)$' 
    else:
        label = f'$\exp(-{scale:.2f}t)$'

    # Calculate the Mean Squared Error (MSE) between the predicted and actual densities
    predicted_density = dist.pdf(month, loc=loc, scale=scale)
    mse = np.mean((predicted_density - density)**2)

    # Append the results (file name, parameters, MSE, p-value) to the results DataFrame
    new_row = pd.DataFrame({
        'file_name': [file_name],
        'params': [params],
        'mse': [mse],
        'p_value': [p_value]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Create the histogram and plot the fitted PDF
    plt.subplots(figsize=(10, 6))
    plt.hist(data_fiting, bins=range(max(month)), density=True, edgecolor='black', alpha=0.5)  # Histogram of the empirical data
    plt.plot(x_for_pdf, pdf, 'r-', label=label)  # Plot the fitted PDF
    plt.xlabel('Month')
    plt.ylabel('Density')
    plt.legend()

    # Save the plot as an image file
    plt.savefig(f'{output_dir}/{file_name.split(".")[0]}_dist_exp.png')
    plt.close()

# Save the distribution fitting results into a CSV file
results_df.to_csv(f'{output_dir}/distribution_parameters_exp.csv', index=False)
