import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure the file paths and names are correct
transition_files = [
    "temporal_0diagnosisTO1st.csv",
    "temporal_1stTO2nd.csv",
    "temporal_2ndTO3rd.csv",
    "temporal_3rdTO4th.csv",
    "temporal_4thTO5th.csv",
    "temporal_5thTO6th.csv",
    "temporal_6thTO7th.csv",
]

# Names used in the plot titles to represent the stages
names = ['diagnosis', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th']

# Create a directory to save the histogram data
output_dir = 'temporal_data_histo'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a list to store statistical summaries for each transition file
stats_list = []

# Loop through each transition file to process the data
for index_file, file_name in enumerate(transition_files):
    # Load the data from the CSV file
    data_load = pd.read_csv(f'Data/temporal_data/{file_name}')
    
    # Calculate the total number of patients per month (sum columns from the second onwards)
    data_load['total'] = data_load.iloc[:, 1:].sum(axis=1)
    
    # Extract the total patient data and the month information
    data = data_load['total']
    month = data_load['month']

    # Prepare a list to store the expanded data for histogram fitting
    data_fiting = []

    # Expand the data so each index repeats according to the number of patients
    for index, nPatient in enumerate(data):
        data_fiting.extend(nPatient * [index])

    # Calculate descriptive statistics
    mean_value = np.mean(data_fiting)
    median_value = np.median(data_fiting)
    std_value = np.std(data_fiting)
    min_value = np.min(data_fiting)
    max_value = np.max(data_fiting)

    # Add the statistics to the list
    stats_list.append({
        "File Name": file_name,
        "Mean": mean_value,
        "Median": median_value,
        "Std Dev": std_value,
        "Min": min_value,
        "Max": max_value
    })

    # Create the histogram plot
    plt.subplots(figsize=(10, 6))
    plt.hist(data_fiting, bins=range(max(month)), density=False, edgecolor='black', alpha=0.5)
    plt.xlabel('Month')
    plt.ylabel('Number of patients')
    plt.title(f'Time from {names[index_file]} progression to {names[index_file + 1]} progression')

    # Save the plot
    plt.savefig(f'{output_dir}/{file_name.split(".")[0]}_info_histo.png')
    plt.close()

# Convert the statistics list into a DataFrame
stats_df = pd.DataFrame(stats_list)

# Save the statistics DataFrame to a CSV file
stats_df.to_csv(f'{output_dir}/statistiques_descriptives.csv', index=False)

print("Descriptive statistics have been saved to", f'{output_dir}/statistiques_descriptives.csv')
