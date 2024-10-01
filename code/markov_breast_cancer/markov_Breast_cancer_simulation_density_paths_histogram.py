import numpy as np  # type: ignore  # Import for numerical operations
import matplotlib.pyplot as plt  # type: ignore  # Import for plotting
import pandas as pd  # type: ignore  # Import for data handling (CSV reading, DataFrame manipulation)
from tqdm import tqdm  # type: ignore  # Import for creating a progress bar during simulation
import os  # Import for file and directory handling

# Load the transition matrix for breast cancer progression states from a CSV file
df = pd.read_csv('Data/brestCancer_markovData.csv')

# Extract the list of states (columns in the CSV file represent different cancer stages)
states = df.columns.tolist()

# Extract the transition matrix values from the DataFrame
P = df.values

# List of files containing empirical data for transition times
transition_files = [
    "temporal_0diagnosisTO1st.csv",
    "temporal_1stTO2nd.csv",
    "temporal_2ndTO3rd.csv",
    "temporal_3rdTO4th.csv",
    "temporal_4thTO5th.csv",
    "temporal_5thTO6th.csv",
    "temporal_6thTO7th.csv",
]

# Set initial simulation parameters
initial_state = 0  # The starting state (initial condition)
num_paths = 10000  # Number of simulation paths to generate
limit_time = 360  # Time limit for simulation in months (30 years)
transition_times = []  # List to store transition times for each file

# Create an output directory to save the simulation results
output_dir = f'simulation_results_{num_paths}_paths_empirical_histo_{limit_time}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to safely generate file names by replacing special characters
def safe_file_name(name):
    return name.replace('/', '_').replace(' ', '_')

# Function to convert state count data into a format suitable for histograms
# It expands the count data into a list of occurrences
def conv_for_histo(data):
    data_conv = []
    for index, nPatient in enumerate(data):
        data_conv.extend(nPatient * [index])  # For each index, append that many occurrences of the state
    return data_conv

# Function to sum the state counts by groups of 12 months (to group data by year)
def sum_by_groups_of_12(df):
    data_to_group = df.iloc[1:]  # Exclude the first row (initial state)
    # Group by year (12 months per group), summing up the states within each group
    grouped_data = data_to_group.groupby(data_to_group.index // 12).sum().reset_index(drop=True)
    return grouped_data

# Function to generate a random transition time from the empirical distribution
def generate_time_step(time_to_metastasis):
    return np.random.choice(time_to_metastasis)

# Function to load empirical transition times from CSV files
def load_transition_time(files):
    # Load the transition data from a CSV file
    transition_time = pd.read_csv(f'Data/temporal_data/{files}')

    # Sum the last 4 columns to get the total number of patients in each month
    transition_time['Total'] = transition_time.iloc[:, 1:].sum(axis=1)

    # Create a list where each month appears as many times as there are patients
    time_to_metastasis = []
    for index, row in transition_time.iterrows():
        month = row['month']
        total_patients = row['Total']
        time_to_metastasis.extend([month] * total_patients)  # Append each month as many times as there are patients
    return time_to_metastasis

# Initialize a matrix to count the states at each month
state_counts = np.zeros((limit_time + 1, len(states)), dtype=int)

# Load the transition times for all files
for k in transition_files:
    time_to_metastasis = load_transition_time(k)
    transition_times.append(time_to_metastasis)

# Run the simulation for the specified number of paths
for j in tqdm(range(num_paths), desc="Simulation Progress"):
    num_previous_states = 0  # Track the number of previous transitions
    current_state = initial_state  # Set the initial state
    current_time = 0  # Initialize the time counter
    states_by_time = []  # List to keep track of states over time

    # Simulate the patient's journey until reaching the terminal state ('Deceased') or the time limit
    while current_state != len(states) - 1 and current_time != limit_time:
        previous_time = current_time
        previous_state = current_state
        time_step = generate_time_step(transition_times[num_previous_states])  # Generate a transition time
        
        # Ensure the next transition doesn't exceed the time limit
        if (previous_time + time_step) < limit_time:
            current_time += time_step  # Advance the time by the time step
            # Record the current state for each time unit until the transition
            for t in range(previous_time, current_time):
                states_by_time.append(current_state)
        else:
            current_time = limit_time  # Cap the time at the limit if exceeded
            for t in range(previous_time, limit_time + 1):
                states_by_time.append(current_state)

        # Move to the next state based on transition probabilities
        current_state = np.random.choice(len(states), p=P[current_state])

        # If the terminal state ('Deceased') is reached before the time limit, extend it to the time limit
        if current_state == len(states) - 1 and current_time < limit_time:
            for t in range(current_time, limit_time + 1):
                states_by_time.append(current_state)

        # Move to the next set of transition times if the state has changed
        if num_previous_states < len(transition_files) - 1 and previous_state != current_state:
            num_previous_states += 1

    # Update state counts for each time step along the path
    for index, state in enumerate(states_by_time):
        state_counts[index, state] += 1

# Convert the state counts matrix into a DataFrame for better readability
state_counts_df = pd.DataFrame(state_counts, columns=states)

# Generate histograms for each state and save them
for i, current_state in enumerate(states):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert the state count data for the histogram
    state_counts_df_histo = conv_for_histo(state_counts_df[current_state])

    # Compute the histogram without displaying it
    counts, bins = np.histogram(state_counts_df_histo, bins=range(362))

    # Normalize the histogram counts by the total number of paths
    normalized_counts = counts / num_paths

    # Plot the histogram as bars
    ax.bar(bins[:-1], normalized_counts, width=(bins[1] - bins[0]), align='edge', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Number of Paths')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the histogram plot for the current state
    plt.savefig(f'{output_dir}/{safe_file_name(current_state)}_over_time.png')
    plt.close(fig)  # Close the figure to free memory

# Group the state counts by year (12 months per group) and sum them
state_counts_df_years = sum_by_groups_of_12(state_counts_df)

# Normalize the state counts by the total number of paths
state_counts_normalized = state_counts_df_years / num_paths

# Save the normalized state counts to a CSV file
state_counts_normalized.to_csv(f'{output_dir}/simulation_{num_paths}_path_empirical.csv', index=False)
