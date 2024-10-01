import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd # type: ignore
from tqdm import tqdm  # type: ignore
import os

# Load the transition matrix for breast cancer progression states from a CSV file
df = pd.read_csv('Data/brestCancer_markovData.csv')

# Extract the list of states (columns in the CSV file represent different cancer stages)
states = df.columns.tolist()

# Extract the transition matrix values from the DataFrame
P = df.values

# List of transition files containing empirical data for transition times
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
num_paths = 500  # Number of simulation paths
limit_time = 360  # Time limit for simulation in months (30 years)
transition_times = []  # List to store transition times for each file

# Create an output directory to save the simulation results
output_dir = f'simulation_results_histogramme/results_{num_paths}_path_histogramme'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def safe_file_name(name):
    return name.replace('/', '_').replace(' ', '_')

# Function to generate a transition time based on the empirical distribution
def generate_time_step(time_to_metastasis):
    return np.random.choice(time_to_metastasis)

# Function to load transition times from CSV files
def load_transition_time(files):
    # Load the transition data from a CSV file
    transition_time = pd.read_csv(f'Data/temporal_data/{files}')
    
    # Sum the last 4 columns to get the total number of patients per month
    transition_time['Total'] = transition_time.iloc[:, 1:].sum(axis=1)

    # Create a list where each month appears as many times as there are patients
    time_to_metastasis = []
    for index, row in transition_time.iterrows():
        month = row['month']
        total_patients = row['Total']
        time_to_metastasis.extend([month] * total_patients)
    return time_to_metastasis

# Initialize state counters for each month (for each transition phase)
states_for_1 = np.zeros((limit_time, len(states)), dtype=int)
states_for_2 = np.zeros((limit_time, len(states)), dtype=int)
states_for_3 = np.zeros((limit_time, len(states)), dtype=int)
states_for_4 = np.zeros((limit_time, len(states)), dtype=int)
states_for_5 = np.zeros((limit_time, len(states)), dtype=int)
states_for_6 = np.zeros((limit_time, len(states)), dtype=int)
states_for_7 = np.zeros((limit_time, len(states)), dtype=int)

# Load transition times for all the files
for k in transition_files:
    time_to_metastasis = load_transition_time(k)
    transition_times.append(time_to_metastasis)

# Run the simulation for the specified number of paths
for j in tqdm(range(num_paths), desc="Simulation Progress"):
    num_previous_states = 0  # Track the number of previous transitions
    current_state = initial_state  # Set the initial state
    current_time = 0  # Initialize the time counter

    # Simulate the patient's journey until reaching the terminal state ('Deceased') or the time limit
    while current_state != len(states) - 1 and current_time != limit_time:
        previous_time = current_time
        previous_state = current_state
        time_step = generate_time_step(transition_times[num_previous_states])

        # Ensure the next transition doesn't exceed the time limit
        if (previous_time + time_step) < limit_time:
            current_time += time_step
            current_state = np.random.choice(len(states), p=P[current_state])

            # Increment the counter for the corresponding transition state
            if num_previous_states == 0:
                states_for_1[current_time, current_state] += 1
            elif num_previous_states == 1:
                states_for_2[current_time, current_state] += 1
            elif num_previous_states == 2:
                states_for_3[current_time, current_state] += 1
            elif num_previous_states == 3:
                states_for_4[current_time, current_state] += 1
            elif num_previous_states == 4:
                states_for_5[current_time, current_state] += 1
            elif num_previous_states == 5:
                states_for_6[current_time, current_state] += 1
            else:
                states_for_7[current_time, current_state] += 1

        else:
            current_time = limit_time  # Cap the time at the limit if exceeded

        # Move to the next set of transition times if the state has changed
        if (num_previous_states < len(transition_files) - 1) and (previous_state != current_state):
            num_previous_states += 1

# Convert the state count matrices into DataFrames for clearer output
states_for_1_df = pd.DataFrame(states_for_1, columns=states)
states_for_2_df = pd.DataFrame(states_for_2, columns=states)
states_for_3_df = pd.DataFrame(states_for_3, columns=states)
states_for_4_df = pd.DataFrame(states_for_4, columns=states)
states_for_5_df = pd.DataFrame(states_for_5, columns=states)
states_for_6_df = pd.DataFrame(states_for_6, columns=states)
states_for_7_df = pd.DataFrame(states_for_7, columns=states)

# Save the simulation results to CSV files
states_for_1_df.to_csv(f'{output_dir}/simulation_relapse1.csv', index=False)
states_for_2_df.to_csv(f'{output_dir}/simulation_relapse2.csv', index=False)
states_for_3_df.to_csv(f'{output_dir}/simulation_relapse3.csv', index=False)
states_for_4_df.to_csv(f'{output_dir}/simulation_relapse4.csv', index=False)
states_for_5_df.to_csv(f'{output_dir}/simulation_relapse5.csv', index=False)
states_for_6_df.to_csv(f'{output_dir}/simulation_relapse6.csv', index=False)
states_for_7_df.to_csv(f'{output_dir}/simulation_relapse7.csv', index=False)
