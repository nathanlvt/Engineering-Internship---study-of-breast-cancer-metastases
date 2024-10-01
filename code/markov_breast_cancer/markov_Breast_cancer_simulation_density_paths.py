import numpy as np  # type: ignore  # Import for numerical operations
import matplotlib.pyplot as plt  # type: ignore  # Import for plotting
import pandas as pd  # type: ignore  # Import for data manipulation (reading CSV files)
from tqdm import tqdm  # type: ignore  # Import for creating a progress bar during simulation
import os  # Import for file handling (directories and file paths)

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
num_paths = 10000  # Number of simulation paths
limit_time = 120  # Time limit for simulation in months
transition_times = []  # List to store transition times for each file

# Create an output directory to save the simulation results
output_dir = f'simulation_results_{num_paths}_paths_empirical'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to create a safe file name by replacing special characters
def safe_file_name(name):
    return name.replace('/', '_').replace(' ', '_')

# Function to generate a transition time based on the empirical distribution
def generate_time_step(time_to_metastasis):
    return np.random.choice(time_to_metastasis)

# Function to load empirical transition times from the files
def load_transition_time(files):
    # Load the transition data from the CSV file
    transition_time = pd.read_csv(f'Data/temporal_data/{files}')

    # Sum the last 4 columns to get the total number of patients in each month
    transition_time['Total'] = transition_time.iloc[:, 1:].sum(axis=1)

    # Create a list where each month appears as many times as there are patients
    time_to_metastasis = []
    for index, row in transition_time.iterrows():
        month = row['month']
        total_patients = row['Total']
        time_to_metastasis.extend([month] * total_patients)
    return time_to_metastasis

# Initialize a matrix to keep track of state counts at each month
state_counts = np.zeros((limit_time + 1, len(states)), dtype=int)

# Load transition times for all the files
for k in transition_files:
    time_to_metastasis = load_transition_time(k)
    transition_times.append(time_to_metastasis)

# Run the simulation for the specified number of paths
for j in tqdm(range(num_paths), desc="Simulation Progress"):
    num_previous_states = 0  # Track the number of previous transitions
    current_state = initial_state  # Set the initial state
    current_time = 0  # Initialize the time counter
    states_by_time = []  # List to keep track of states over time

    # Simulate until reaching the terminal state or time limit
    while current_state != len(states) - 1 and current_time != limit_time:
        previous_time = current_time
        previous_state = current_state
        time_step = generate_time_step(transition_times[num_previous_states])

        # Ensure the time step does not exceed the time limit
        if (previous_time + time_step) < limit_time:
            current_time += time_step
            # Record the current state for each time unit
            for t in range(previous_time, current_time):
                states_by_time.append(current_state)
        else:
            current_time = limit_time
            for t in range(previous_time, limit_time + 1):
                states_by_time.append(current_state)
        
        # Choose the next state based on transition probabilities
        current_state = np.random.choice(len(states), p=P[current_state])

        # If the terminal state is reached before the time limit, extend it to the time limit
        if current_state == len(states) - 1 and current_time < limit_time:
            for t in range(current_time, limit_time + 1):
                states_by_time.append(current_state)

        # Move to the next set of transition times if the state has changed
        if num_previous_states < len(transition_files) - 1 and previous_state != current_state:
            num_previous_states += 1

    # Update state counts based on the path
    for index, state in enumerate(states_by_time):
        state_counts[index, state] += 1

# Convert the state counts matrix into a DataFrame for better readability
state_counts_df = pd.DataFrame(state_counts, columns=states)

# Calculate the density of paths by dividing by the total number of paths
state_density_df = state_counts_df / num_paths

# Plot the density of paths over time for each state (except 'Deceased')
fig, ax1 = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(states) - 2))

# Plot the state densities
for i, current_state in enumerate(states[1:len(states) - 1]):
    ax1.plot(state_density_df.index, state_density_df[current_state], label=current_state, color=colors[i % len(colors)])

# Customize x and y labels, grid, and legend
ax1.set_xlabel('Time (months)')
ax1.set_ylabel('Density of Paths')
ax1.grid(True)

# Add a second y-axis for the 'Deceased' state
ax2 = ax1.twinx()
ax2.plot(state_density_df.index, state_density_df["Deceased"], label="Deceased", color='black')
ax2.plot(state_density_df.index, state_density_df["Breast"], label="Breast", color='green')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize='small', loc='best')

# Save the combined plot
plt.savefig(f'{output_dir}/all_states_over_time.png')
plt.show()

# Generate and save individual plots for each state
for i, current_state in enumerate(states):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(state_density_df.index, state_density_df[current_state], label=current_state)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Density of Paths')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/{safe_file_name(current_state)}_over_time.png')  # Safe file name for saving
    plt.close(fig)  # Close figure to free memory

# Save the state densities to a CSV file
state_density_df.to_csv(f'{output_dir}/simulation_{num_paths}_path_empirical.csv', index=False)
