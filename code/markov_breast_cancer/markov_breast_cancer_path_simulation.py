import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import pandas as pd  # For data handling (CSV reading, DataFrame manipulation)
from scipy.stats import wald, pareto, alpha, moyal, genexpon  # Optional statistical distributions, currently not used

# Load the transition matrix from a CSV file containing Markov data
df = pd.read_csv('Data/brestCancer_markovData.csv')

# List of files with temporal transition data
transition_files = [
    "temporal_0diagnosisTO1st.csv",
    "temporal_1stTO2nd.csv",
    "temporal_2ndTO3rd.csv",
    "temporal_3rdTO4th.csv",
    "temporal_4thTO5th.csv",
    "temporal_5thTO6th.csv",
    "temporal_6thTO7th.csv",
]

# Extract the states (the first row/column of the CSV file contains the state names)
states = df.columns.tolist()

# Extract the values of the transition matrix from the DataFrame (the rest of the file)
P = df.values

# Initial conditions and simulation parameters
initial_state = 1  # Starting from the second state (index 1)
num_paths = 4  # Number of paths (simulated sequences)
num_transition_files = 7  # Number of transition files used
limit_time = 120  # Maximum simulation time (e.g., months)
transition_times = []  # Placeholder for storing the transition times

# Helper function to convert a time in months to years/months format
def convert_months_to_years_months(months):
    years = months // 12  # Whole years
    remaining_months = months % 12  # Remaining months
    return f"{years}/{remaining_months+1:02d}"  # Return formatted string

# Function to generate a time step randomly based on the transition time distribution
def generate_time_step(time_to_metastasis):
    return np.random.choice(time_to_metastasis)

# Load transition times from CSV files
def load_transition_time(files):
    # Load transition data from a CSV file
    transition_time = pd.read_csv(f'Data/temporal_data/{files}')

    # Sum the last 4 columns to calculate the total number of patients in each month
    transition_time['Total'] = transition_time.iloc[:, 1:].sum(axis=1)

    # Create a list where each month is repeated by the number of patients
    time_to_metastasis = []
    for index, row in transition_time.iterrows():
        month = row['month']
        total_patients = row['Total']
        time_to_metastasis.extend([month] * total_patients)  # Add the month as many times as there are patients
    return time_to_metastasis

# Loop through each transition file to load the corresponding transition times
for k in transition_files:
    time_to_metastasis = load_transition_time(k)
    transition_times.append(time_to_metastasis)

# Create the figure and axis for plotting the simulated paths
fig, ax = plt.subplots(figsize=(10, 6))

# Use a default color map for distinguishing paths
colors = plt.cm.tab10.colors  # 'tab10' color palette provides 10 distinct colors

# Simulate the paths (sample sequences of state transitions)
all_times = []  # To store all unique times for labeling the x-axis
path_colors = {}  # To store the color for each time point on the paths

for j in range(num_paths):
    current_time = 0  # Start time
    current_state = initial_state  # Start from the initial state
    times = [current_time]  # List of times for the current path
    states_history = [current_state]  # List of states visited during the path
    num_previous_states = 0  # Track which transition file to use

    # Simulate until the terminal state or the time limit is reached
    while current_state != len(states) - 1 and current_time != limit_time:
        previous_time = current_time
        previous_state = current_state
        # Generate a time step using the transition times for the current state
        time_step = generate_time_step(transition_times[num_previous_states])

        # Ensure the next transition doesn't exceed the time limit
        if (previous_time + time_step) <= limit_time:
            current_time += time_step  # Advance time by the time step
            # Move to the next state based on the transition probabilities from the current state
            current_state = np.random.choice(len(states), p=P[current_state])
        else:
            current_time = limit_time  # Cap the time at the limit if exceeded
        
        # Store the time and state history
        times.append(current_time)
        states_history.append(current_state)

        # Move to the next transition phase if more transition files are available
        if num_previous_states < len(transition_files) - 1:
            num_previous_states += 1

    # Store all times for axis labels and associate path times with colors
    all_times.extend(times)
    path_colors.update({time: colors[j % len(colors)] for time in times})
    # Plot the path as a step function with transitions shown as steps and dots at transition points
    ax.step(times, states_history, where='post', linewidth=2, color=colors[j % len(colors)], label=f'Path {j+1}')
    ax.plot(times, states_history, 'o', color=colors[j % len(colors)])  # Add transition points as dots

# Customize the x-axis labels (convert time to years/months)
unique_times = sorted(set(all_times))
labels = [convert_months_to_years_months(int(time)) for time in unique_times]

ax.set_xlabel('Time (year/month)')
ax.set_ylabel('State')
ax.set_yticks(range(len(states)))
ax.set_yticklabels(states)

# Set the x-axis ticks and labels
ax.set_xticks(unique_times)
ax.set_xticklabels(labels)

# Color the x-axis labels based on the path color
for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
    label.set_color(path_colors[tick])

# Add a legend for the different paths
ax.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
# Add a grid to improve readability
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
# Display the plot
plt.show()
