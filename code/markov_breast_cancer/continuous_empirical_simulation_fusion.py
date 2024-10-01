import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import pandas as pd #type: ignore
from scipy.integrate import solve_ivp #type: ignore
from tqdm import tqdm  # type: ignore
import os

# Load data for breast cancer progression states
df = pd.read_csv('Data/brestCancer_markovData.csv')

# Extract states (columns in the CSV file) representing different cancer stages
states = df.columns.tolist()

# Extract transition matrix values from the DataFrame
P = df.values

# List of CSV files containing empirical transition times between stages
transition_files = [
    "temporal_0diagnosisTO1st.csv",
    "temporal_1stTO2nd.csv",
    "temporal_2ndTO3rd.csv",
    "temporal_3rdTO4th.csv",
    "temporal_4thTO5th.csv",
    "temporal_5thTO6th.csv",
    "temporal_6thTO7th.csv",
]

# Initial simulation parameters
initial_state = 0  # Start from the first state
limit_month = 360  # Limit simulation to 360 months (30 years)
step_time = 1  # Time step for evaluation (in months)
p0 = np.zeros(len(states))  # Initial probability vector, all zeros
p0[initial_state] = 1  # Set initial state probability to 1

num_paths = 10000  # Number of simulation paths
transition_times = []  # List to store transition times for each file

# Create an output directory to store the simulation results
output = f'fusion_empiral_continue_{num_paths}_paths_{limit_month}'
if not os.path.exists(output):
    os.makedirs(output)

# Load empirical distribution parameters
dist_data = pd.read_csv('temporal_distribution_results_exp/distribution_parameters_exp.csv')
params = eval(dist_data['params'][0])
loc = params[-2]  # Location parameter for the empirical distribution
scale = params[-1]  # Scale parameter

# Utility functions to handle file names and data conversion
def safe_file_name(name):
    return name.replace('/', '_').replace(' ', '_')

def conv_for_histo(data):
    # Convert data into a format usable for histograms
    data_conv = []
    for index, nPatient in enumerate(data):
        data_conv.extend(nPatient * [index])
    return data_conv

def sum_by_groups_of_12(df):
    # Group data by 12 months for better aggregation
    data_to_group = df.iloc[1:]  # Exclude the first row
    grouped_data = data_to_group.groupby(data_to_group.index // 12).sum().reset_index(drop=True)
    return grouped_data

# Function to generate transition times based on the empirical distribution
def generate_time_step(time_to_metastasis):
    return np.random.choice(time_to_metastasis)

# Load transition time data from CSV files
def load_transition_time(files):
    # Load transition data
    transition_time = pd.read_csv(f'Data/temporal_data/{files}')
    
    # Sum columns to get total patients per month
    transition_time['Total'] = transition_time.iloc[:, 1:].sum(axis=1)
    
    # Create a list of months where each month appears based on the number of patients
    time_to_metastasis = []
    for index, row in transition_time.iterrows():
        month = row['month']
        total_patients = row['Total']
        time_to_metastasis.extend([month] * total_patients)
    return time_to_metastasis

# Calculate the Q matrix for continuous time Markov process (used in Kolmogorov Forward Equations)
def calculate_Q_matrix(P, scale):
    n = P.shape[0]  # Number of states
    Q = np.zeros((n, n))  # Initialize Q matrix with zeros
    lambda_rates = 1 / scale  # Convert scale into exit rates (lambda)
    
    # Fill the Q matrix based on transition probabilities
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = lambda_rates * P[i, j]  # Off-diagonal elements
        Q[i, i] = -np.sum(Q[i, :])  # Diagonal elements ensure row sum is zero
    return Q

# Function to represent the Kolmogorov Forward Equations
def forward_equations(t, p, Q):
    return p.dot(Q)

# Empirical Simulation Part
############################################################################################################
# Initialize counters to track states at each month during the simulation
state_counts = np.zeros((limit_month + 1, len(states)), dtype=int)

# Load transition times for each phase
for k in transition_files:
    time_to_metastasis = load_transition_time(k)
    transition_times.append(time_to_metastasis)

# Run the empirical simulation for the specified number of paths
for j in tqdm(range(num_paths), desc="Simulation Progress"):
    num_previous_states = 0
    current_state = initial_state  # Start from the initial state
    current_time = 0
    states_by_time = []  # Track the state of the patient over time

    # Simulate each patient's path until they reach the terminal state or the time limit
    while current_state != len(states) - 1 and current_time != limit_month:
        previous_time = current_time
        previous_state = current_state
        time_step = generate_time_step(transition_times[num_previous_states])  # Get transition time
        
        # Update the current time and state if within the time limit
        if (previous_time + time_step) < limit_month:
            current_time += time_step
            for t in range(previous_time, current_time):
                states_by_time.append(current_state)
        else:
            current_time = limit_month
            for t in range(previous_time, limit_month + 1):
                states_by_time.append(current_state)

        # Transition to a new state based on transition probabilities
        current_state = np.random.choice(len(states), p=P[current_state])

        # If the terminal state is reached, fill the remaining time
        if (current_state == len(states) - 1) and (current_time < limit_month):
            for t in range(current_time, limit_month + 1):
                states_by_time.append(current_state)

        # Move to the next transition phase if the state changes
        if (num_previous_states < len(transition_files) - 1) and (previous_state != current_state):
            num_previous_states += 1

    # Update state counters based on simulation results
    for index, current_state in enumerate(states_by_time):
        state_counts[index, current_state] += 1

# Convert state counts into a DataFrame for clearer presentation
state_counts_df = pd.DataFrame(state_counts, columns=states)

############################################################################################################
# Kolmogorov Forward Equations

# Set up time span and evaluation points for the simulation
t_span = (0, limit_month)
t_eval = np.linspace(0, limit_month, int(limit_month / step_time))

# Calculate the Q matrix
Q = calculate_Q_matrix(P, scale)

# Solve the Kolmogorov Forward Equations using the initial probability vector p0
sol_forward = solve_ivp(forward_equations, t_span, p0, args=(Q,), t_eval=t_eval)

# Plot and compare empirical simulation results with the Kolmogorov forward equation results
for i, current_state in enumerate(states):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for the empirical histogram
    state_counts_df_histo = conv_for_histo(state_counts_df[current_state]) 
    counts, bins = np.histogram(state_counts_df_histo, bins=range(362))  # Create histogram bins

    # Normalize the histogram values
    normalized_counts = counts / num_paths

    # Plot the results from the forward equations
    ax.plot(sol_forward.t, sol_forward.y[i], color='red', label='Forward Kolmogorov Equations')
    
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Density of Paths')
    ax.set_title(f'Density of Paths for {current_state} forward')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(f'{output}/{safe_file_name(current_state)}_forward_histo.png')
    plt.close(fig)  # Close the figure to free memory

# Save forward equation results to CSV
sol_forward_df = pd.DataFrame(sol_forward.y.T, columns=states)
sol_forward_df.to_csv(f'{output}/forward_results.csv', index=False)
