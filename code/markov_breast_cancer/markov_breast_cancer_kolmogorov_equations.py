import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore  
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
import os 

# Load the transition matrix for breast cancer progression states from a CSV file
df = pd.read_csv('Data/brestCancer_markovData.csv')

# Extract the list of states (columns in the CSV file represent different cancer stages)
states = df.columns.tolist()

# Extract the transition matrix values from the DataFrame
P = df.values

# Create a directory to save the forward simulation results
output_fordward_dir = f'markov_with_Q_fordward_results'
if not os.path.exists(output_fordward_dir):
    os.makedirs(output_fordward_dir)

# Load distribution parameters for temporal data (assumed to be exponential distribution)
dist_data = pd.read_csv('temporal_distribution_results_exp/distribution_parameters_exp.csv')

# Extract the parameters for the distribution (location and scale)
params = eval(dist_data['params'][0])
loc = params[-2]  # Location parameter of the distribution
scale = params[-1]  # Scale parameter of the distribution

# Function to create a safe file name by replacing special characters
def safe_file_name(name):
    return name.replace('/', '_').replace(' ', '_')

# Function to calculate the generator matrix Q from the transition matrix P
# Q is computed by adjusting the transition matrix with exit rates (based on scale)
def calculate_Q_matrix(P, scale):
    n = P.shape[0]  # Number of states
    Q = np.zeros((n, n))  # Initialize Q matrix with zeros
    lambda_rates = 1 / scale  # Convert the scale into exit rates (lambda)
    
    # Fill the Q matrix based on transition probabilities and exit rates
    for i in range(n):
        for j in range(n):
            if i != j:  # Off-diagonal elements represent transitions between different states
                Q[i, j] = lambda_rates * P[i, j]
        Q[i, i] = -np.sum(Q[i, :])  # Ensure each row sums to zero (generator property)
    
    return Q

# Function representing the forward Kolmogorov equations
# This is the system of differential equations to be solved
def forward_equations(t, p, Q):
    return p.dot(Q)  # The rate of change of the probability vector p is p * Q

# Initial conditions for the forward equation simulation
initial_state = 0  # Initial state (the simulation starts from state 0)
limit_month = 120  # Time limit for the simulation (in months)
step_time = 1  # Step size for time intervals (in months)
p0 = np.zeros(len(states))  # Initialize the probability vector with zeros
p0[initial_state] = 1  # Set the initial probability to 1 for the initial state

# Time span for the simulation (0 to limit_month), and time points for evaluation
t_span = (0, limit_month)
t_eval = np.linspace(0, limit_month, int(limit_month / step_time))  # Time evaluation points

# Calculate the generator matrix Q using the transition matrix P and the scale parameter
Q = calculate_Q_matrix(P, scale)

# Solve the system of forward equations using solve_ivp
# This integrates the differential equations over the given time span
sol_forward = solve_ivp(forward_equations, t_span, p0, args=(Q,), t_eval=t_eval)

# Plot the results of the forward simulation
fig, ax1 = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(states) - 2))  # Generate distinct colors for the states

# Plot the probability evolution for each state (except "Deceased") on the left y-axis
for i, current_state in enumerate(states[1:len(states) - 1]):
    ax1.plot(sol_forward.t, sol_forward.y[i], label=current_state, color=colors[i % len(colors)])

# Set the labels, title, and grid for the left y-axis
ax1.set_xlabel('Time (months)')
ax1.set_ylabel('Probability')
ax1.set_title('Probabilities Over Time (Forward Equations)')
ax1.grid(True)

# Add a second y-axis for the "Deceased" state
ax2 = ax1.twinx()
ax2.plot(sol_forward.t, sol_forward.y[len(states) - 1], label="Deceased", color='black')  # Plot "Deceased" state
ax2.plot(sol_forward.t, sol_forward.y[0], label="Breast", color='green')  # Plot "Breast" state

# Combine the legends from both y-axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize='small', loc='best')

# Save the plot showing the probability evolution for all states over time
plt.savefig(f'{output_fordward_dir}/all_states_forward.png')
plt.show()

# Plot individual state evolution and save them
for i, state in enumerate(states):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sol_forward.t, sol_forward.y[i], label=state)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Density of Paths')
    ax.set_title(f'Density of Paths for {state} forward')
    plt.grid(True)
    plt.legend()
    
    # Save the individual plot for the current state
    plt.savefig(f'{output_fordward_dir}/{safe_file_name(state)}_forward.png')
    plt.close(fig)  # Close the figure after saving to free up memory

# Convert the solution from the forward equations into a DataFrame for saving
sol_forward_df = pd.DataFrame(sol_forward.y.T, columns=states)
# Save the results of the forward simulation to a CSV file
sol_forward_df.to_csv(f'{output_fordward_dir}/forward_results.csv', index=False)
