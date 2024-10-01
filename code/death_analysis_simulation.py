import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd # type: ignore
from tqdm import tqdm  # type: ignore
import os


# Data
df = pd.read_csv('Data/brestCancer_markovData.csv')

# Extraire les états (première ligne du fichier CSV)
states = df.columns.tolist()

# Extraire les valeurs de la matrice (le reste du fichier)
P = df.values

transition_files = [
    "temporal_0diagnosisTO1st.csv",
    "temporal_1stTO2nd.csv",
    "temporal_2ndTO3rd.csv",
    "temporal_3rdTO4th.csv",
    "temporal_4thTO5th.csv",
    "temporal_5thTO6th.csv",
    "temporal_6thTO7th.csv",
]

initial_state = 0
num_paths = 500
limit_time = 360
transition_times = []


output_dir = f'analyseDeath_{num_paths}_paths'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def safe_file_name(name):
    return name.replace('/', '_').replace(' ', '_')

def conv_for_histo(data):
    data_conv = []
    for index, nPatient in enumerate(data):
        data_conv.extend(nPatient * [index])
    return data_conv

def sum_by_groups_of_12(df):
    data_to_group = df.iloc[1:]  # Exclure la première ligne
    grouped_data = data_to_group.groupby(data_to_group.index // 12).sum().reset_index(drop=True)
    return grouped_data

# Fonction pour générer des temps de transition basés sur la distribution empirique
def generate_time_step(time_to_metastasis):
    return np.random.choice(time_to_metastasis)

def load_transition_time(files):
    # Chargement des données
    transition_time = pd.read_csv(f'Data/temporal_data/{files}')

    # Additionner les 4 dernières colonnes pour obtenir le nombre total de patients
    transition_time['Total'] = transition_time.iloc[:, 1:].sum(axis=1)

    # Créer une liste de mois, chaque mois apparaissant le nombre de fois correspondant au total des patients
    time_to_metastasis = []
    for index, row in transition_time.iterrows():
        month = row['month']
        total_patients = row['Total']
        time_to_metastasis.extend([month] * total_patients)
    return time_to_metastasis

# Initialiser les compteurs pour les états à chaque mois
state_counts = np.zeros((limit_time + 1, len(states) - 1), dtype=int)
num_living_after_end_simulation = np.zeros(len(states) - 1, dtype=int)

for k in transition_files:
    time_to_metastasis = load_transition_time(k)
    transition_times.append(time_to_metastasis)

for init_state in tqdm(range(len(states) - 1), desc="Simulation Progress"):

    # Simulation des chemins échantillons
    for j in range(num_paths):
        num_previous_states = 0
        current_state = init_state 
        current_time = 0
        path_still_alive = num_paths

        while current_state != len(states) - 1 and current_time != limit_time:  # Jusqu'à atteindre 'Deceased' ou fin de la simu
            state_counts[current_time, init_state] += path_still_alive
            
            previous_time = current_time
            previous_state = current_state
            time_step = generate_time_step(transition_times[num_previous_states])
            
            if current_time + time_step <= limit_time:
                current_time += time_step
                current_state = np.random.choice(len(states), p=P[current_state])

                if current_state == len(states) - 1:
                    path_still_alive -= 1  # Réduire le nombre de patients encore en vie

                if (num_previous_states < len(transition_files) - 1) and (previous_state != current_state):
                    num_previous_states += 1

            else: 
                current_time = limit_time
                num_living_after_end_simulation[init_state] += path_still_alive  # Enregistrer les patients encore en vie à la fin de la simulation
            
        state_counts[current_time, init_state] += path_still_alive  # Ajouter le nombre de patients encore en vie à la fin de la simulation


# Convertir en DataFrame pour affichage plus clair
state_counts_df = pd.DataFrame(state_counts, columns=states[:-1])
num_living_after_end_simulation_df = pd.DataFrame(num_living_after_end_simulation, index=states[:-1])

# Enregistrer les données au format CSV
state_counts_df.to_csv(f'{output_dir}/analyseLivingData.csv', index=False)
num_living_after_end_simulation_df.to_csv(f'{output_dir}/num_living_after_end_simulation.csv', index=True)

for i, current_state in enumerate(states[:-1]):
    fig, ax = plt.subplots(figsize=(10, 6))

    state_counts_df_years = sum_by_groups_of_12(state_counts_df)
    state_counts_df_histo = conv_for_histo(state_counts_df_years[current_state]) 

    # Calculer l'histogramme sans afficher
    counts, bins = np.histogram(state_counts_df_histo, bins=range(32))  # vous pouvez ajuster le nombre de bins selon votre besoin

    # Normaliser les valeurs de l'histogramme
    normalized_counts = counts / num_paths

    ax.bar(bins[:-1], normalized_counts, width=(bins[1]-bins[0]), align='edge', edgecolor='black', alpha=0.7)
    #ax.hist(state_counts_df_histo, bins=range(32), alpha=0.7, edgecolor='black', density=False)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Number of Paths')
    ax.set_title(f'{current_state}')
    #plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/{safe_file_name(current_state)}.png')  # Nom du fichier basé sur l'état
    plt.close(fig)  # Fermer la figure après la sauvegarde pour libérer la mémoire




