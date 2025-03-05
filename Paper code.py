#------------------------------------------------------------------------------
#IMPORT PACKAGES
#------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
#------------------------------------------------------------------------------
#STIMULUS LABELLING
#------------------------------------------------------------------------------
# Load the Excel file containing the stimulus information
excel_data_stim = pd.read_excel(r'Stimuli/stimuli.xlsx', sheet_name = None)
stimulus_data = pd.concat(excel_data_stim.values(), ignore_index=True)

#The stim_id will label the order of appearence of the stimuli
stimulus_data['stim_id'] = 0
bird_id = stimulus_data.bird.unique()
for bird in bird_id:
    current_bird = stimulus_data[stimulus_data.bird == bird]
    exp_id = current_bird.phase.unique()
    for experiment in exp_id:
        current_experiment = current_bird[current_bird.phase == experiment]
        trial_id = current_experiment.trial.unique()
        for num_trial in trial_id:
            current_trial = current_experiment[current_experiment.trial == num_trial]
            i = 0 #counter for the stimulus
            for index, stimulus in current_trial.iterrows():
                if stimulus['freq'] >= 10000:
                    i+=1 
                else:
                    stimulus_data.at[index,'stim_id'] = i
#Get the starting times for each stimulus
stimulus_info = [] # Initialize an empty list to store the data

for bird in bird_id:
    current_bird = stimulus_data[stimulus_data.bird == bird]
    exp_id = current_bird.phase.unique()
    for experiment in exp_id:
        current_experiment = current_bird[current_bird.phase == experiment]
        trial_id = current_experiment.trial.unique()
        for num_trial in trial_id:
            current_trial = current_experiment[current_experiment.trial == num_trial]                
            stimulus_id = current_trial.stim_id.unique()
            for s in stimulus_id:
                if s == 0: #starting pitch
                    continue #skip the zero index for the stimulus   
                current_stim = current_trial[current_trial.stim_id == s]
                temp = current_stim.t_i - current_stim.t_f.shift(1)
                gaps_stim = temp.shift(-1)
                gap_stim = gaps_stim.iloc[-2]
                first_whistle = current_stim.iloc[0]
                last_whistle = current_stim.iloc[-1]
                
                T0 = first_whistle.t_i #Stimulus start
                f = last_whistle.freq #Stimulus pitch
                d = last_whistle.duration #Stimulus duration
                
                stimulus_info.append({'bird': bird, 'phase': experiment, 'trial': num_trial,
                                      'stim_id': s, 'f0': f, 'd0': d, 'gap': gap_stim,
                                      'interval': d + gap_stim, 'T0': T0})
#Data frame containing the stimulus info
stim_info = pd.DataFrame(stimulus_info)
#Store stim_info
stim_info.to_excel('stim_info.xlsx', index=False)
#------------------------------------------------------------------------------
#CONTROL DATA (from previous season)
#------------------------------------------------------------------------------
all_birds= pd.read_pickle(r"Control prev season/all_birds.pkl")
#Store control data
control = all_birds[all_birds.phase != 'playback']
control.to_excel('control_previous_season.xlsx', index=False)

#song analysis for the control from previous season
whistle_songs_control = [] # Initialize an empty list to store the data
bird_control_id = control.bird.unique()
for bird in bird_control_id:
    current_bird = control[control.bird == bird]
    phase_idx = current_bird.phase.unique()
    for phase_id in phase_idx:
        if phase_id == 'playback':
            continue
        current_phase = current_bird[current_bird.phase == phase_id]
        set_idx = current_phase.set.unique()
        for set_id in set_idx:
            current_set = current_phase[current_phase.set == set_id]
            snippet_id = current_set.snippet_idx.unique()
            for snippet in snippet_id:
                current_song = current_set[current_set.snippet_idx == snippet]
                range_duration = current_song['duration'].max() - current_song['duration'].min()
                average_duration = current_song['duration'].mean()
                average_freq = current_song['pitch_whistles'].mean()
                duration_last_whistle = current_song['duration'].iloc[-1]
                duration_first_whistle = current_song['duration'].iloc[0]
                median_duration = current_song['duration'].median()
                median_freq = current_song['pitch_whistles'].median()
                freq_last_whistle = current_song['pitch_whistles'].iloc[-1]
                freq_first_whistle = current_song['pitch_whistles'].iloc[0]
                d_first_to_last  = duration_first_whistle - duration_last_whistle
                f_first_to_last = freq_first_whistle - freq_last_whistle
                whistle_songs_control.append({'bird': bird, 'song': snippet, 'last_d': duration_last_whistle, 
                                      'last_f': freq_last_whistle, 'f_average': average_freq, 'd_average': average_duration,
                                      'd_median': median_duration, 'f_median': median_freq,
                                      'first_d': duration_first_whistle, 'first_f': freq_first_whistle, 'd_range': range_duration,
                                      'd_first_to_last': d_first_to_last, 'f_first_to_last': f_first_to_last})  
whistle_songs_control = pd.DataFrame(whistle_songs_control)                
#Store birds data info
whistle_songs_control.to_excel('whistle_songs_control.xlsx', index=False)
#------------------------------------
#Select the cutt off values for the trimodal distribution
#------------------------------------
min1, min2 = 0.14, 0.31
fast_data = control[control.duration <= min1]
med_data = control[(control.duration > min1) & (control.duration <= min2)]
slow_data = control[control.duration > min2]

#Compute the extreme values of the distributions
min_fast, max_fast = np.min(fast_data.duration), np.max(fast_data.duration)
min_med, max_med = np.min(med_data.duration), np.max(med_data.duration)
min_slow, max_slow = np.min(slow_data.duration), np.max(slow_data.duration)

#Compute the 5th and the 95th percentile for each distribution
perc5_fast, perc95_fast = np.percentile(fast_data.duration, 5), np.percentile(fast_data.duration, 95)
perc5_med, perc95_med = np.percentile(med_data.duration, 5), np.percentile(med_data.duration, 95)
perc5_slow, perc95_slow = np.percentile(slow_data.duration, 5), np.percentile(slow_data.duration, 95)

#Choose the values for the whistle duration as the boundaries of the shaded regions and their middle point
duration_playbacks = [min_fast,(min_fast + perc5_fast)/2, perc5_fast,
                 perc95_fast, min1 , perc5_med,
                 perc95_med, min2 , perc5_slow,
                 perc95_slow, (perc95_slow + max_slow)/2 , max_slow]
#------------------------------------------------------------------------------
#BIRDS RESPONSES
#-------------------------------------------------------------------------------
# Load the Excel file with multiple sheets into a data frame
excel_data_birds = pd.read_excel(r'Birds responses/Birds_responses.xlsx', sheet_name = None)
birds_data = pd.concat(excel_data_birds.values(), ignore_index=True)

#Compute the start of each song and the gap between its whistles
output_df = pd.DataFrame(columns = ['gap', 'n_whistles', 'temp_distance', 'freq_distance'])
bird_id = birds_data.bird.unique()
for bird in bird_id:
    current_bird = birds_data[birds_data.bird == bird]
    exp_id = current_bird.phase.unique()
    for experiment in exp_id:
        current_experiment = current_bird[current_bird.phase == experiment]
        trial_id = current_experiment.trial.unique()
        for num_trial in trial_id:
            current_trial = current_experiment[current_experiment.trial == num_trial]
            song_id = current_trial.song.unique()
            for s in song_id:
                new_df = pd.DataFrame(columns = ['gap', 'n_whistles', 'temp_distance', 'freq_distance'])
                current_song = current_trial[current_trial.song == s]
                #Compute the silent gaps between whistles s_i = start_i+1 - end_i
                temp = current_song.t_i - current_song.t_f.shift(1)
                gap = temp.shift(-1)
                new_df['gap'] = gap
                n_whistles = len(current_song.duration)
                new_df['n_whistles'] = n_whistles
                #Compute temporal and spectral distance for sequential whistles  
                if n_whistles >= 2:
                    temp_distance = current_song.duration - current_song.duration.shift(1)
                    freq_distance = current_song.freq_mean - current_song.freq_mean.shift(1)                           
                    new_df['temp_distance'] = temp_distance.shift(-1)
                    new_df['freq_distance'] = freq_distance.shift(-1)                                                              
                output_df = pd.concat([output_df.astype(new_df.dtypes), new_df.astype(output_df.dtypes)])
birds_data['gap'] = output_df['gap']
birds_data['n_whistles'] = output_df['n_whistles']
birds_data['temp_distance'], birds_data['freq_distance'] = output_df['temp_distance'], output_df['freq_distance']
birds_data['interval'] = birds_data.duration + birds_data.gap 

#Check for stimuli responses
birds_data['stim_id'] = -1.0 #ID of the stimulus they responded to
birds_data['f0'] = -1.0 #Freq of the playback in case of the whistle being a response
birds_data['T0'] = -1.0 #Start of the stimuli
birds_data['d0'] = -1.0 #duration of the whistle playback
birds_data['df'] = -1.0 #Diff between the stimuli f and the response f
birds_data['dT'] = -1.0 #Diff between the stimuli whistle duration and the response duration
birds_data['category'] = 'none'
bird_id = birds_data.bird.unique()

for bird in bird_id:
    current_bird = birds_data[birds_data.bird == bird]
    exp_id = current_bird.phase.unique()
    for experiment in exp_id:
        current_experiment= current_bird[current_bird.phase == experiment]
        trial_id = current_experiment.trial.unique()
        for num_trial in trial_id:
            current_trial = current_experiment[current_experiment.trial == num_trial]
            #Get the stimulus info
            current_stim_info = stim_info[(stim_info.bird == bird) & 
                                          (stim_info.phase == experiment) &
                                          (stim_info.trial == num_trial)]
            song_id = current_trial.song.unique()
            for s in song_id:
                current_song = current_trial[current_trial.song == s]
                t0 = current_song.t_i.iloc[0] #Start of the whistles
                response = current_song.stim_response.iloc[0]
                if response == '+':
                    #Get the largest T_stim s.t T_stim < t0
                    T_stim = current_stim_info[current_stim_info['T0'] < t0]['T0'].max()
                    stim_data = current_stim_info[current_stim_info['T0'] == T_stim]
                    f0 = stim_data['f0'].values[0]
                    d0 = stim_data['d0'].values[0]
                    for index, syllable in current_song.iterrows():
                        birds_data.at[index, 'stim_id'] = stim_data['stim_id'].values[0]
                        birds_data.at[index, 'f0'] = f0
                        birds_data.at[index, 'd0'] = d0
                        birds_data.at[index, 'T0'] = T_stim
                        birds_data.at[index, 'df'] = birds_data.at[index, 'freq_mean'] - f0 
                        birds_data.at[index, 'dT'] = birds_data.at[index, 'duration'] - d0
                        if (d0 > 0) & (d0 <= 0.1):
                            birds_data.at[index, 'category'] = 'R1'
                        elif (d0 > 0.1) & (d0 <= 0.2):
                            birds_data.at[index, 'category'] = 'R2'
                        elif (d0 > 0.2) & (d0 <= 0.4):
                            birds_data.at[index, 'category'] = 'R3'
                        elif (d0 > 0.4):
                            birds_data.at[index, 'category'] = 'R4'
#Store birds data info
birds_data.to_excel('birds_data.xlsx', index=False)

#Get a general analysis per song
birds_songs_info = [] # Initialize an empty list to store the data

for bird in bird_id:
    current_bird = birds_data[birds_data.bird == bird]
    exp_id = current_bird.phase.unique()
    for experiment in exp_id:
        current_experiment = current_bird[current_bird.phase == experiment]
        trial_id = current_experiment.trial.unique()
        for num_trial in trial_id:
            current_trial = current_experiment[current_experiment.trial == num_trial]                
            song_id = current_trial.song.unique()
            for s in song_id:
                current_song = current_trial[current_trial.song == s]
                
                t_first = current_song['t_i'].iloc[0] 
                stim_response = current_song['stim_response'].iloc[0]
                stim_id = current_song['stim_id'].iloc[0]
                f0 = current_song['f0'].iloc[0]
                d0 = current_song['d0'].iloc[0]
                category = 'none'
                if (d0 > 0) & (d0 <= 0.1):
                    category = 'R1'
                elif (d0 > 0.1) & (d0 <= 0.2):
                    category = 'R2'
                elif (d0 > 0.2) & (d0 <= 0.4):
                    category = 'R3'
                elif (d0 > 0.4):
                    category = 'R4'
                n_whistles = current_song['n_whistles'].iloc[0]
                
                best_dT = np.min(np.abs(current_song['dT']))
                average_duration = current_song['duration'].mean()
                median_duration = current_song['duration'].median()
                median_freq = current_song['freq_mean'].median()
                average_freq = current_song['freq_mean'].mean()
                duration_last_whistle = current_song['duration'].iloc[-1]
                duration_first_whistle = current_song['duration'].iloc[0]
                if n_whistles >= 2:
                    duration_last_gap = current_song['gap'].dropna().iloc[-1]
                    duration_last_interval = current_song['interval'].dropna().iloc[-1]
                
                    duration_first_gap = current_song['gap'].iloc[0]
                    duration_first_interval = current_song['interval'].iloc[0]
                
                    gap_first_to_last = duration_first_gap - duration_last_gap
                    interval_first_to_last = duration_first_interval - duration_last_interval
                freq_last_whistle = current_song['freq_mean'].iloc[-1]
                freq_first_whistle = current_song['freq_mean'].iloc[0]
                range_duration = current_song['duration'].max() - current_song['duration'].min()
                d_first_to_last  = duration_first_whistle - duration_last_whistle
                f_first_to_last = freq_first_whistle - freq_last_whistle
                time_response = -1
                if stim_response == '+':
                    time_response = t_first - current_song['T0'].iloc[0]    
                # Find the index of the row where dT is small
                best_dT_idx = np.argmin(np.abs(current_song['dT']))
                # Get the corresponding duration for best_dT
                d_best = current_song['duration'].iloc[best_dT_idx]
                position_song = best_dT_idx/n_whistles
                birds_songs_info.append({'bird': bird, 'phase': experiment, 'trial': num_trial, 'd_best': d_best, 'd_median': median_duration,
                                      'last_d': duration_last_whistle, 'last_f': freq_last_whistle, 't_first': t_first,     
                                      'song': s, 'f_average': average_freq, 'f_median': median_freq, 'd_average': average_duration, 'time_response': time_response,
                                      'first_d': duration_first_whistle, 'first_f': freq_first_whistle, 'n_whistles': n_whistles,
                                      'd_first_to_last': d_first_to_last, 'f_first_to_last': f_first_to_last, 'd_range': range_duration,
                                      'gap_first': duration_first_gap, 'gap_last': duration_last_gap, 'gap_first_to_last': gap_first_to_last,
                                      'interval_first': duration_first_interval, 'interval_last': duration_last_interval, 'interval_first_to_last': interval_first_to_last,
                                      'stim_response': stim_response, 'stim_id': stim_id, 'f0': f0, 'd0': d0, 'category': category, 'best_dT': best_dT,
                                      'ix_best': best_dT_idx, 'loc_best': position_song})   
birds_songs_info = pd.DataFrame(birds_songs_info)                
#Store birds data info
birds_songs_info.to_excel('birds_songs_info.xlsx', index=False)
#------------------------------------------------------------------------------------------------------------------------------
#Data shuffling----------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
def shuffle_data(data):
    # Extract relevant columns
    responses_info = data.filter(['bird', 'song', 'f0', 'd0', 'd_median', 'f_median', 'category'], axis=1)

    # Perform bootstrap sampling (with replacement) on `d_median` and `f_median`
    bootstrap_indices = np.random.choice(responses_info.index, size=len(responses_info), replace=True)
    responses_info[['d_median', 'f_median']] = responses_info.loc[bootstrap_indices, ['d_median', 'f_median']].values

    return responses_info
#------------------------------------------------------------------------------------------------------------------------------
#Filter the birds
#------------------------------------------------------------------------------------------------------------------------------
bird_filter_1 = (birds_songs_info.bird == 'pepsi') | (birds_songs_info.bird == 'clipper') | (birds_songs_info.bird ==
              'georgew') | (birds_songs_info.bird == 'kalimero') |   (birds_songs_info.bird ==
              'leo') | (birds_songs_info.bird == 'mate') | (birds_songs_info.bird == 'chonksMcGee')

bird_filter_2 = (birds_songs_info.bird == 'pepsi') | (birds_songs_info.bird == 'clipper') | (birds_songs_info.bird ==
              'georgew') | (birds_songs_info.bird == 'hope') | (birds_songs_info.bird ==
              'grape') | (birds_songs_info.bird ==
              'chonksMcGee') | (birds_songs_info.bird ==            
              'gregory')| (birds_songs_info.bird == 'matchbox')  
#------------------------------------------------------------------------------------------------------------------------------
#DATA FOR THE PLOTS -----------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------ 
positive_responses_phase_1 = (birds_songs_info.phase == 1) & (birds_songs_info.stim_response == '+') 
positive_responses_phase_2 = (birds_songs_info.phase == 2) & (birds_songs_info.stim_response == '+')

filtered_data_1 = birds_songs_info[positive_responses_phase_1 & bird_filter_1] 
filtered_data_2 = birds_songs_info[positive_responses_phase_2 & bird_filter_2] 
#Example of bootstraped data
shuffled_responses_1 = shuffle_data(filtered_data_1)
shuffled_responses_2 = shuffle_data(filtered_data_2)

# Define categories R1, R2, R3, and R4 for EXP 1
CATEGORY = ['R1', 'R2', 'R3', 'R4']
# Colors corresponding to regions in EXP 1
COLORS = ["#E78652", "#DFA71F", "#C69C6D", "#7E4F25"]  # Updated colors for each region

# Playback times by region in EXP1
f_exp1 = 2.4 #KHz
playback_times_exp1 = {
    'R1': [0.03972789115646259, 0.06284580498866213, 0.08596371882086168],
    'R2': [0.13479591836734695, 0.14, 0.14883446712018142],
    'R3': [0.2917619047619047, 0.31, 0.32537414965986394],
    'R4': [0.5814002267573696, 0.7211649659863946, 0.8609297052154194]
}
# Playback times and durations by region in EXP2 
playback_exp2_d_A= [0.1, 0.1, 0.1]
playback_exp2_f_A= [6, 7, 8]
playback_exp2_d_B= [0.6, 0.7, 0.8]
playback_exp2_f_B= [8, 7, 6]
playback_exp2_d_C= [0.6, 0.7, 0.8]
playback_exp2_f_C= [1, 2, 3]

# Dictionary for playback data
playback_data_exp2 = {
    "A": (playback_exp2_d_A, playback_exp2_f_A),
    "B": (playback_exp2_d_B, playback_exp2_f_B),
    "C": (playback_exp2_d_C, playback_exp2_f_C),
}

zone_A = birds_songs_info.d0 < 0.2
zone_A_positive = (positive_responses_phase_2) & (zone_A)

zone_B = (birds_songs_info.f0  > 5000) & (birds_songs_info.d0 >= 0.5)
zone_B_positive = (positive_responses_phase_2) & (zone_B)

zone_C = birds_songs_info.f0 < 4000
zone_C_positive = (positive_responses_phase_2) & (zone_C)

zones = {
    "A": zone_A_positive,
    "B": zone_B_positive,
    "C": zone_C_positive,
}

# Define the positive phase 2 filter
positive_responses_phase_2_shuffled = shuffled_responses_2

# Define regions and their conditions
zone_A_shuffled = shuffled_responses_2.d0 < 0.2
zone_B_shuffled = (shuffled_responses_2.f0 > 5000) & (shuffled_responses_2.d0 >= 0.5)
zone_C_shuffled = shuffled_responses_2.f0 < 4000

# Define regions and their corresponding data and titles
regions_shuffled = {
    "A": shuffled_responses_2[zone_A_shuffled],
    "B": shuffled_responses_2[zone_B_shuffled],
    "C": shuffled_responses_2[zone_C_shuffled]}
#------------------------------------------------------------------------------------------------------------------------------                                
#FIG 1-------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------  
def plot_responses(data_set, parameter, bw_value):
    # Create a figure with 1x4 subplots, sharing x and y axes
    fig, ax = plt.subplots(1, 4, figsize=(3.5, 3), sharey=True, sharex='col')  
    median_values = []

    for idx, category in enumerate(CATEGORY):
        # Filter data for each category (region)
        data = data_set[data_set.category == category]
        kde_r = gaussian_kde(data[parameter], bw_method=bw_value)
        kde_control = gaussian_kde(whistle_songs_control[parameter], bw_method=0.5)

        ratios = kde_r(data[parameter]) / (kde_control(data[parameter]) + 0.0001)

        # Create a color map based on KDE ratios
        cmap = plt.cm.RdGy_r  # Red-White-Blue colormap
        norm = plt.Normalize(vmin=0, vmax=2)  # Normalize ratios to [0, 2]

        # Horizontal jitter for scatter plot
        jitter = np.random.uniform(-0.1, 0.1, size=data[parameter].shape)  # Random jitter

        # First row: Jitter scatter plot with KDE ratio-based coloring
        ax[idx].scatter(jitter, data[parameter], edgecolor='none', c=ratios, cmap=cmap, norm=norm, s=50)
        ax[idx].set_xlim(-0.2, 0.2)
        ax[idx].set_xticks([])
        ax[idx].set_title(f'{category}')

        # Add horizontal lines for playback events
        for playback in playback_times_exp1[category]:
            ax[idx].hlines(y=playback, xmin=0.1, xmax=-0.1, color='gray', linestyle='-', linewidth=2)

        # Find the median
        median_value = np.median(np.sort(data[parameter]))
        median_values.append(median_value)

        # Plot median as a horizontal line
        ax[idx].hlines(y=median_value, xmin=0.1, xmax=-0.1, color='k', linestyle='-', linewidth=3)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.4)
    plt.tight_layout()

    # Save and show plot
    plt.savefig('Plots/Results_exp_1_normalized.pdf', transparent=True)
    plt.show()

    return median_values
#--------------------------------------
#FIG 1B
#--------------------------------------
plt.style.use('default')
fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(5,8), sharex=True, 
                         gridspec_kw={'height_ratios': [2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5]})

# Make KDE plots share the y-axis
for i in range(0, 10, 2): 
    axes[i].sharey(axes[4])

# --------------------------------------
# Control density (unchanged)
# --------------------------------------
kde = sns.kdeplot(whistle_songs_control.d_median, color='black', linestyle='-', lw=2, 
                   bw_adjust=0.4, common_norm=True, ax=axes[0])
x_vals = kde.lines[0].get_xdata()  # Extract x values from the KDE plot
dx = np.mean(np.diff(x_vals))
print(dx)
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()
axes[0].fill_between(x_values, y_values, color='gray', alpha=0.2)
axes[0].set_xlim([0, 0.9])

# Scatter Jittered Data for Control 
control_data = whistle_songs_control.d_median
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(control_data))
axes[1].scatter(control_data, np.zeros_like(control_data) + y_jittered, 
                color='black', alpha=0.5, s=2)
# Compute and plot median
median_value = np.median(control_data)
axes[1].vlines(median_value, -0.08, -0.2, color='black', lw=2, linestyle='-', zorder=6)

for region, color, times in zip(playback_times_exp1.keys(), COLORS, playback_times_exp1.values()):
    for time in times:
        axes[1].vlines(time, -0.08, -0.2, color=color, lw=2, zorder=5)
axes[1].set_yticks([])
axes[1].set_ylabel("Control")

# --------------------------------------
# FIG 1C: Loop through experimental categories
# --------------------------------------
kde_control = gaussian_kde(whistle_songs_control.d_median, bw_method=0.5)
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=0, vmax=2)

for idx, (category, color) in enumerate(zip(CATEGORY, COLORS)):
    data = filtered_data_1[filtered_data_1["category"] == category]
    kde = sns.kdeplot(data.d_median, color='black', linestyle='-', lw=2, 
                       bw_adjust=0.4, ax=axes[2 * (idx + 1)])
    
    x_values = kde.lines[0].get_xdata()
    y_values = kde.lines[0].get_ydata()
    axes[2 * (idx + 1)].fill_between(x_values, y_values, color=color)
    
    # Compute KDE ratio
    kde_r = gaussian_kde(data.d_median, bw_method=0.5)
    ratios = np.clip(kde_r(data.d_median) / (kde_control(data.d_median)), 0, 2)
    
    # Scatter Jittered Data with KDE ratio coloring
    y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(data.d_median))
    scatter = axes[2 * (idx + 1) + 1].scatter(
        data.d_median, np.zeros_like(data.d_median) + y_jittered, 
        c=ratios, cmap=cmap, norm=norm, edgecolor='none', s=10
    )
    
    if category in playback_times_exp1:
        for time in playback_times_exp1[category]:
            axes[2 * (idx + 1) + 1].vlines(time, -0.08, -0.2, color=color, lw=2, zorder=5)
    
    # Compute and plot median
    median_value = np.median(data.d_median)
    axes[2 * (idx + 1) + 1].vlines(median_value, -0.08, -0.2, color='black', lw=2, linestyle='-', zorder=6)
    
    axes[2 * (idx + 1) + 1].set_yticks([])
    axes[2 * (idx + 1) + 1].set_ylabel(category)

# Add colorbar for KDE ratio visualization
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(scatter, cax=cbar_ax)
cbar.set_label("KDE Ratio")

plt.tight_layout(rect=[0, 0, 0.9, 1])
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Results exp1.pdf', transparent=True)
plt.show()
#--------------------------------------
#Compute the real medians
#--------------------------------------
plt.style.use('default')
median_real = plot_responses(filtered_data_1, 'd_median', 0.6)
#--------------------------------------
#FIG 1F
#--------------------------------------
plt.style.use('default')

# Define a custom colormap from white to yellowgreen
custom_cmap = LinearSegmentedColormap.from_list("white_to_olivedrab", ["white", "olivedrab"])

fig = plt.figure(figsize=(5, 5))  # Making figure square

# Scatter plot for playback points (marked with red '+')
plt.plot(playback_exp2_d_A, playback_exp2_f_A, 
         color="#12B568", marker='+', linestyle='None', 
         markersize=10, markeredgewidth=4)
plt.plot(playback_exp2_d_B, playback_exp2_f_B, 
         color="#12B568", marker='+', linestyle='None', 
         markersize=10, markeredgewidth=4)
plt.plot(playback_exp2_d_C, playback_exp2_f_C, 
         color="#12B568", marker='+', linestyle='None', 
         markersize=10, markeredgewidth=4)

# KDE plot using the custom colormap
kde = sns.kdeplot(x=whistle_songs_control.d_median, 
                   y=whistle_songs_control.f_median / 1000, 
                   cmap=custom_cmap, fill=True, 
                   bw_adjust=0.7, levels=20, thresh=0.1, zorder=-1)

# Outer contour with a reversed red colormap
outside = sns.kdeplot(x=whistle_songs_control.d_median, 
                      y=whistle_songs_control.f_median / 1000, 
                      cmap="Reds_r", fill=False,
                      bw_adjust=0.7, levels=1, thresh=0.1)

# Scatter plot for whistle songs control (small black points)
plt.scatter(whistle_songs_control.d_median, 
            whistle_songs_control.f_median / 1000, 
            c='k', s=2, zorder=1)

# Labels and axis limits
plt.xlabel('Median of whistle durations per song (s)')
plt.ylabel('Median frequency (kHz)')
plt.xlim([0, 0.9])
plt.ylim([0, 9])

# Square aspect ratio
plt.gca().set_box_aspect(1)

plt.tight_layout()

# Set font type for PDF export
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/control_meanvspitch.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG 1G and H
#--------------------------------------
# Define global bins
global_x_bins = np.linspace(0, 1, 100)
global_y_bins = np.linspace(0, 10000, 100)
medians_per_region = {}

plt.style.use('default')

# Create figure and axes (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

# Define regions and corresponding filters
regions = {
    'A': (birds_songs_info[zone_A_positive & bird_filter_2], playback_exp2_d_A, playback_exp2_f_A),
    'B': (birds_songs_info[zone_B_positive & bird_filter_2], playback_exp2_d_B, playback_exp2_f_B),
    'C': (birds_songs_info[zone_C_positive & bird_filter_2], playback_exp2_d_C, playback_exp2_f_C)
}

# Define color map for KDE ratio
vmin, vmax = 0, 2
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Iterate over regions to create subplots
for i, (region, (data, playback_d, playback_f)) in enumerate(regions.items()):
    
    # First Row: Scatter and KDE
    ax_kde = axes[0, i]
    
    # Scatter plot of playback points
    ax_kde.plot(playback_d, [x * 1000 for x in playback_f], 
            color="#12B568", marker='+', linestyle='None', 
            markersize=6, markeredgewidth=3)
    
    # KDE plots
    sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_kde)
    sns.kdeplot(x=data.d_median, y=data.f_median, cmap=custom_cmap, fill=True,
                bw_adjust=0.7, levels=10, thresh=0.1, ax=ax_kde, zorder=-1)
    ax_kde.scatter(data.d_median, data.f_median, c='k', s=4, zorder=1)
    
    ax_kde.set_title(f"Region {region}")
    ax_kde.set_xlim([0, 0.9])
    ax_kde.set_ylim([0, 9000])
    ax_kde.set_box_aspect(1)  # Ensure the subplot is square

    # Second Row: KDE Ratio
    ax_ratio = axes[1, i]
    
    # Compute KDEs
    x_exp, y_exp = data.d_median, data.f_median
    kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3)
    
    x_ctrl, y_ctrl = whistle_songs_control.d_median, whistle_songs_control.f_median
    valid_mask = np.isfinite(x_ctrl) & np.isfinite(y_ctrl)
    kde_ctrl = gaussian_kde(np.vstack([x_ctrl[valid_mask], y_ctrl[valid_mask]]))

    # Generate KDE grids
    x_grid = np.linspace(0, 0.9, 100)
    y_grid = np.linspace(0, 9000, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Compute KDE values
    Z_exp = kde_exp(positions).reshape(X.shape)
    Z_ctrl = kde_ctrl(positions).reshape(X.shape)
    
    # Compute KDE ratio and ensure values stay within defined range
    ratio = np.clip(Z_exp / (Z_ctrl + 0.0003), vmin, vmax)

    # Contour plot for ratio
    ax_ratio.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)
    
    # Overlay control KDE
    sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median, 
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_ratio)
    
    # Scatter plot of playback points
    ax_ratio.plot(playback_d, [x * 1000 for x in playback_f], 
              color="#12B568", marker='+', linestyle='None', 
              markersize=6, markeredgewidth=3)

    ax_ratio.set_xlim([0, 0.9])
    ax_ratio.set_ylim([0, 9000])
    ax_ratio.set_box_aspect(1)  # Ensure the subplot is square

# Adjust figure layout and add colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

# Remove individual axis labels
for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

# Set common axis labels
fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)  # X-axis label
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)  # Y-axis label

# Use ScalarMappable for global colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array since it's only used for color mapping
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')

mpl.rcParams['pdf.fonttype'] = 42

# Save and display the plot
plt.savefig('Plots/Exp_2_per_region_and_ratio.pdf', transparent=True)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------                              
#FIG S1-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------
#Analysis for the BIC value
#---------------------------------------------------------------------
control_clean = control[['duration']].dropna().reset_index(drop=True)
X = control_clean.values  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the best number of clusters using BIC
bic_scores = []
n_components_range = range(1, 10)  # Trying 1 to 10 clusters
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))

# Choose the best k (min BIC)
best_k = n_components_range[np.argmin(bic_scores)]
gmm = GaussianMixture(n_components=best_k, random_state=42)
control_clean['cluster'] = gmm.fit_predict(X_scaled)

# Print the duration range for each cluster
print(f"Optimal number of clusters for the whistle duration: {best_k}\n")
for cluster in range(best_k):
    cluster_data = control_clean[control_clean.cluster == cluster]['duration']
    print(f"Cluster {cluster}: min = {cluster_data.min():.3f} (s), max = {cluster_data.max():.3f} (s)")
#---------------------------------------------------------------------    
#Analysis for the Silhouette value
#---------------------------------------------------------------------
control_clean = control[['duration', 'pitch_whistles']].dropna().reset_index(drop=True)
X = control_clean.values  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
silhouette_scores = []
k_range = range(2, 10)  # Silhouette isn't valid for k=1
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
# Choose best k based on silhouette score
best_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters for the soundspace based on silhouette score: {best_k}")

# Apply K-means clustering with best k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
control_clean['cluster'] = kmeans.fit_predict(X_scaled)

plt.style.use('default')

# Create a figure and primary axis (BIC Score - Left)
fig, ax1 = plt.subplots(figsize=(5, 5))
# Plot BIC Score (Green) on the left y-axis
ax1.plot(range(1, 10), bic_scores, color='#008000', marker='o', linestyle='--', label='Duration')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('BIC', color='#008000')
ax1.tick_params(axis='y', labelcolor='#008000')
ax1.set_title('Cluster Evaluation Metrics')
ax1.set_box_aspect(1)

# Create a second y-axis (right side) for Silhouette Score
ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_scores, color='#800080', marker='o', linestyle='-', label='Duration and frequency')
ax2.set_ylabel('Silhouette Score', color='#800080')
ax2.tick_params(axis='y', labelcolor='#800080')
ax2.set_box_aspect(1)

if 3 in k_range:
    idx = k_range.index(3)  # Find index of k=3
    ax1.scatter(3, bic_scores[idx + 1], edgecolor='#008000', facecolor='#008000', marker='s', s=100)
    ax2.scatter(3, silhouette_scores[idx], edgecolor='#800080', facecolor='#800080', marker='s', s=100)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(
    handles1 + handles2, 
    labels1 + labels2, 
    loc='upper right',  
    frameon=False       
)

fig.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig("Plots/Control_Clusters_Comparison.pdf", transparent=True)
plt.show()

#--------------------------------------
#FIG S1A
#--------------------------------------
plt.style.use('default')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 1]})
# KDE plot (First Subplot)
ax1 = axes[0]
kde = sns.kdeplot(control.duration, color='black', linestyle='-', lw=2, bw_adjust=0.4, ax=ax1)
# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()  # Get the latest plotted KDE line

# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)

# Print the first spacing (since it's usually uniform)
print("Distance between consecutive x-values:", x_spacing[0])
ax1.set_ylabel('Normalized counts per bin', fontsize=12)
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()

# Define region boundaries and colors
region_bounds = [0, 0.15, 0.3, 0.9]  # Adjust as needed
region_colors = ['limegreen', 'yellowgreen', 'darkgreen']

# Fill each region without overlap
for i in range(len(region_bounds) - 1):
    mask = np.logical_and(x_values > region_bounds[i], x_values < region_bounds[i + 1])
    ax1.fill_between(x_values[mask], y_values[mask], color=region_colors[i], alpha=0.3)
ax1.set_xlim([0, 0.9])
# Scatter plot with jittered data (Second Subplot)
ax2 = axes[1]
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(control.duration))
ax2.scatter(control.duration, np.zeros_like(control.duration) + y_jittered, 
                color='black', alpha=0.2, s=2)

for region, color, times in zip(playback_times_exp1.keys(), COLORS, playback_times_exp1.values()):
    for time in times:
        ax2.vlines(time, -0.08, -0.1, color=color, lw=2, zorder=5)
ax2.set_yticks([])
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/control_all_whistle_durations.pdf", transparent=True)
plt.show()
#--------------------------------------
#FIG S1B
#--------------------------------------
# Define birds and filter data for examples
bird_names = ["Quentin", "Coco", "Isaac"]
bird_data = [control[control.bird == bird] for bird in bird_names]

plt.style.use('default')
# Create subplots (2 rows per bird: 1 for KDE, 1 for scatter)
fig, axes = plt.subplots(len(bird_names) * 2, 1, figsize=(4, 8), 
                         sharex=True, gridspec_kw={'height_ratios': [3, 1] * len(bird_names)})
# Make KDE plots share the y-axis
for i in range(2, 5, 2):  # Indices of KDE plots (excluding the first one)
    axes[i].sharey(axes[0])
for i, (bird, data) in enumerate(zip(bird_names, bird_data)):
    ax_kde = axes[i * 2]       # Top row: KDE plot
    ax_scatter = axes[i * 2 + 1]  # Bottom row: Scatter plot

    # --- KDE Plot ---
    kde = sns.kdeplot(data.duration, color='black', linestyle='-', lw=2, bw_adjust=0.4, ax=ax_kde)
    x_values = ax_kde.lines[-1].get_xdata()  # Get the latest plotted KDE line

    # Compute the spacing between consecutive x-values
    x_spacing = np.diff(x_values)

    # Print the first spacing (since it's usually uniform)
    print("Distance between consecutive x-values:", x_spacing[0])
    ax1.set_ylabel('Normalized counts per bin', fontsize=12)
    x_values = kde.lines[0].get_xdata()
    y_values = kde.lines[0].get_ydata()
    ax_kde.fill_between(x_values, y_values, color='gray', alpha=0.2)
    
    ax_kde.set_title(bird, fontsize=12)
    ax_kde.set_xlim([0, 0.9])

    # --- Scatter Plot with Jitter ---
    jitter = np.random.uniform(low=-0.075, high=0.075, size=len(data.duration))  # Small vertical jitter
    ax_scatter.scatter(data.duration, np.zeros_like(data.duration) + jitter, color='black', alpha=0.2, s=2)

    # Remove y-axis labels for scatter plots (cleaner look)
    ax_scatter.set_yticks([])
    ax_scatter.set_ylabel("")
    
    # Label only the last scatter plot
    if i == len(bird_names) - 1:
        ax_scatter.set_xlabel("Whistle duration (s)", fontsize=12)

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/Control_whistle_durations_three birds.pdf", transparent=True)
plt.show()
#--------------------------------------
#FIG S1C
#--------------------------------------
# Define three distinct purple tones
purple_shades = ['#6a0dad', '#9b5de5', '#d8b4f8']  # Dark purple, medium purple, light purple
cluster_colors = purple_shades[:best_k]

# Plot scatter with three shades of purple
plt.figure(figsize=(6, 5))
for cluster, color in zip(range(best_k), cluster_colors):
    subset = control_clean[control_clean.cluster == cluster]
    plt.scatter(subset['duration'], subset['pitch_whistles']/1000, s=2, color=color)
outside = sns.kdeplot(x=whistle_songs_control.d_median, 
                   y=whistle_songs_control.f_median/1000, 
                   cmap="Reds_r", fill=False,
                   bw_adjust=0.7, levels=1, thresh=0.1)
plt.xlim([0,0.9])
plt.ylim([0,9])
plt.xlabel('Whistles duration (s)')
plt.ylabel('Frequency (kHz)')
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/Control_durationvspitch_duration.pdf", transparent=True)
plt.show()
#--------------------------------------
#FIG S1D
#--------------------------------------
plt.style.use('default')
fig, axes = plt.subplots(3, 1, figsize=(4, 10), sharex=True, sharey=True)
for i, (ax, bird, data) in enumerate(zip(axes, bird_names, bird_data)):
    ax.scatter(data.duration, data.pitch_whistles/1000, c = 'black', s = 10, alpha = 0.5)
    outside = sns.kdeplot(x=whistle_songs_control.d_median, 
                       y=whistle_songs_control.f_average/1000, 
                       cmap="Reds_r", fill=False,
                       bw_adjust=0.7, levels=1, thresh=0.1, ax = ax)
    ax.set_title(bird)
    ax.set_xlim([0,1])
    ax.set_ylim([0,9])
    ax.set_xlabel("Whistle duration (s)")
    ax.set_ylabel("Whistle frequency (kHz)")
# Ensure proper layout
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/Control_whistle_durations_pitch__three_birds.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S1E
#--------------------------------------
plt.style.use('default')

# Compute mean and standard deviation
mean_val = np.mean(whistle_songs_control.d_first_to_last)
std_val = np.std(whistle_songs_control.d_first_to_last, ddof=0)  # Population standard deviation

# Format values with two significant digits
mean_str = f"{mean_val:.2g}"
std_str = f"{std_val:.2g}"

# Define a square figure
fig, ax = plt.subplots(figsize=(5, 5))  # Square figure

# KDE plot
kde = sns.kdeplot(whistle_songs_control.d_first_to_last, color='black', linestyle='-', lw=3, bw_adjust=0.5, ax=ax)
x_values = ax.lines[-1].get_xdata()  # Get the latest plotted KDE line

# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)

# Print the first spacing (since it's usually uniform)
print("Distance between consecutive x-values:", x_spacing[0])
ax1.set_ylabel('Normalized counts per bin', fontsize=12)
# Get KDE data for shading
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()
ax.fill_between(x_values, y_values, color='gray', alpha=0.2)

# Add vertical line at x = 0
ax.axvline(x=0, c='tomato', linestyle='--', lw=3)

ax.set_ylabel('normalized frequency')
ax.set_xlabel('first to last (s) within one song')

# Ensure axes have the same length
ax.set_box_aspect(1)  # Maintains equal length for x and y axes

# Add legend
ax.legend([f"Mean: {mean_str}, SD: {std_str}"], loc="upper left", frameon=False)

# Adjust layout
fig.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/control_distribution_d_first-last.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S1F
#--------------------------------------
plt.style.use('default')

# Compute mean and standard deviation
mean_val = np.mean(control.temp_distance)
std_val = np.std(control.temp_distance, ddof=0)  # Population standard deviation

# Format values with two significant digits
mean_str = f"{mean_val:.2g}"
std_str = f"{std_val:.2g}"

# Define a square figure
fig, ax = plt.subplots(figsize=(5, 5))  # Make figure square

# KDE plot
kde = sns.kdeplot(control.temp_distance, color='k', linestyle='-', lw=3, bw_adjust=0.5, ax=ax)
x_values = ax.lines[-1].get_xdata()  # Get the latest plotted KDE line

# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)

# Print the first spacing (since it's usually uniform)
print("Distance between consecutive x-values:", x_spacing[0])
# Get KDE data for shading
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()
ax.fill_between(x_values, y_values, color='gray', alpha=0.2)

# Add vertical line at x = 0
ax.axvline(x=0, c='tomato', linestyle='--', lw=3)

ax.set_ylabel('normalized frequency')
ax.set_xlabel('distance subsequent')

# Ensure axes have the same length
ax.set_box_aspect(1)  # Maintains equal length for x and y axes

# Add legend
ax.legend([f"Mean: {mean_str}, SD: {std_str}"], loc="upper right", frameon=False)

# Adjust layout
fig.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/control_distribution_d_distance.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S1H
#--------------------------------------
plt.style.use('default')

fig, ax = plt.subplots(figsize=(5, 5))  # Square figure
# KDE plot
kde = sns.kdeplot(control.gap, color='black', linestyle='-', lw=2, bw_adjust=0.5, ax=ax)
# Get KDE data for shading
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()
ax.fill_between(x_values, y_values, color='gray', alpha=0.2)
ax.set_xlabel('Gap duration (s)')
ax.set_ylabel('Normalized counts per bin')
ax.set_xlim([0,0.7])
# Ensure axes have the same length
ax.set_box_aspect(1)  # Keeps x and y axis physically equal in length

# Adjust layout
fig.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42

fig.savefig('Plots/Control_gap.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S1K
#--------------------------------------
df = control[['duration', 'gap']].dropna()
# Compute linear regression
slope, intercept, r_value, _, _ = linregress(df.duration, df.gap)
r_squared = r_value**2

# Generate fitted line
x_vals = np.linspace(df.duration.min(), df.duration.max(), 100)
y_vals = slope * x_vals + intercept

plt.style.use('default')

fig, ax = plt.subplots(figsize=(5, 5))  # Square figure
# Scatter plot
ax.scatter(df.duration, df.gap, color='k', s=3, alpha=0.5)
ax.set_ylim([0,0.6])
ax.set_xlim([0,0.9])
# Regression line
ax.plot(x_vals, y_vals, color='r', linewidth=2, label=f'(R²={r_squared:.2f})')
ax.set_xlabel('Whistle duration (s)')
ax.set_ylabel('Gap duration (s)')
# Ensure axes have the same length
ax.set_box_aspect(1)  # Keeps x and y axis physically equal in length
ax.legend(frameon=False)
fig.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Control_gapvsduration.pdf', transparent=True)
plt.show()
#------------------------------------------------------------------------------------------------------------------------------                                
#FIG S2-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------
#FIG S2A
#--------------------------------------
def compute_medians(data):
    median_values = {category: np.nan for category in CATEGORY}  # Ensure all regions are included
    for category in CATEGORY:
        region_data = data[data["category"] == category]["d_median"].dropna()  # Drop NaNs
        if not region_data.empty:
            median_values[category] = np.median(region_data)
    return median_values

# Function to plot the results
def plot_results(data, playback_times, save_path):
    birds = data["bird"].unique()
    num_birds = len(birds)
    
    fig, axes = plt.subplots(4, num_birds, figsize=(3 * num_birds, 4), sharex=True, sharey=True)

    for col, bird in enumerate(birds):
        bird_data = data[data["bird"] == bird]
        median_values = compute_medians(bird_data)
        
        for row, category in enumerate(CATEGORY):
            ax = axes[row, col] if num_birds > 1 else axes[row]
            region_data = bird_data[bird_data["category"] == category]

            # Scatter plot with jitter
            y_jittered = np.random.uniform(low =-0.2, high=0.2, size=len(region_data))
            x_values = region_data["d_average"].dropna()
            ax.scatter(x_values, y_jittered, color=COLORS[row], s=10)
            # Remove y-axis values
            ax.set_yticks([])

            # Playback times
            for x in playback_times.get(category, []):
                ax.vlines(x, ymin=-0.65, ymax=-0.35, colors='gray', linewidth=2)

            # Median values
            if not np.isnan(median_values[category]):
                ax.vlines(median_values[category], ymin=-0.65, ymax=-0.35, 
                          colors='black', linewidth=3, linestyle='-')

            if col == 0:
                ax.set_ylabel(region)
            if row == 3:
                ax.set_xlabel("Median whistle duration (s)")
            if row == 0:
                ax.set_title(bird)

    plt.tight_layout()
    mpl.rcParams['pdf.fonttype'] = 42
    plt.savefig(save_path, transparent=True)
    plt.show()
# Ensure styles are applied
plt.style.use('default')
# Call the function
plot_results(filtered_data_1, playback_times_exp1, 'Plots/Results_exp1_all_birds.pdf')    
#--------------------------------------
#FIG S2B
#--------------------------------------
shuffled_responses_1 = shuffle_data(filtered_data_1)
plt.style.use('default')
median_random = plot_responses(shuffled_responses_1, 'd_median', 0.6)

plt.style.use('default')
fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(5,8), sharex=True, 
                         gridspec_kw={'height_ratios': [2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5]})

# Make KDE plots share the y-axis
for i in range(0, 10, 2): 
    axes[i].sharey(axes[4])

# --------------------------------------
# Control density (unchanged)
# --------------------------------------
kde = sns.kdeplot(whistle_songs_control.d_median, color='black', linestyle='-', lw=2, 
                   bw_adjust=0.4, common_norm=True, ax=axes[0])
x_vals = kde.lines[0].get_xdata()  # Extract x values from the KDE plot
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()
axes[0].fill_between(x_values, y_values, color='gray', alpha=0.2)
axes[0].set_xlim([0, 0.9])

# Scatter Jittered Data for Control 
control_data = whistle_songs_control.d_median
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(control_data))
axes[1].scatter(control_data, np.zeros_like(control_data) + y_jittered, 
                color='black', alpha=0.5, s=2)
# Compute and plot median
median_value = np.median(control_data)
axes[1].vlines(median_value, -0.08, -0.2, color='black', lw=2, linestyle='-', zorder=6)

for region, color, times in zip(playback_times_exp1.keys(), COLORS, playback_times_exp1.values()):
    for time in times:
        axes[1].vlines(time, -0.08, -0.2, color=color, lw=2, zorder=5)
axes[1].set_yticks([])
axes[1].set_ylabel("Control")

kde_control = gaussian_kde(whistle_songs_control.d_median, bw_method=0.5)
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=0, vmax=2)

for idx, (category, color) in enumerate(zip(CATEGORY, COLORS)):
    data = shuffled_responses_1[shuffled_responses_1["category"] == category]
    kde = sns.kdeplot(data.d_median, color='black', linestyle='-', lw=2, 
                       bw_adjust=0.4, ax=axes[2 * (idx + 1)])
    
    x_values = kde.lines[0].get_xdata()
    y_values = kde.lines[0].get_ydata()
    axes[2 * (idx + 1)].fill_between(x_values, y_values, color=color)
    
    # Compute KDE ratio
    kde_r = gaussian_kde(data.d_median, bw_method=0.5)
    ratios = np.clip(kde_r(data.d_median) / (kde_control(data.d_median)), 0, 2)
    
    # Scatter Jittered Data with KDE ratio coloring
    y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(data.d_median))
    scatter = axes[2 * (idx + 1) + 1].scatter(
        data.d_median, np.zeros_like(data.d_median) + y_jittered, 
        c=ratios, cmap=cmap, norm=norm, edgecolor='none', s=10
    )
    
    if category in playback_times_exp1:
        for time in playback_times_exp1[category]:
            axes[2 * (idx + 1) + 1].vlines(time, -0.08, -0.2, color=color, lw=2, zorder=5)
    
    # Compute and plot median
    median_value = np.median(data.d_median)
    axes[2 * (idx + 1) + 1].vlines(median_value, -0.08, -0.2, color='black', lw=2, linestyle='-', zorder=6)
    
    axes[2 * (idx + 1) + 1].set_yticks([])
    axes[2 * (idx + 1) + 1].set_ylabel(category)

# Add colorbar for KDE ratio visualization
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(scatter, cax=cbar_ax)
cbar.set_label("KDE Ratio")

plt.tight_layout(rect=[0, 0, 0.9, 1])
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Shuffled results exp1.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S2C
#--------------------------------------
num_shufflings = 10000
def plot_random_shufflings(num_iterations, parameter):
    median_null = {category: [] for category in CATEGORY}
    for iteration in tqdm(range(num_iterations), desc="Shuffling Progress"):
        shuffled_data = shuffle_data(filtered_data_1) 
        for category in CATEGORY:
            data = shuffled_data[shuffled_data.category == category]
            median_value = np.median(np.sort(data[parameter]))  # Directly get median
            median_null[category].append(median_value)
    return median_null

median_null = plot_random_shufflings(num_shufflings, 'd_median')

# Create figure
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(4, 4), sharex=True, sharey=True)
axes = axes.flatten()

for idx, category in enumerate(CATEGORY):
    color = COLORS[idx]  
    null_distribution = np.array(median_null[category])  # Convert to array
    real_value = median_real[idx]

    if category == "R1":  # Left-tailed test
        p_value = np.mean(null_distribution <= real_value)
        print(f"One-tailed p-value (left) for {category}: {p_value:.10f}")

    elif category == "R4":  # Right-tailed test
        p_value = np.mean(null_distribution >= real_value)
        print(f"One-tailed p-value (right) for {category}: {p_value:.10f}")

    elif category in ["R2", "R3"]:  # Two-tailed test
        p_value = 2 * min(
            np.mean(null_distribution >= real_value),
            np.mean(null_distribution <= real_value)
        )
        print(f"Two-tailed p-value for {category}: {p_value:.10f}")

    # Plot histogram
    axes[idx].hist(null_distribution, bins=30, edgecolor=color, facecolor=color)
    axes[idx].axvline(real_value, color='k', linestyle='-', linewidth=1.5)
    axes[idx].set_title(f'{category}\n(p={p_value:.10f})')
    axes[idx].set_xlabel('Median (s)')
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_xlim([0.1, 0.4])

# Show plot
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/statistics_exp1.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S2D
#--------------------------------------
# Extract unique birds from filtered_data_2
birds = filtered_data_2.bird.unique()  

plt.style.use('default')

# Create subplots: Regions as rows, birds as columns
fig, axes = plt.subplots(3, len(birds), figsize=(2 * len(birds), 6), sharex=True, sharey=True)
axes = np.atleast_2d(axes)  # Ensure axes is always 2D

regions = {
    "Region A": (zone_A_positive, "A"),
    "Region B": (zone_B_positive, "B"),
    "Region C": (zone_C_positive, "C"),
}

# Iterate through regions and birds
for j, (title, (zone_mask, key)) in enumerate(regions.items()):
    for i, bird in enumerate(birds):
        ax = axes[j, i]  # Regions as rows, birds as columns

        # Filter data
        data = filtered_data_2[zone_mask & (filtered_data_2.bird == bird)]

        # Scatter plots
        ax.scatter(data.d_median, data.f_median / 1000, color='k', s = 20, zorder=1)
        ax.plot(*playback_data_exp2[key], 
        color="#12B568", marker='+', linestyle='None', 
        markersize=7, markeredgewidth=3)

        # Formatting
        ax.set_xlabel("Median duration")
        ax.set_ylabel("Median pitch")
        ax.set_xlim([0, 0.9])
        ax.set_ylim([0, 9])
        if j == 0:
            ax.set_title(bird)  # Only set bird names in top row

# Adjust layout
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/Exp2_per_region_with_medians_all_birds.pdf', transparent=True)
plt.show()
#------------------------------------
#FIG S2E
#------------------------------------
# Define the positive phase 2 filter
shuffled_responses_2 = shuffle_data(filtered_data_2)

# Define regions and their conditions
zone_A_shuffled = shuffled_responses_2.d0 < 0.2
zone_B_shuffled = (shuffled_responses_2.f0 > 5000) & (shuffled_responses_2.d0 >= 0.5)
zone_C_shuffled = shuffled_responses_2.f0 < 4000


plt.style.use('default')

# Create figure and axes (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

# Define regions and corresponding filters
regions_shuffled = {
    'A': (shuffled_responses_2[zone_A_shuffled], playback_exp2_d_A, playback_exp2_f_A),
    'B': (shuffled_responses_2[zone_B_shuffled], playback_exp2_d_B, playback_exp2_f_B),
    'C': (shuffled_responses_2[zone_C_shuffled], playback_exp2_d_C, playback_exp2_f_C)
}

# Define color map for KDE ratio
vmin, vmax = 0, 2
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Iterate over regions to create subplots
for i, (region, (data, playback_d, playback_f)) in enumerate(regions_shuffled.items()):
    
    # First Row: Scatter and KDE
    ax_kde = axes[0, i]
    
    # Scatter plot of playback points
    ax_kde.plot(playback_d, [x * 1000 for x in playback_f], 
            color="#12B568", marker='+', linestyle='None', 
            markersize=10, markeredgewidth=4)
    
    # KDE plots
    sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median,
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_kde)
    sns.kdeplot(x=data.d_median, y=data.f_median, cmap=custom_cmap, fill=True,
                bw_adjust=0.7, levels=10, thresh=0.1, ax=ax_kde, zorder=-1)
    ax_kde.scatter(data.d_median, data.f_median, c='k', s=4, zorder=1)
    
    ax_kde.set_title(f"Region {region}")
    ax_kde.set_xlim([0, 0.9])
    ax_kde.set_ylim([0, 9000])
    ax_kde.set_box_aspect(1)  # Ensure the subplot is square

    # Second Row: KDE Ratio
    ax_ratio = axes[1, i]
    
    # Compute KDEs
    x_exp, y_exp = data.d_median, data.f_median
    kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3)
    
    x_ctrl, y_ctrl = whistle_songs_control.d_median, whistle_songs_control.f_median
    valid_mask = np.isfinite(x_ctrl) & np.isfinite(y_ctrl)
    kde_ctrl = gaussian_kde(np.vstack([x_ctrl[valid_mask], y_ctrl[valid_mask]]))

    # Generate KDE grids
    x_grid = np.linspace(0, 0.9, 100)
    y_grid = np.linspace(0, 9000, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Compute KDE values
    Z_exp = kde_exp(positions).reshape(X.shape)
    Z_ctrl = kde_ctrl(positions).reshape(X.shape)
    
    # Compute KDE ratio and ensure values stay within defined range
    ratio = np.clip(Z_exp / (Z_ctrl + 0.0003), vmin, vmax)

    # Contour plot for ratio
    ax_ratio.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)
    
    # Overlay control KDE
    sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median, 
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax_ratio)
    
    # Scatter plot of playback points
    ax_ratio.plot(playback_d, [x * 1000 for x in playback_f], 
              color="#12B568", marker='+', linestyle='None', 
              markersize=10, markeredgewidth=4)

    ax_ratio.set_xlim([0, 0.9])
    ax_ratio.set_ylim([0, 9000])
    ax_ratio.set_box_aspect(1)  # Ensure the subplot is square

# Adjust figure layout and add colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

# Remove individual axis labels
for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

# Set common axis labels
fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)  # X-axis label
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)  # Y-axis label

# Use ScalarMappable for global colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array since it's only used for color mapping
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')

mpl.rcParams['pdf.fonttype'] = 42

# Save and display the plot
plt.savefig('Plots/Shuffled_Exp_2_per_region_and_ratio.pdf', transparent=True)
plt.show()

#--------------------------------------
#FIG S2F
#--------------------------------------
# Standardize the data BEFORE adding the cluster column
scaler = StandardScaler()
control_scaled = scaler.fit_transform(control_clean[['duration', 'pitch_whistles']])  # Only two columns

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
control_clean['cluster'] = kmeans.fit_predict(control_scaled)  # Now add cluster labels

# Get cluster centers in original scale
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Create meshgrid for decision boundaries
x_min, x_max = control_clean['duration'].min() - 0.1, control_clean['duration'].max() + 0.1
y_min, y_max = control_clean['pitch_whistles'].min() - 0.1, control_clean['pitch_whistles'].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predict cluster labels for each point in the meshgrid
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points)  # Now should work fine
Z = kmeans.predict(mesh_points_scaled)
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Purples_r')  # Decision regions
for i in range(3):
    subset = control_clean[control_clean['cluster'] == i]
    plt.scatter(subset['duration'], subset['pitch_whistles'], label=f'C{i + 1}', color = purple_shades[i])
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='k', marker='x', s=100, label='Centers')
plt.xlabel('Duration')
plt.ylabel('Pitch Whistles')
plt.legend(frameon = False)
plt.title('K-Means Clustering with Decision Boundaries')
plt.show()

def classify_and_label(df, scaler, kmeans):

    if not {'d_median', 'f_median'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'd_median' and 'f_median' columns.")

    # Copy and rename columns to match those used in training
    new_data = df.rename(columns={'d_median': 'duration', 'f_median': 'pitch_whistles'}).copy()

    # Standardize the renamed data
    new_data_scaled = scaler.transform(new_data[['duration', 'pitch_whistles']])

    # Predict clusters
    new_data['cluster'] = kmeans.predict(new_data_scaled)

    # Map cluster numbers to labels
    cluster_to_label = {0: 'C1', 1: 'C2', 2: 'C3'}
    new_data['cluster_region'] = new_data['cluster'].map(cluster_to_label)

    # Drop the temporary cluster column
    new_data = new_data.drop(columns=['cluster'])

    return new_data

data_expA = filtered_data_2[zone_A_positive]
data_expB = filtered_data_2[zone_B_positive]
data_expC = filtered_data_2[zone_C_positive]

# Classify each dataset
data_expA_labeled = classify_and_label(data_expA, scaler, kmeans)
data_expB_labeled = classify_and_label(data_expB, scaler, kmeans)
data_expC_labeled = classify_and_label(data_expC, scaler, kmeans)

# Use the exact real counts you provided
real_counts_A = data_expA_labeled['cluster_region'].value_counts() 
real_counts_B = data_expB_labeled['cluster_region'].value_counts()
real_counts_C = data_expC_labeled['cluster_region'].value_counts() 

# Print results
print("Counts for data_expA:\n", real_counts_A )
print("\nCounts for data_expB:\n", real_counts_B )
print("\nCounts for data_expC:\n", real_counts_C )

def simulate_cluster_counts(df, scaler, kmeans, shuffle_func, N=100):
    results = {'A': [], 'B': [], 'C': []}

    for _ in tqdm(range(N)):
        # Shuffle data
        shuffled_responses_2 = shuffle_data(df)

        # Define regions and their conditions
        zone_A_shuffled = shuffled_responses_2.d0 < 0.2
        zone_B_shuffled = (shuffled_responses_2.f0 > 5000) & (shuffled_responses_2.d0 >= 0.5)
        zone_C_shuffled = shuffled_responses_2.f0 < 4000

        # Classify each dataset
        data_expA_labeled = classify_and_label(shuffled_responses_2[zone_A_shuffled], scaler, kmeans)
        data_expB_labeled = classify_and_label(shuffled_responses_2[zone_B_shuffled], scaler, kmeans)
        data_expC_labeled = classify_and_label(shuffled_responses_2[zone_C_shuffled], scaler, kmeans)

        # Count occurrences of each cluster region
        count_A = data_expA_labeled['cluster_region'].value_counts()
        count_B = data_expB_labeled['cluster_region'].value_counts()
        count_C = data_expC_labeled['cluster_region'].value_counts()

        # Append counts to results (if a category is missing, assume count = 0)
        results['A'].append(count_A)
        results['B'].append(count_B)
        results['C'].append(count_C)
    return results

# Run the simulation
simulated_counts = simulate_cluster_counts(filtered_data_2, scaler, kmeans, shuffle_data, N=10000)

# Convert to NumPy arrays
counts_A = np.array(simulated_counts['A'])
counts_B = np.array(simulated_counts['B'])
counts_C = np.array(simulated_counts['C'])

# Define settings
regions = ['A', 'B', 'C']
cluster_labels = ['C1', 'C2', 'C3']

data_list = [counts_A, counts_B, counts_C]
real_counts = [real_counts_A, real_counts_B, real_counts_C]
# Create figure with square subplots (clusters as rows, regions as columns)
fig, axes = plt.subplots(len(cluster_labels), len(regions), figsize=(7, 7), sharex=True, sharey=True)
# Loop through regions and clusters (swapped order for correct layout)
for i, region in tqdm(enumerate(regions)):  # Columns (regions)
    # Compute total counts per iteration for the whole region
    total_counts_per_iteration = np.sum(data_list[i], axis=1)  # Sum across clusters

    for j, (cluster, color) in enumerate(zip(cluster_labels, purple_shades)):  # Rows (clusters)
        ax = axes[j, i]  # Swap indexing: clusters as rows, regions as columns
        
        # Compute percentages: (cluster count / total count per iteration) * 100
        cluster_percentages = (data_list[i][:, j] / total_counts_per_iteration) * 100
        
        # Plot histogram with percentages
        ax.hist(cluster_percentages, bins=20, color=color, edgecolor=color)

        # Compute p-value if the cluster exists in real_count
        p_value = None
        if cluster in real_counts[i]:
            observed_value = (real_counts[i][cluster] / np.sum(real_counts[i])) * 100  # Convert to percentage

            ax.axvline(observed_value, color='black', linestyle='-', linewidth=2)  
            
            # Compute two-sided p-value
            more_extreme = np.sum(cluster_percentages >= observed_value)  # One-tailed (right)
            p_value = (2 * min(more_extreme, len(cluster_percentages) - more_extreme)) / len(cluster_percentages)  # Two-tailed

        # Set title with p-value
        p_text = f" (p = {p_value:.3f})" if p_value is not None else ""
        ax.set_title(f"C{cluster} - R{region}{p_text}", fontsize=10)

        # Ensure square subplots while allowing independent axis ranges
        ax.set_box_aspect(1)  # Keeps each subplot physically square

        ax.set_xlabel("Percentage of counts")
        ax.set_ylabel("Frequency")

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Exp2 p-values.pdf', transparent=True)
plt.show()