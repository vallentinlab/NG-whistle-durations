#------------------------------------------------------------------------------
#IMPORT PACKAGES
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import linregress
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
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
stimulus_info = [] 

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
                    continue 
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
#BIRDS DATA (from previous season)
#------------------------------------------------------------------------------
all_birds= pd.read_pickle(r"Control prev season/all_birds.pkl")
#------------------------------------------------------------------------------
#PITCH DATA (from previous season)
#------------------------------------------------------------------------------
pitch_pb = all_birds[all_birds.phase == 'playback']
pitch_pb.to_excel('pitch_previous_season.xlsx', index=False)

#song analysis for the pitch experiment from previous season
whistle_songs_pitch = [] 
bird_pitch_id = pitch_pb.bird.unique()
for bird in bird_pitch_id:
    current_bird = pitch_pb[pitch_pb.bird == bird]
    phase_idx = current_bird.phase.unique()
    for phase_id in phase_idx:
        if phase_id != 'playback':
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
                n_syl = len(current_song['duration'])
                d0 = 0.440 #duration of the whistle pb
                f0 = current_song['stimulus_freq'].iloc[0]
                if n_syl > 1:
                    last_int = current_song['interval'].iloc[-2]
                    first_int = current_song['interval'].iloc[0]
                    interval_first_to_last  = first_int - last_int
                    
                    last_gap = current_song['gap'].iloc[-2]
                    first_gap = current_song['gap'].iloc[0]
                    gap_first_to_last  = first_gap - last_gap 
                else:
                    last_int = np.nan
                    first_int = np.nan
                    interval_first_to_last  = np.nan
                    
                    last_gap = np.nan
                    first_gap = np.nan
                    gap_first_to_last  = np.nan   
                median_gap = current_song['gap'].median()
                median_int = current_song['interval'].median()
                whistle_songs_pitch.append({'bird': bird, 'song': snippet, 'last_d': duration_last_whistle,
                                      'last_gap': last_gap, 'last_int': last_int, 'first_gap': first_gap, 'first_int': first_int,
                                      'last_f': freq_last_whistle, 'f_average': average_freq, 'd_average': average_duration,
                                      'd_median': median_duration, 'f_median': median_freq, 'n_syl': n_syl, 'gap_median': median_gap, 'int_median': median_int,
                                      'first_d': duration_first_whistle, 'first_f': freq_first_whistle, 'd_range': range_duration,
                                      'int_first_to_last': interval_first_to_last, 'gap_first_to_last': gap_first_to_last,
                                      'd_first_to_last': d_first_to_last, 'f_first_to_last': f_first_to_last, 'd0': d0, 'f0': f0})  
whistle_songs_pitch = pd.DataFrame(whistle_songs_pitch)                
#Store birds data info
whistle_songs_pitch.to_excel('whistle_songs_pitch.xlsx', index=False)
#------------------------------------------------------------------------------
#CONTROL DATA 
#------------------------------------------------------------------------------
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
                n_syl = len(current_song['duration']) 
                if n_syl > 1:
                    last_int = current_song['interval'].iloc[-2]
                    first_int = current_song['interval'].iloc[0]
                    interval_first_to_last  = first_int - last_int
                    
                    last_gap = current_song['gap'].iloc[-2]
                    first_gap = current_song['gap'].iloc[0]
                    gap_first_to_last  = first_gap - last_gap 
                else:
                    last_int = np.nan
                    first_int = np.nan
                    interval_first_to_last  = np.nan
                    
                    last_gap = np.nan
                    first_gap = np.nan
                    gap_first_to_last  = np.nan   
                median_gap = current_song['gap'].median()
                median_int = current_song['interval'].median()
                whistle_songs_control.append({'bird': bird, 'song': snippet, 'last_d': duration_last_whistle,
                                      'last_gap': last_gap, 'last_int': last_int, 'first_gap': first_gap, 'first_int': first_int,
                                      'last_f': freq_last_whistle, 'f_average': average_freq, 'd_average': average_duration,
                                      'd_median': median_duration, 'f_median': median_freq, 'n_syl': n_syl, 'gap_median': median_gap, 'int_median': median_int,
                                      'first_d': duration_first_whistle, 'first_f': freq_first_whistle, 'd_range': range_duration,
                                      'int_first_to_last': interval_first_to_last, 'gap_first_to_last': gap_first_to_last,
                                      'd_first_to_last': d_first_to_last, 'f_first_to_last': f_first_to_last})  
whistle_songs_control = pd.DataFrame(whistle_songs_control)                
#Store birds data info
whistle_songs_control.to_excel('whistle_songs_control.xlsx', index=False)

print('Mean: ', np.mean(whistle_songs_control.d_median), " STD: ", np.std(whistle_songs_control.d_median))
print('Median: ', np.median(whistle_songs_control.d_median), " MAD: ", np.median(np.abs(whistle_songs_control.d_median - np.median(whistle_songs_control.d_median))))
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
                ix_median = (np.abs(current_song['duration'] - median_duration)).argmin()
                loc_median = ix_median / n_whistles
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
                median_int = current_song['interval'].median()
                median_gap = current_song['gap'].median()
                # Find the index of the row where dT is small
                best_dT_idx = np.argmin(np.abs(current_song['dT']))
                # Get the corresponding duration for best_dT
                d_best = current_song['duration'].iloc[best_dT_idx]
                position_song = best_dT_idx/n_whistles
                birds_songs_info.append({'bird': bird, 'phase': experiment, 'trial': num_trial, 'd_best': d_best, 'd_median': median_duration,
                                      'last_d': duration_last_whistle, 'last_f': freq_last_whistle, 't_first': t_first,'int_median': median_int,     
                                      'song': s, 'f_average': average_freq, 'f_median': median_freq, 'd_average': average_duration, 'time_response': time_response,
                                      'first_d': duration_first_whistle, 'first_f': freq_first_whistle, 'n_syl': n_whistles, 'gap_median': median_gap,
                                      'd_first_to_last': d_first_to_last, 'f_first_to_last': f_first_to_last, 'd_range': range_duration,
                                      'gap_first': duration_first_gap, 'gap_last': duration_last_gap, 'gap_first_to_last': gap_first_to_last,
                                      'interval_first': duration_first_interval, 'interval_last': duration_last_interval, 'interval_first_to_last': interval_first_to_last,
                                      'stim_response': stim_response, 'stim_id': stim_id, 'f0': f0, 'd0': d0, 'category': category, 'best_dT': best_dT,
                                      'ix_median':ix_median, 'loc_median': loc_median, 'ix_best': best_dT_idx, 'loc_best': position_song})   
birds_songs_info = pd.DataFrame(birds_songs_info)                
#Store birds data info
birds_songs_info.to_excel('birds_songs_info.xlsx', index=False)
#------------------------------------------------------------------------------------------------------------------------------
#Data shuffling----------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
def shuffle_data(data):
    # Extract relevant columns
    responses_info = data.filter(['bird', 'song', 'f0', 'd0', 'd_median', 'int_median', 'gap_median', 'f_median', 'category'], axis=1)

    # Perform bootstrap sampling 
    bootstrap_indices = np.random.choice(responses_info.index, size=len(responses_info), replace=True)
    responses_info[['d_median', 'f_median', 'int_median', 'gap_median']] = responses_info.loc[bootstrap_indices, ['d_median', 'f_median', 'int_median', 'gap_median']].values

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
#Consider the data from the selected birds and only those who were responses to our playbacks
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
COLORS = ["#E78652", "#DFA71F", "#C69C6D", "#7E4F25"]  

# Playback times by region in EXP1
f_exp1 = 2.4 #KHz
playback_times_exp1 = {
    'R1': [0.03972789115646259, 0.06284580498866213, 0.08596371882086168],
    'R2': [0.13479591836734695, 0.14, 0.14883446712018142],
    'R3': [0.2917619047619047, 0.31, 0.32537414965986394],
    'R4': [0.5814002267573696, 0.7211649659863946, 0.8609297052154194]
}
# Playback times and durations by region in EXP2 
playback_exp2_d_A= [0.14, 0.14, 0.14]
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

def _clear_axis_collections(ax):
    for coll in list(ax.collections):
        try:
            coll.remove()
        except Exception:
            pass

plt.style.use('default')
fig, axes = plt.subplots(
    nrows=10, ncols=1, figsize=(5, 9), sharex=True,
    gridspec_kw={'height_ratios': [2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5, 2, 1.5]}
)

for i in range(0, 10, 2):
    axes[i].sharey(axes[4])
ymax_list = []

# --------------------------------------
# Control KDE 
# --------------------------------------
kde = sns.kdeplot(
    whistle_songs_control.d_median,
    color='black', linestyle='-', lw=2, bw_adjust=0.4,
    ax=axes[0]
)

# Extract the drawn KDE curve (density)
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()


dx_ref = float(np.mean(np.diff(x_values)))
print("Δx (bin width) =", dx_ref)


y_prob = y_values * dx_ref

kde.lines[0].set_ydata(y_prob)
_clear_axis_collections(axes[0])
axes[0].fill_between(x_values, y_prob, color='gray', alpha=0.2, zorder=1)
axes[0].set_xlim([0, 0.9])
axes[0].set_ylabel("Probability")
axes[0].relim()
axes[0].autoscale_view()
ymax_list.append(np.nanmax(y_prob))

# --------------------------------------
# Scatter Jittered Data for Control 
# --------------------------------------
control_data = whistle_songs_control.d_median
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(control_data))
axes[1].scatter(
    control_data, np.zeros_like(control_data) + y_jittered,
    color='black', alpha=0.5, s=2
)

# Control median and playback markers
median_value = np.median(control_data)
axes[1].vlines(median_value, -0.08, -0.2, color='black', lw=2, linestyle='-', zorder=6)

for region, color, times in zip(playback_times_exp1.keys(), COLORS, playback_times_exp1.values()):
    for time in times:
        axes[1].vlines(time, -0.08, -0.2, color=color, lw=2, zorder=5)

axes[1].set_yticks([])
axes[1].set_ylabel("Control")

# --------------------------------------
# FIG 1D: Experimental categories
# --------------------------------------
# KDE for ratios 
kde_control = gaussian_kde(whistle_songs_control.d_median, bw_method=0.5)
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=0, vmax=2)

scatter = None  
for idx, (category, color) in enumerate(zip(CATEGORY, COLORS)):
    data = filtered_data_1[filtered_data_1["category"] == category]
    print(category, "N =", len(data.d_median))

    # KDE line 
    kde_ax = axes[2 * (idx + 1)]
    kde_line = sns.kdeplot(
        data.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.4, ax=kde_ax
    )

    x_vals_exp = kde_line.lines[0].get_xdata()
    y_vals_exp = kde_line.lines[0].get_ydata()
    y_prob_exp = y_vals_exp * dx_ref  

    kde_line.lines[0].set_ydata(y_prob_exp)
    _clear_axis_collections(kde_ax)
    kde_ax.fill_between(x_vals_exp, y_prob_exp, color=color, zorder=1)
    kde_ax.set_ylabel("Probability")
    kde_ax.relim()
    kde_ax.autoscale_view()

    ymax_list.append(np.nanmax(y_prob_exp))

    # KDE ratio for scatter coloring 
    kde_r = gaussian_kde(data.d_median, bw_method=0.5)
    ratios = np.clip(kde_r(data.d_median) / (kde_control(data.d_median)), 0, 2)

    # Scatter with jitter, colored by KDE ratio
    y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(data.d_median))
    scatter = axes[2 * (idx + 1) + 1].scatter(
        data.d_median, np.zeros_like(data.d_median) + y_jittered,
        c=ratios, cmap=cmap, norm=norm, edgecolor='none', s=10
    )

    # Playback vertical lines for this category
    if category in playback_times_exp1:
        for time in playback_times_exp1[category]:
            axes[2 * (idx + 1) + 1].vlines(time, -0.08, -0.2, color=color, lw=2, zorder=5)

    # Median line for this category
    median_value = np.median(data.d_median)
    print(category, ' Mean: ', np.mean(data.d_median), " STD: ", np.std(data.d_median))
    print(category, ' Median: ', np.median(data.d_median), " MAD: ",
          np.median(np.abs(data.d_median - np.median(data.d_median))))

    axes[2 * (idx + 1) + 1].vlines(median_value, -0.08, -0.2, color='black', lw=2, linestyle='-', zorder=6)
    axes[2 * (idx + 1) + 1].set_yticks([])
    axes[2 * (idx + 1) + 1].set_ylabel(category)

ymax = float(max(ymax_list)) * 1.05  
for ax in axes[::2]:
    ax.set_ylim(0, ymax)
# --------------------------------------
# Colorbar for KDE ratio 
# --------------------------------------
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
#Statistical analysis for the insets in Fig 1D
#--------------------------------------
num_shufflings = 10000
def plot_random_shufflings(num_iterations, parameter):
    median_null = {category: [] for category in CATEGORY}
    for iteration in tqdm(range(num_iterations), desc="Shuffling Progress"):
        shuffled_data = shuffle_data(filtered_data_1) 
        for category in CATEGORY:
            data = shuffled_data[shuffled_data.category == category]
            median_value = np.median(np.sort(data[parameter]))
            median_null[category].append(median_value)
    return median_null

median_null = plot_random_shufflings(num_shufflings, 'd_median')

# Create figure
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(4, 4), sharex=True, sharey=True)
axes = axes.flatten()

for idx, category in enumerate(CATEGORY):
    color = COLORS[idx]  
    null_distribution = np.array(median_null[category])  
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

#------------------------------------------------------------------------------------------------------------------------------                                
#FIG 3-------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------  
#--------------------------------------
#FIG 3A
#--------------------------------------
# -------------------------
# 1. Prepare data 
# -------------------------
control_clean = control[['duration', 'pitch_whistles']].dropna().reset_index(drop=True)
control_clean['pitch_kHz'] = control_clean['pitch_whistles'] / 1000
X = control_clean[['duration', 'pitch_kHz']].values

# -------------------------
# 2. Clustering
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k
silhouette_scores = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {best_k}")

# Fit final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
control_clean['cluster'] = kmeans.fit_predict(X_scaled)

# -------------------------
# 3. Plotting
# -------------------------
plt.style.use('default')
custom_cmap = LinearSegmentedColormap.from_list("white_to_olivedrab", ["white", "olivedrab"])
fig = plt.figure(figsize=(5, 5))

# Playback points
plt.plot(playback_exp2_d_A, playback_exp2_f_A, color="#12B568", marker='+', linestyle='None', markersize=10, markeredgewidth=4)
plt.plot(playback_exp2_d_B, playback_exp2_f_B, color="#12B568", marker='+', linestyle='None', markersize=10, markeredgewidth=4)
plt.plot(playback_exp2_d_C, playback_exp2_f_C, color="#12B568", marker='+', linestyle='None', markersize=10, markeredgewidth=4)

# KDE
kde = sns.kdeplot(
    x=whistle_songs_control.d_median, 
    y=whistle_songs_control.f_median / 1000, 
    cmap=custom_cmap, fill=True, 
    bw_adjust=0.7, levels=20, thresh=0.1, zorder=-1
)

# Outer contour
outside = sns.kdeplot(
    x=whistle_songs_control.d_median, 
    y=whistle_songs_control.f_median / 1000, 
    cmap="Reds_r", fill=False,
    bw_adjust=0.7, levels=1, thresh=0.1
)

# Scatter of control points
plt.scatter(
    whistle_songs_control.d_median, 
    whistle_songs_control.f_median / 1000, 
    c='k', s=2, zorder=1
)

# -------------------------
# 4. Decision boundaries
# -------------------------
# Create meshgrid
x_min, x_max = 0, 0.9
y_min, y_max = 0, 9
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Scale grid points
grid_points_scaled = scaler.transform(grid_points)
Z = kmeans.predict(grid_points_scaled)
Z = Z.reshape(xx.shape)

# Plot boundaries 
plt.contour(xx, yy, Z, levels=np.arange(best_k + 1) - 0.5, colors='gray', linestyles='dashed', linewidths=1)

# Labels, limits
plt.xlabel('Median of whistle durations per song (s)')
plt.ylabel('Median frequency (kHz)')
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.gca().set_box_aspect(1)

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/control_meanvspitch_with_clusters.pdf', transparent=True)
plt.show()

#--------------------------------------
#FIG 3E
#--------------------------------------
# Define global bins
global_x_bins = np.linspace(0, 1, 100)
global_y_bins = np.linspace(0, 10000, 100)
medians_per_region = {}

plt.style.use('default')

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
    print(region, len(data.d_median))
    ax_kde.set_title(f"Region {region}")
    ax_kde.set_xlim([0, 0.9])
    ax_kde.set_ylim([0, 9000])
    ax_kde.set_box_aspect(1)  

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
fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12)  
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12) 

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')

mpl.rcParams['pdf.fonttype'] = 42

# Save and display the plot
plt.savefig('Plots/Exp_2_per_region_and_ratio.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG 3F
#--------------------------------------
plt.style.use('default')

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

regions = {
    'A': (birds_songs_info[zone_A_positive & bird_filter_2], playback_exp2_d_A, playback_exp2_f_A),
    'B': (birds_songs_info[zone_B_positive & bird_filter_2], playback_exp2_d_B, playback_exp2_f_B),
    'C': (birds_songs_info[zone_C_positive & bird_filter_2], playback_exp2_d_C, playback_exp2_f_C)
}

duration_match_percent = {}
pitch_match_percent = {}

for i, (region, (data, playback_d, playback_f)) in enumerate(regions.items()):
    ax = axes[i]

    # Compute error
    d_err = data.d_median - data.d0
    f_err = (data.f_median - data.f0)/1000

    # Plot all in black
    ax.scatter(d_err, f_err, c='k', s=20, zorder=1)

    # Threshold lines
    ax.axvline(x=0.0, color='green', linestyle='--', linewidth=2)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2)

    ax.set_title(f"Region {region}")
    ax.set_box_aspect(1)


fig.text(0.5, 0.04, "Error Duration (s)", ha='center', fontsize=12)
fig.text(0.06, 0.5, "Error Pitch (kHz)", va='center', rotation='vertical', fontsize=12)

mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/Exp_2_errors.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG 3G
#--------------------------------------
regions_masks = {
    'A': zone_A_positive & bird_filter_2,
    'B': zone_B_positive & bird_filter_2,
    'C': zone_C_positive & bird_filter_2
}

# Define sigmoid similarity
def sigmoid_similarity(error, mu, beta):
    return 1 / (1 + np.exp((error - mu) / beta))

# Compute errors from full dataset
duration_errors_all = np.abs(birds_songs_info.d_median - birds_songs_info.d0)
pitch_errors_all = np.abs(birds_songs_info.f_median - birds_songs_info.f0)

# Sigmoid parameters
mu_d = np.percentile(duration_errors_all, 50)
mu_f = np.percentile(pitch_errors_all, 50)
beta_d = 0.1 * (np.max(duration_errors_all) - np.min(duration_errors_all))
beta_f = 0.1 * (np.max(pitch_errors_all) - np.min(pitch_errors_all))

# Max similarities (at error = 0)
duration_sim_at_zero = sigmoid_similarity(0, mu_d, beta_d)
pitch_sim_at_zero = sigmoid_similarity(0, mu_f, beta_f)

print(f"mu_d = {mu_d:.4f}, beta_d = {beta_d:.4f}, duration_sim_at_zero = {duration_sim_at_zero:.4f}")
print(f"mu_f = {mu_f:.4f}, beta_f = {beta_f:.4f}, pitch_sim_at_zero = {pitch_sim_at_zero:.4f}")

fig = plt.figure(figsize=(8.2, 2.8))
gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
axs = []

last_im = None 

for i, region in enumerate(regions):
    mask = regions_masks[region]
    data = birds_songs_info[mask]

    duration_error = np.abs(data.d_median - data.d0)
    pitch_error = np.abs(data.f_median - data.f0)

    duration_similarity = sigmoid_similarity(duration_error, mu_d, beta_d)
    pitch_similarity = sigmoid_similarity(pitch_error, mu_f, beta_f)

    gray_mask = (duration_similarity < 0.5) & (pitch_similarity < 0.5)
    colored_mask = ~gray_mask

    # Create subplot for current region
    ax = fig.add_subplot(gs[0, i])
    axs.append(ax)

    # KDE
    xy = np.vstack([duration_similarity[colored_mask], pitch_similarity[colored_mask]])
    kde = gaussian_kde(xy, bw_method=0.4)
    xgrid, ygrid = np.mgrid[0:1:100j, 0:1:100j]
    grid_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])
    z = kde(grid_coords).reshape(100, 100)

    dx = 1 / 100
    dy = 1 / 100
    bin_area = dx * dy
    z_prob = z * bin_area

    im = ax.imshow(z_prob.T, origin='lower', extent=[0, 1, 0, 1],
                   cmap=custom_cmap, aspect='equal')
    last_im = im

    # Scatter
    ax.scatter(duration_similarity[gray_mask], pitch_similarity[gray_mask],
               s=20, alpha=0.2, color='gray', edgecolor='none')
    ax.scatter(duration_similarity[colored_mask], pitch_similarity[colored_mask],
               s=20, alpha=1.0, color='black', edgecolor='none')

    ax.plot([0, 1], [0, 1], 'r--', lw=1)
    ax.axvline(0.5, color='k', linestyle='--', lw=1)
    ax.axhline(0.5, color='k', linestyle='--', lw=1)

    # Axes settings
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_box_aspect(1)  

    ticks = np.linspace(0, 1, 5) 
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_title(f'Region {region}', fontsize=12)
    ax.set_xlabel('Duration similarity')
    if i == 0:
        ax.set_ylabel('Pitch similarity')

cbar_ax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(last_im, cax=cbar_ax)
cbar.set_label('Probability per 2D bin')

# Adjust layout without distorting aspect ratios
fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.15, wspace=0.3)
fig.align_labels()

mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/Exp2_similarities_with_filter.pdf', transparent=True)
plt.show()
#Statistical analysis
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

for i, region in enumerate(regions):
    mask = regions_masks[region]
    data = birds_songs_info[mask]

    duration_error = np.abs(data.d_median - data.d0)
    pitch_error = np.abs(data.f_median - data.f0)

    duration_similarity = sigmoid_similarity(duration_error, mu_d, beta_d)
    pitch_similarity = sigmoid_similarity(pitch_error, mu_f, beta_f)

    gray_mask = (duration_similarity < 0.5) & (pitch_similarity < 0.5)
    colored_mask = ~gray_mask

    x = duration_similarity[colored_mask]
    y = pitch_similarity[colored_mask]

    signed_dist = (y - x) / np.sqrt(2)

    # Wilcoxon test
    stat, p = wilcoxon(signed_dist)
    print(f"Region {region}: Wilcoxon stat = {stat:.2f}, p = {p:.4f}")

    # Compute histogram manually to get probabilities per bin
    counts, bins = np.histogram(signed_dist, bins=20, density=True)
    bin_width = np.diff(bins)[0]
    probs = counts * bin_width
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot probability histogram
    axs[i].bar(bin_centers, probs, width=bin_width, color='black', edgecolor='black')
    axs[i].axvline(0, color='red', linestyle='--', lw=2)
    axs[i].set_xlim(-0.7, 0.7)
    axs[i].set_title(f"Region {region}\nWilcoxon p = {p:.4f}", fontsize=11)
    axs[i].set_xlabel('Signed distance')

axs[0].set_ylabel('Probability')

plt.tight_layout()
plt.savefig('Plots/Exp2_signed_distances_masked_only.pdf', transparent=True)
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
n_components_range = range(1, 10)  
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
k_range = range(2, 10)  
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

fig, ax1 = plt.subplots(figsize=(5, 5))
# Plot BIC Score (Green) on the left y-axis
ax1.plot(range(1, 10), bic_scores, color='#008000', marker='o', linestyle='--', label='Duration')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('BIC', color='#008000')
ax1.tick_params(axis='y', labelcolor='#008000')
ax1.set_title('Cluster Evaluation Metrics')
ax1.set_box_aspect(1)

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_scores, color='#800080', marker='o', linestyle='-', label='Duration and frequency')
ax2.set_ylabel('Silhouette Score', color='#800080')
ax2.tick_params(axis='y', labelcolor='#800080')
ax2.set_box_aspect(1)

if 3 in k_range:
    idx = k_range.index(3)  
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

ax1 = axes[0]
kde = sns.kdeplot(control.duration, color='black', linestyle='-', lw=2, bw_adjust=0.4, ax=ax1)
# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()  

x_spacing = np.diff(x_values)

print(' Mean: ', np.mean(control.duration), " STD: ", np.std(control.duration))
print(' Median: ', np.median(control.duration), " MAD: ", np.median(np.abs(control.duration - np.median(control.duration))))
ax1.set_ylabel('Normalized counts per bin', fontsize=12)
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()


region_bounds = [0, 0.15, 0.3, 0.9]
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
ax2.set_xlabel('Whistle duration (s)')
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
for i in range(2, 5, 2): 
    axes[i].sharey(axes[0])
for i, (bird, data) in enumerate(zip(bird_names, bird_data)):
    ax_kde = axes[i * 2]       # Top row: KDE plot
    ax_scatter = axes[i * 2 + 1]  # Bottom row: Scatter plot
    print(bird, ' Mean: ', np.mean(data.duration), " STD: ", np.std(data.duration))
    print(bird, ' Median: ', np.median(data.duration), " MAD: ", np.median(np.abs(data.duration - np.median(data.duration))))
    # --- KDE Plot ---
    kde = sns.kdeplot(data.duration, color='black', linestyle='-', lw=2, bw_adjust=0.4, ax=ax_kde)
    x_values = ax_kde.lines[-1].get_xdata()  # Get the latest plotted KDE line
    print(bird, len(data.duration))
    x_spacing = np.diff(x_values)

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

    ax_scatter.set_yticks([])
    ax_scatter.set_ylabel("")
    
    if i == len(bird_names) - 1:
        ax_scatter.set_xlabel("Whistle duration (s)", fontsize=12)

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/Control_whistle_durations_three birds.pdf", transparent=True)
plt.show()

#--------------------------------------
#FIG S1C
#--------------------------------------
plt.style.use('default')

# Compute mean and standard deviation
mean_val = np.mean(control.temp_distance)
std_val = np.std(control.temp_distance, ddof=0)  

mean_str = f"{mean_val:.2g}"
std_str = f"{std_val:.2g}"

fig, ax = plt.subplots(figsize=(5, 5))  

# KDE plot
kde = sns.kdeplot(control.temp_distance, color='k', linestyle='-', lw=3, bw_adjust=0.5, ax=ax)
x_values = ax.lines[-1].get_xdata() 

x_spacing = np.diff(x_values)
print(' Median: ', np.nanmedian(control.temp_distance), " MAD: ", np.nanmedian(np.abs(control.temp_distance - np.nanmedian(control.temp_distance))))
print("Distance between consecutive x-values:", x_spacing[0])
# Get KDE data for shading
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()
ax.fill_between(x_values, y_values, color='gray', alpha=0.2)

ax.axvline(x=0, c='tomato', linestyle='--', lw=3)

ax.set_ylabel('normalized frequency')
ax.set_xlabel('distance subsequent')
ax.set_box_aspect(1)  
ax.legend([f"Mean: {mean_str}, SD: {std_str}"], loc="upper right", frameon=False)
fig.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/control_distribution_d_distance.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S1D
#--------------------------------------
plt.style.use('default')

mean_val = np.mean(whistle_songs_control.d_first_to_last)
std_val = np.std(whistle_songs_control.d_first_to_last, ddof=0) 


mean_str = f"{mean_val:.2g}"
std_str = f"{std_val:.2g}"


fig, ax = plt.subplots(figsize=(5, 5))
# KDE plot
kde = sns.kdeplot(whistle_songs_control.d_first_to_last, color='black', linestyle='-', lw=3, bw_adjust=0.5, ax=ax)
x_values = ax.lines[-1].get_xdata()  
print(' Median: ', np.median(whistle_songs_control.d_first_to_last), " MAD: ", np.median(np.abs(whistle_songs_control.d_first_to_last - np.median(whistle_songs_control.d_first_to_last))))
x_spacing = np.diff(x_values)
print("Distance between consecutive x-values:", x_spacing[0])
ax1.set_ylabel('Normalized counts per bin', fontsize=12)
# Get KDE data for shading
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()
ax.fill_between(x_values, y_values, color='gray', alpha=0.2)

ax.axvline(x=0, c='tomato', linestyle='--', lw=3)
ax.set_ylabel('normalized frequency')
ax.set_xlabel('first to last (s) within one song')
ax.set_box_aspect(1) 
ax.legend([f"Mean: {mean_str}, SD: {std_str}"], loc="upper left", frameon=False)
fig.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/control_distribution_d_first-last.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S1E
#--------------------------------------
df = control[['duration', 'gap']].dropna()

slope, intercept, r_value, _, _ = linregress(df.duration, df.gap)
r_squared = r_value**2

x_vals = np.linspace(df.duration.min(), df.duration.max(), 100)
y_vals = slope * x_vals + intercept

plt.style.use('default')

fig, ax = plt.subplots(figsize=(5, 5))
# Scatter plot
ax.scatter(df.duration, df.gap, color='k', s=3, alpha=0.5)
ax.set_ylim([0,0.6])
ax.set_xlim([0,0.9])
# Regression line
ax.plot(x_vals, y_vals, color='r', linewidth=2, label=f'(R²={r_squared:.2f})')
ax.set_xlabel('Whistle duration (s)')
ax.set_ylabel('Gap duration (s)')

ax.set_box_aspect(1) 
ax.legend(frameon=False)
fig.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Control_gapvsduration.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S1F
#--------------------------------------
plt.style.use('default')

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 4),
                         sharex=True,
                         gridspec_kw={'height_ratios': [2, 1.5, 2, 1.5]})
# --------------------------------------
# Control density 
# --------------------------------------
kde = sns.kdeplot(whistle_songs_control.d_median, color='black', linestyle='-', lw=2,
                  bw_adjust=0.4, common_norm=True, ax=axes[0])
x_vals = kde.lines[0].get_xdata()
y_vals = kde.lines[0].get_ydata()
dx = np.mean(np.diff(x_vals))

# Convert density to probability
y_vals_prob = y_vals * dx

axes[0].lines[-1].remove()
axes[0].plot(x_vals, y_vals_prob, color='black', linestyle='-', lw=2)
axes[0].fill_between(x_vals, y_vals_prob, color='gray', alpha=0.2)
axes[0].set_xlim([0, 0.9])
axes[0].set_ylabel("Probability")

control_data = whistle_songs_control.d_median
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(control_data))
axes[1].scatter(control_data, np.zeros_like(control_data) + y_jittered,
                color='black', alpha=0.5, s=2)
axes[1].set_yticks([])
axes[1].set_ylabel("Control")
# --------------------------------------
# Pitch match data
# --------------------------------------
kde_control = gaussian_kde(whistle_songs_control.d_median, bw_method=0.5)
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=0, vmax=2)

data = whistle_songs_pitch[np.abs(whistle_songs_pitch.f_median - whistle_songs_pitch.f0) < 100]
print(category, len(data.d_median))

# Plot pitch KDE using Seaborn
kde = sns.kdeplot(data.d_median, color='black', linestyle='-', lw=2,
                  bw_adjust=0.4, ax=axes[2])

x_vals_pitch = kde.lines[0].get_xdata()
y_vals_pitch = kde.lines[0].get_ydata()
dx_pitch = np.mean(np.diff(x_vals_pitch))
y_vals_prob_pitch = y_vals_pitch * dx_pitch

axes[2].lines[-1].remove()
axes[2].plot(x_vals_pitch, y_vals_prob_pitch, color='black', linestyle='-', lw=2)
axes[2].fill_between(x_vals_pitch, y_vals_prob_pitch, color='mediumaquamarine')
axes[2].set_ylabel("Probability")

# Compute KDE ratio
kde_r = gaussian_kde(data.d_median, bw_method=0.5)
ratios = np.clip(kde_r(data.d_median) / kde_control(data.d_median), 0, 2)

# Scatter Jittered Data with KDE ratio coloring
y_jittered = np.random.uniform(low=0, high=0.075, size=len(data.d_median))
axes[3].scatter(
    data.d_median, np.zeros_like(data.d_median) + y_jittered,
    c=ratios, cmap=cmap, norm=norm, edgecolor='none', s=10
)
axes[3].vlines(0.440, -0.08, -0.15, color='mediumaquamarine', lw=2, zorder=5)
axes[3].set_yticks([])
axes[3].set_ylabel(category)

print(category, ' Mean: ', np.mean(data.d_median), " STD: ", np.std(data.d_median))
print(category, ' Median: ', np.median(data.d_median),
      " MAD: ", np.median(np.abs(data.d_median - np.median(data.d_median))))

ylim_max = max(max(y_vals_prob), max(y_vals_prob_pitch)) * 1.05
axes[0].set_ylim(0, ylim_max)
axes[2].set_ylim(0, ylim_max)

axes[3].set_xlabel('Whistle duration (s)')

plt.tight_layout(rect=[0, 0, 0.9, 1])
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Results exp_pitch.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S1G
#--------------------------------------
# Extract durations
control_durations = whistle_songs_control.d_median.values
playback_durations = data.d_median.values  

observed_diff = np.median(playback_durations) - np.median(control_durations)
combined = np.concatenate([control_durations, playback_durations])
n_control = len(control_durations)
n_iterations = 10000
null_distribution = []

# Permutation test
for _ in range(n_iterations):
    np.random.shuffle(combined)
    perm_control = combined[:n_control]
    perm_playback = combined[n_control:]
    diff = np.median(perm_playback) - np.median(perm_control)
    null_distribution.append(diff)

# Convert to array
null_distribution = np.array(null_distribution)

# Two-sided p-value
p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))

# Print results
print("Observed median difference:", observed_diff)
print("p-value (2-sided):", p_value)

plt.figure(figsize=(4, 4)) 
counts, bins, patches = plt.hist(null_distribution, bins=40, color='mediumaquamarine')  
plt.cla()

# Plot counts instead of probability
plt.bar(bins[:-1], counts, width=bins[1] - bins[0], align='edge', color='mediumaquamarine')
plt.axvline(observed_diff, color='red', linestyle='--', linewidth=2, label='Observed diff')
plt.xlabel('Median difference (Playback - Control)')
plt.ylabel('Count')  # Updated label
plt.title(f'Permutation Test\np-value = {p_value:.4f}')
plt.legend(frameon=False)

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/pitchexp_p_value_counts.pdf', transparent=True)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------                              
#FIG S2-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------------------
#FIG S2A
#--------------------------------------
def compute_medians(data):
    median_values = {category: np.nan for category in CATEGORY}  # Ensure all regions are included
    for category in CATEGORY:
        region_data = data[data["category"] == category]["d_median"].dropna()  # Drop NaNs
        if not region_data.empty:
            median_values[category] = np.median(region_data)
            print(category, ' Median: ', np.median(region_data), " MAD: ", np.median(np.abs(region_data - np.median(region_data))))
            print(category, 'Mean: ', np.mean(region_data), 'std: ', np.std(region_data))
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
            print(bird, category, len(region_data.d_average)) 
            # Scatter plot with jitter
            y_jittered = np.random.uniform(low =-0.2, high=0.2, size=len(region_data))
            x_values = region_data["d_average"].dropna()
            ax.scatter(x_values, y_jittered, color=COLORS[row], s=10)

            ax.set_yticks([])


            for x in playback_times.get(category, []):
                ax.vlines(x, ymin=-0.65, ymax=-0.35, colors='gray', linewidth=2)

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
plt.style.use('default')
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
    print(category, ' Median: ', np.median(data.d_median), " MAD: ", np.median(np.abs(data.d_median - np.median(data.d_median))))
    print(category, 'Mean: ', np.mean(data.d_median), 'std: ', np.std(data.d_median))
    axes[2 * (idx + 1) + 1].vlines(median_value, -0.08, -0.2, color='black', lw=2, linestyle='-', zorder=6)
    
    axes[2 * (idx + 1) + 1].set_yticks([])
    axes[2 * (idx + 1) + 1].set_ylabel(category)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(scatter, cax=cbar_ax)
cbar.set_label("KDE Ratio")

plt.tight_layout(rect=[0, 0, 0.9, 1])
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Shuffled results exp1.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S2 C,D
#--------------------------------------
def plot_duration_and_pitch_errors(
    data_set,
    bw_value=0.2,
    zero_line=True,
    savepath='Plots/Duration_Pitch_errors_exp_1.pdf'
):
    try:
        cats = CATEGORY
    except NameError:
        cats = ['R1', 'R2', 'R3', 'R4']

    fig, ax = plt.subplots(2, 4, figsize=(3, 5),
                           sharex=True, sharey='row')

    mode_positions_duration = []
    mode_positions_pitch = []
    cmap = plt.cm.cividis

    # ------------------------------------------------------
    # 1. DURATION ERROR (TOP ROW)
    # ------------------------------------------------------
    for idx, category in enumerate(cats):
        dreg = data_set[data_set['category'] == category].copy()
        dreg = dreg.dropna(subset=['d_median', 'd0'])
        n_points = len(dreg)
        if n_points == 0:
            print(f"[Duration] {category}: no data points.")
            mode_positions_duration.append(np.nan)
            continue

        delta_T = dreg['d_median'].values - dreg['d0'].values

        if len(delta_T) >= 2 and np.std(delta_T) > 0:
            kde = gaussian_kde(delta_T, bw_method=bw_value)
            dens = kde(delta_T)
            norm = plt.Normalize(vmin=dens.min(), vmax=dens.max())
            colors = cmap(norm(dens))
            x_grid = np.linspace(np.min(delta_T), np.max(delta_T), 500)
            kde_vals = kde(x_grid)
            mode_val = x_grid[np.argmax(kde_vals)]
        else:
            colors = cmap(np.full_like(delta_T, 0.6, dtype=float))
            mode_val = np.nan

        print(f"[Duration] {category}: mode = {mode_val:.3f} s, N = {n_points}")

        jitter = np.random.uniform(-0.15, 0.15, size=len(delta_T))
        ax[0, idx].scatter(jitter, delta_T, s=20, edgecolor='none', c=colors)
        ax[0, idx].set_xlim(-0.2, 0.2)
        ax[0, idx].set_xticks([])
        ax[0, idx].set_title(f'{category}', fontsize=11, fontweight='bold')

        if zero_line:
            ax[0, idx].hlines(0, xmin=-0.2, xmax=0.2, color='k', linestyle='--', linewidth=1)
        if not np.isnan(mode_val):
            ax[0, idx].hlines(mode_val, xmin=0.2, xmax=-0.2, color='r', linestyle='-', linewidth=2)

        if idx == 0:
            ax[0, idx].set_ylabel('Response - playback duration (s)', fontsize=10)

        mode_positions_duration.append(mode_val)

    # ------------------------------------------------------
    # 2. PITCH ERROR (BOTTOM ROW)
    # ------------------------------------------------------
    for idx, category in enumerate(cats):
        dreg = data_set[data_set['category'] == category].copy()
        dreg = dreg.dropna(subset=['f_median', 'f0'])
        n_points = len(dreg)
        if n_points == 0:
            print(f"[Pitch] {category}: no data points.")
            mode_positions_pitch.append(np.nan)
            continue

        delta_F = (dreg['f_median'].values - dreg['f0'].values) / 1000.0  # Hz → kHz

        if len(delta_F) >= 2 and np.std(delta_F) > 0:
            kde = gaussian_kde(delta_F, bw_method=bw_value)
            dens = kde(delta_F)
            norm = plt.Normalize(vmin=dens.min(), vmax=dens.max())
            colors = cmap(norm(dens))
            x_grid = np.linspace(np.min(delta_F), np.max(delta_F), 500)
            kde_vals = kde(x_grid)
            mode_val = x_grid[np.argmax(kde_vals)]
        else:
            colors = cmap(np.full_like(delta_F, 0.6, dtype=float))
            mode_val = np.nan

        print(f"[Pitch] {category}: mode = {mode_val:.3f} kHz, N = {n_points}")

        jitter = np.random.uniform(-0.15, 0.15, size=len(delta_F))
        ax[1, idx].scatter(jitter, delta_F, s=20, edgecolor='none', c=colors)
        ax[1, idx].set_xlim(-0.2, 0.2)
        ax[1, idx].set_xticks([])

        if zero_line:
            ax[1, idx].hlines(0, xmin=-0.2, xmax=0.2, color='k', linestyle='--', linewidth=1)
        if not np.isnan(mode_val):
            ax[1, idx].hlines(mode_val, xmin=0.2, xmax=-0.2, color='r', linestyle='-', linewidth=2)

        if idx == 0:
            ax[1, idx].set_ylabel('Response - playback pitch (kHz)', fontsize=10)

        mode_positions_pitch.append(mode_val)

    plt.subplots_adjust(left=0.1, right=0.95, hspace=0.3, wspace=0.3)
    plt.tight_layout()
    mpl.rcParams['pdf.fonttype'] = 42
    plt.savefig(savepath, transparent=True)
    plt.show()

    return mode_positions_duration, mode_positions_pitch

mode_dur_exp1, mode_pitch_exp1 = plot_duration_and_pitch_errors(
    filtered_data_1,
    bw_value=0.2,
    savepath='Plots/Duration_Pitch_errors_exp1.pdf')

# --------------------------------------
# Fig S2 E
# --------------------------------------
plt.style.use('default')

all_pitch = whistle_songs_pitch.copy()
all_pitch = all_pitch.dropna(subset=['d_median', 'd0', 'f0'])
all_pitch['abs_dur_err'] = np.abs(all_pitch['d_median'] - all_pitch['d0'])

fig3, ax3 = plt.subplots(figsize=(7, 2))
cmap = plt.cm.cividis

pitches = np.sort(all_pitch['f0'].unique())
xpos = np.arange(len(pitches))

mode_values = []  

for i, p in enumerate(pitches):
    sub = all_pitch[all_pitch['f0'] == p]['abs_dur_err'].dropna().values
    if len(sub) == 0:
        mode_values.append(np.nan)
        continue

    if len(sub) >= 2 and np.std(sub) > 0:
        kde = gaussian_kde(sub, bw_method=0.5)
        dens = kde(sub)
        norm = plt.Normalize(vmin=dens.min(), vmax=dens.max())
        colors = cmap(norm(dens))

        x_grid = np.linspace(np.min(sub), np.max(sub), 400)
        kde_vals = kde(x_grid)
        mode_val = x_grid[np.argmax(kde_vals)]
    else:
        colors = cmap(np.full(len(sub), 0.6))
        mode_val = np.nan

    jitter = np.random.uniform(-0.25, 0.25, size=len(sub))
    ax3.scatter(xpos[i] + jitter, sub, s=12, edgecolor='none', c=colors, zorder=3)

    if not np.isnan(mode_val):
        ax3.hlines(mode_val, xpos[i] - 0.25, xpos[i] + 0.25, color='r', linestyle='-', linewidth=2, zorder=4)

    mode_values.append(mode_val)

ax3.set_xlim(-0.6, len(pitches) - 0.4)
ax3.set_xticks(xpos)
ax3.set_xticklabels([f"{p/1000:.1f}" for p in pitches])
ax3.set_xlabel('Playback pitch (kHz)')
ax3.set_ylabel(r'|ΔT| (s)')

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
fig3.savefig('Plots/Results_exp_pitch_abs_error_all_responses.pdf', transparent=True)
plt.show()

print("KDE mode (max density) per playback pitch:")
for p, m in zip(pitches, mode_values):
    print(f"{p/1000:.1f} kHz → {m:.3f} s")

mode_values_clean = np.array(mode_values, dtype=float)
mode_values_clean = mode_values_clean[~np.isnan(mode_values_clean)]

if len(mode_values_clean) > 0:
    mean_mode = np.mean(mode_values_clean)
    min_mode = np.min(mode_values_clean)
    print(f"\nMean of red line (KDE modes): {mean_mode:.3f} s")
    print(f"Minimum of red line (KDE modes): {min_mode:.3f} s")
else:
    print("\nNo valid KDE modes to summarize.")
    
# --------------------------------------
# Fig S2 F
# --------------------------------------
def _match_counts_by_region(data_set, dur_thresh=0.04, pitch_thresh=100):

    try:
        cats = CATEGORY
    except NameError:
        cats = ['R1', 'R2', 'R3', 'R4']

    counts_list = []
    totals_all = []
    neither_list = []

    for cat in cats:
        dreg = data_set[data_set['category'] == cat].copy()
        dreg = dreg.dropna(subset=['d_median', 'd0', 'f_median', 'f0'])

        n_all = len(dreg)
        if n_all == 0:
            counts_list.append([0, 0, 0])
            totals_all.append(0)
            neither_list.append(0)
            continue

        dur_err   = np.abs(dreg['d_median'].values - dreg['d0'].values)
        pitch_err = np.abs(dreg['f_median'].values - dreg['f0'].values)

        dur_match   = dur_err   <= dur_thresh
        pitch_match = pitch_err <= pitch_thresh

        both        = np.sum(dur_match & pitch_match)
        dur_only    = np.sum(dur_match & ~pitch_match)
        pitch_only  = np.sum(~dur_match & pitch_match)
        matched_sum = both + dur_only + pitch_only
        neither     = n_all - matched_sum

        counts_list.append([both, dur_only, pitch_only])
        totals_all.append(n_all)
        neither_list.append(neither)

    return cats, np.vstack(counts_list).astype(int), np.array(totals_all, int), np.array(neither_list, int)


def plot_match_grouped_focus(
    data_set,
    dur_thresh=0.04,
    pitch_thresh=100,
    savepath='Plots/Match_grouped_focus_exp1_custom_edges.pdf',
):

    cats, counts3, totals_all, neither_counts = _match_counts_by_region(
        data_set, dur_thresh, pitch_thresh
    )

    props = counts3 / np.clip(totals_all[:, None], 1, None)

    labels = ['Both', 'Duration only', 'Pitch only']
    colors = ['linen', 'salmon', 'darkkhaki']  

    x = np.arange(len(cats))
    width = 0.22

    mpl.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(4, 3))

    for i, (lab, col) in enumerate(zip(labels, colors)):
        ax.bar(
            x + (i - 1)*width,
            props[:, i],
            width,
            label=lab,
            color=col,
            edgecolor='dimgray',
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10, fontweight='bold')
    ax.set_ylabel('Proportion of responses')
    ax.set_ylim(0, 1)  # since we’re plotting proportions over all responses
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_linewidth(1)
    ax.grid(False)

    ax.legend(
        frameon=False, ncol=3, loc='upper center',
        bbox_to_anchor=(0.5, 1.25), handlelength=1
    )
    plt.ylim([0,0.4])
    plt.tight_layout()
    fig.savefig(savepath, transparent=True, bbox_inches='tight')
    plt.show()

    print("Region | Both | DurOnly | PitchOnly | Neither | Total")
    for cat, c3, n_neither, tot in zip(cats, counts3, neither_counts, totals_all):
        print(f"{cat:>6} | {c3[0]:>4} | {c3[1]:>7} | {c3[2]:>9} | {n_neither:>7} | {tot}")

plot_match_grouped_focus(
    filtered_data_1,
    dur_thresh=0.04,
    pitch_thresh=100,
    savepath='Plots/Match_grouped_focus_exp1_custom_edges.pdf'
)

# --------------------------------------
# Fig S2 G
# --------------------------------------
def plot_match_histograms(
    data_set,
    dur_thresh=0.04,        
    pitch_thresh_hz=100,    
    bins=20,
    savepath='Plots/Hist_errors_matches_exp1_sharedy.pdf'
):

    df = data_set.dropna(subset=['d_median', 'd0', 'f_median', 'f0']).copy()
    dT = df['d_median'].values - df['d0'].values                     
    dF_khz = (df['f_median'].values - df['f0'].values) / 1000.0      

    pitch_match = np.abs(dF_khz * 1000) <= pitch_thresh_hz
    dur_match   = np.abs(dT) <= dur_thresh

    dT_given_pitch = dT[pitch_match]
    dF_given_dur   = dF_khz[dur_match]

    print(f"Number of pitch-match points: {len(dT_given_pitch)}")
    print(f"Number of duration-match points: {len(dF_given_dur)}")


    mpl.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(1, 2, figsize=(4,2), sharey=True)


    if dT_given_pitch.size > 0:
        edges_T = np.linspace(np.min(dT_given_pitch), np.max(dT_given_pitch), bins + 1)
        counts_T, bins_T = np.histogram(dT_given_pitch, bins=edges_T, density=True)
        bin_width_T = np.diff(bins_T)[0]
        ax[0].bar(
            (bins_T[:-1] + bins_T[1:]) / 2,
            counts_T * bin_width_T,  
            width=bin_width_T,
            color='darkkhaki',
            edgecolor='darkkhaki',
            linewidth=0.6
        )
    ax[0].set_xlabel('Duration error (s)')
    ax[0].set_ylabel('Probability')
    ax[0].spines[['top', 'right']].set_visible(False)

    if dF_given_dur.size > 0:
        edges_F = np.linspace(np.min(dF_given_dur), np.max(dF_given_dur), bins + 1)
        counts_F, bins_F = np.histogram(dF_given_dur, bins=edges_F, density=True)
        bin_width_F = np.diff(bins_F)[0]
        ax[1].bar(
            (bins_F[:-1] + bins_F[1:]) / 2,
            counts_F * bin_width_F,  
            width=bin_width_F,
            color='salmon',
            edgecolor='salmon',
            linewidth=0.6
        )
    ax[1].set_xlabel('Pitch error (kHz)')
    ax[1].spines[['top', 'right']].set_visible(False)

    for a in ax:
        a.spines[['left', 'bottom']].set_linewidth(1)
        a.tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    fig.savefig(savepath, transparent=True, bbox_inches='tight')
    plt.show()

plot_match_histograms(
    filtered_data_1,
    dur_thresh=0.04,
    pitch_thresh_hz=100,
    bins=15,
    savepath='Plots/Hist_errors_matches_exp1_sharedy.pdf'
)

# --------------------------------------
# Fig S2 H
# --------------------------------------

def plot_gmm_bic_by_region(
    data_set,
    feature='d_median',
    max_components=5,
    savepath='Plots/GMM_BIC_by_region_exp1.pdf',
    random_state=0
):

    try:
        cats = CATEGORY
    except NameError:
        cats = ['R1', 'R2', 'R3', 'R4']

    try:
        region_colors = dict(zip(CATEGORY, COLORS))
    except NameError:
        
        region_colors = {'R1': '#E78652', 'R2': '#DFA71F', 'R3': '#C69C6D', 'R4': '#7E4F25'}

    ks = np.arange(1, max_components + 1)
    bic_map = {}   
    best_k = {}    
    best_bic = {}  

    for cat in cats:
        df_reg = data_set[data_set['category'] == cat].dropna(subset=[feature]).copy()
        x = df_reg[feature].values.reshape(-1, 1)

        bics = []
        for k in ks:
            if len(x) < k:          # not enough points to fit k components
                bics.append(np.nan)
                continue
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    n_init=5,
                    random_state=random_state
                )
                gmm.fit(x)
                bics.append(gmm.bic(x))
            except Exception:
                bics.append(np.nan)

        bic_map[cat] = np.array(bics, dtype=float)
        
        finite_mask = np.isfinite(bic_map[cat])
        if not np.any(finite_mask):
            best_k[cat] = np.nan
            best_bic[cat] = np.nan
        else:
            idx = np.nanargmin(bic_map[cat])
            best_k[cat] = ks[idx]
            best_bic[cat] = bic_map[cat][idx]

    mpl.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(3,3))

    for cat in cats:
        color = region_colors.get(cat, 'gray')
        bics = bic_map[cat]
        ax.plot(ks, bics, marker='o', ms=4, lw=1.8, color=color, label=cat)
        if np.isfinite(best_bic[cat]):
            ax.scatter(best_k[cat], best_bic[cat],
                       marker='s', s=50, color=color,
                       edgecolor= color, linewidth=0.8, zorder=3)

    ax.set_xlabel('Number of Gaussian components')
    ax.set_ylabel('BIC')
    ax.set_xticks(ks)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_linewidth(1)
    # No grid (matches your current style)
    ax.legend(frameon=False, ncol=len(cats), loc='upper center', bbox_to_anchor=(0.5, 1.22), handlelength=1)

    plt.tight_layout()
    fig.savefig(savepath, transparent=True, bbox_inches='tight')
    plt.show()
    
    print("Best number of components per region (by min BIC):")
    for cat in cats:
        print(f"  {cat}: k = {best_k[cat]}  (BIC = {best_bic[cat]:.1f})")

# ---- Usage ----
plot_gmm_bic_by_region(
    filtered_data_1,
    feature='d_median',          # or 'last_d' etc., if you prefer
    max_components=5,
    savepath='Plots/GMM_BIC_by_region_exp1.pdf',
    random_state=0
)

#-----------------------------------------------------------------------------------------------------------------------------                              
#FIG S3-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------

#See Model scripts

#-----------------------------------------------------------------------------------------------------------------------------                              
#FIG S4-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
#--------------------------------------
#FIG S4A
#--------------------------------------
# Define three distinct purple tones
purple_shades = ['#6a0dad', '#9b5de5', '#d8b4f8']  
cluster_colors = purple_shades[:best_k]

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


for cluster in range(best_k):
    subset = control_clean[control_clean.cluster == cluster]
    center_x = subset['duration'].mean()
    center_y = subset['pitch_whistles'].mean() / 1000  
    std_x = subset['duration'].std()
    std_y = (subset['pitch_whistles'].std()) / 1000 
    print(f"Cluster {cluster}: Center at ({center_x:.3f} ± {std_x:.3f}, {center_y:.3f} ± {std_y:.3f} kHz)")
#--------------------------------------
#FIG S4B
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
#FIG S4C
#--------------------------------------
birds = filtered_data_2.bird.unique()  
plt.style.use('default')
fig, axes = plt.subplots(3, len(birds), figsize=(2 * len(birds), 6), sharex=True, sharey=True)
axes = np.atleast_2d(axes) 

regions = {
    "Region A": (zone_A_positive, "A"),
    "Region B": (zone_B_positive, "B"),
    "Region C": (zone_C_positive, "C"),
}

# Iterate through regions and birds
for j, (title, (zone_mask, key)) in enumerate(regions.items()):
    for i, bird in enumerate(birds):
        ax = axes[j, i]  

        # Filter data
        data = filtered_data_2[zone_mask & (filtered_data_2.bird == bird)]
        print(bird, title, len(data.d_median))
        # Scatter plots
        ax.scatter(data.d_median, data.f_median / 1000, color='k', s = 20, zorder=1)
        ax.plot(*playback_data_exp2[key], 
        color="#12B568", marker='+', linestyle='None', 
        markersize=7, markeredgewidth=3)

        ax.set_xlabel("Median duration")
        ax.set_ylabel("Median pitch")
        ax.set_xlim([0, 0.9])
        ax.set_ylim([0, 9])
        if j == 0:
            ax.set_title(bird) 
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/Exp2_per_region_with_medians_all_birds.pdf', transparent=True)
plt.show()
#------------------------------------
#FIG S4D
#------------------------------------
shuffled_responses_2 = shuffle_data(filtered_data_2)

zone_A_shuffled = shuffled_responses_2.d0 < 0.2
zone_B_shuffled = (shuffled_responses_2.f0 > 5000) & (shuffled_responses_2.d0 >= 0.5)
zone_C_shuffled = shuffled_responses_2.f0 < 4000

plt.style.use('default')

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

regions_shuffled = {
    'A': (shuffled_responses_2[zone_A_shuffled], playback_exp2_d_A, playback_exp2_f_A),
    'B': (shuffled_responses_2[zone_B_shuffled], playback_exp2_d_B, playback_exp2_f_B),
    'C': (shuffled_responses_2[zone_C_shuffled], playback_exp2_d_C, playback_exp2_f_C)
}

vmin, vmax = 0, 2
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Iterate over regions to create subplots
for i, (region, (data, playback_d, playback_f)) in enumerate(regions_shuffled.items()):
    
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
    ax_kde.set_box_aspect(1)
    ax_ratio = axes[1, i]

    x_exp, y_exp = data.d_median, data.f_median
    kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.3)
    
    x_ctrl, y_ctrl = whistle_songs_control.d_median, whistle_songs_control.f_median
    valid_mask = np.isfinite(x_ctrl) & np.isfinite(y_ctrl)
    kde_ctrl = gaussian_kde(np.vstack([x_ctrl[valid_mask], y_ctrl[valid_mask]]))

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
    ax_ratio.set_box_aspect(1) 

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

# Set common axis labels
fig.text(0.5, 0.04, "Median Duration (s)", ha='center', fontsize=12) 
fig.text(0.06, 0.5, "Median Pitch (Hz)", va='center', rotation='vertical', fontsize=12)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) 
fig.colorbar(sm, cax=cbar_ax, label='KDE Ratio')

mpl.rcParams['pdf.fonttype'] = 42

plt.savefig('Plots/Shuffled_Exp_2_per_region_and_ratio.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S4E
#--------------------------------------
plt.style.use('default')

cluster_labels = ['C1', 'C2', 'C3']  

scaler = StandardScaler()
control_scaled = scaler.fit_transform(control_clean[['duration', 'pitch_whistles']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
control_clean['cluster'] = kmeans.fit_predict(control_scaled)

# Decision boundary grid (for left column visuals)
x_min, x_max = control_clean['duration'].min() - 0.1, control_clean['duration'].max() + 0.1
y_min, y_max = control_clean['pitch_whistles'].min() - 0.1, control_clean['pitch_whistles'].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points)
Z = kmeans.predict(mesh_points_scaled).reshape(xx.shape)

def classify_and_label(df, scaler, kmeans):
    if not {'d_median', 'f_median'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'd_median' and 'f_median' columns.")
    new_data = df.rename(columns={'d_median': 'duration', 'f_median': 'pitch_whistles'}).copy()
    new_data_scaled = scaler.transform(new_data[['duration', 'pitch_whistles']])
    new_data['cluster'] = kmeans.predict(new_data_scaled)
    new_data['cluster_region'] = new_data['cluster'].map({0:'C1',1:'C2',2:'C3'})
    return new_data

def simulate_cluster_counts_df(df, scaler, kmeans, N=1000):
    """Return dict region->DataFrame (N rows x cluster_labels) with counts per iteration."""
    out = {'A': [], 'B': [], 'C': []}
    for _ in tqdm(range(N), desc="Simulating cluster counts"):
        shuf = shuffle_data(df)

        zone_A = shuf.d0 < 0.2
        zone_B = (shuf.f0 > 5000) & (shuf.d0 >= 0.5)
        zone_C = shuf.f0 < 4000

        for key, mask in [('A', zone_A), ('B', zone_B), ('C', zone_C)]:
            labeled = classify_and_label(shuf[mask], scaler, kmeans)
            vc = labeled['cluster_region'].value_counts()
            
            vc = vc.reindex(cluster_labels, fill_value=0)
            out[key].append(vc)

    
    return {k: pd.DataFrame(v)[cluster_labels] for k, v in out.items()}

# Real counts 
data_expA_labeled = classify_and_label(filtered_data_2[zone_A_positive], scaler, kmeans)
data_expB_labeled = classify_and_label(filtered_data_2[zone_B_positive], scaler, kmeans)
data_expC_labeled = classify_and_label(filtered_data_2[zone_C_positive], scaler, kmeans)

real_counts = [
    data_expA_labeled['cluster_region'].value_counts().reindex(cluster_labels, fill_value=0),
    data_expB_labeled['cluster_region'].value_counts().reindex(cluster_labels, fill_value=0),
    data_expC_labeled['cluster_region'].value_counts().reindex(cluster_labels, fill_value=0),
]
regions = ['A', 'B', 'C']


sim = simulate_cluster_counts_df(filtered_data_2, scaler, kmeans, N=1000)
sim_pct = {k: df.div(df.sum(axis=1).replace(0, np.nan), axis=0) * 100 for k, df in sim.items()}

fig, axes = plt.subplots(len(cluster_labels) + 1, len(regions) + 1, figsize=(8, 8))

# Top row 
for i, region in enumerate(regions):
    ax = axes[0, i + 1]
    # Outside contour
    sns.kdeplot(x=whistle_songs_control.d_median,
                y=whistle_songs_control.f_median/1000,
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax)
    # Region playbacks
    if region == 'A':
        ax.plot(playback_exp2_d_A, playback_exp2_f_A, color="#12B568", marker='+',
                linestyle='None', markersize=10, markeredgewidth=4)
    elif region == 'B':
        ax.plot(playback_exp2_d_B, playback_exp2_f_B, color="#12B568", marker='+',
                linestyle='None', markersize=10, markeredgewidth=4)
    else:
        ax.plot(playback_exp2_d_C, playback_exp2_f_C, color="#12B568", marker='+',
                linestyle='None', markersize=10, markeredgewidth=4)
    ax.text(0.5, 0.5, f"Region {region}", fontsize=12, ha='center', va='center', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Left column
for j, cluster in enumerate(cluster_labels):
    ax = axes[j + 1, 0]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Purples_r')
    sns.kdeplot(x=whistle_songs_control.d_median,
                y=whistle_songs_control.f_median,
                cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1, ax=ax)
    for i_c in range(3):
        subset = control_clean[control_clean['cluster'] == i_c]
        ax.scatter(subset['duration'], subset['pitch_whistles'], s=1,
                   color=purple_shades[i_c], alpha=(1.0 if (i_c == j) else 0.0))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'Cluster {cluster}')
    for spine in ax.spines.values():
        spine.set_visible(False)

# Histograms: 
for i, region in enumerate(regions):
    perc_df = sim_pct[region]   # (N x 3) percentages, columns in cluster_labels order
    obs_series = real_counts[i]
    obs_total = obs_series.sum()
    obs_pct = (obs_series / (obs_total if obs_total > 0 else 1)) * 100

    for j, (cluster, color) in enumerate(zip(cluster_labels, purple_shades)):
        ax = axes[j + 1, i + 1]
        vals = perc_df[cluster].dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=20, edgecolor=color, facecolor=color)
        # Observed vertical line
        ax.axvline(obs_pct[cluster], color='black', linestyle='-', linewidth=2)
        ax.set_title(f"{cluster} – R{region}", fontsize=10)
        ax.set_box_aspect(1)
        ax.set_xlabel('%'); ax.set_ylabel('Freq')

for r in range(1, len(cluster_labels) + 1):
    for c in range(1, len(regions) + 1):
        if r == 1:
            axes[r, c].sharex(axes[1, 1])
        if c == 1:
            axes[r, c].sharey(axes[1, 1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Exp2_p-values_table_with_contours.pdf', transparent=True)
plt.show()
#--------------------------------------
#FIG S4F
#--------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)  # 3 square subplots

for i, region in enumerate(regions):
    # Get region data
    mask = regions_masks[region]
    data = birds_songs_info[mask]

    # Compute similarities
    duration_error = np.abs(data.d_median - data.d0)
    pitch_error = np.abs(data.f_median - data.f0)

    duration_similarity = sigmoid_similarity(duration_error, mu_d, beta_d)
    pitch_similarity = sigmoid_similarity(pitch_error, mu_f, beta_f)

    # Masks
    gray_mask = (duration_similarity < 0.5) & (pitch_similarity < 0.5)
    colored_mask = ~gray_mask

    ax = axs[i]

    # Plot gray low-similarity responses
    ax.scatter(data.d_median[gray_mask], data.f_median[gray_mask] / 1000,
               s=20, alpha=0.2, color='gray', edgecolor='none')

    # Plot black high-similarity responses
    ax.scatter(data.d_median[colored_mask], data.f_median[colored_mask] / 1000,
               s=20, alpha=1.0, color='black', edgecolor='none')

    # KDE plot
    sns.kdeplot(x=whistle_songs_control.d_median, 
                y=whistle_songs_control.f_median / 1000, 
                cmap="Reds_r", fill=False,
                bw_adjust=0.7, levels=1, thresh=0.1, ax=ax)

    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 9)

    ax.set_box_aspect(1)

    # Labels and title
    ax.set_title(f'Region {region}', fontsize=12)
    ax.set_xlabel('Median duration (s)')
    if i == 0:
        ax.set_ylabel('Median pitch (kHz)')

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/Exp2_similarities_filtered_data.pdf', transparent=True)
plt.show()
