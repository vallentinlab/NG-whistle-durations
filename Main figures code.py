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
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
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
#BIRDS DATA (from previous season)
#------------------------------------------------------------------------------
all_birds= pd.read_pickle(r"Control prev season/all_birds.pkl")
#------------------------------------------------------------------------------
#PITCH DATA (from previous season)
#------------------------------------------------------------------------------
pitch_pb = all_birds[all_birds.phase == 'playback']
pitch_pb.to_excel('pitch_previous_season.xlsx', index=False)

#song analysis for the pitch experiment from previous season
whistle_songs_pitch = [] # Initialize an empty list to store the data
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
#CONTROL DATA (from previous season)
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

    # Perform bootstrap sampling (with replacement) on `d_median` and `f_median`
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

def _clear_axis_collections(ax):
    """Safely remove all collections currently on the axis (works across Matplotlib versions)."""
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

# Make KDE plots share the y-axis
for i in range(0, 10, 2):
    axes[i].sharey(axes[4])

# To gather max y across all KDE panels after scaling
ymax_list = []

# --------------------------------------
# Control KDE -> probability per bin
# --------------------------------------
kde = sns.kdeplot(
    whistle_songs_control.d_median,
    color='black', linestyle='-', lw=2, bw_adjust=0.4,
    ax=axes[0]
)

# Extract the drawn KDE curve (density)
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()

# Use the control grid spacing as a common "bin width" (Δx) for all panels
dx_ref = float(np.mean(np.diff(x_values)))
print("Δx (bin width) =", dx_ref)

# Convert density to probability per bin
y_prob = y_values * dx_ref

# Update the plot to show probability per bin
kde.lines[0].set_ydata(y_prob)
# remove any auto-added collections on this axis (if any)
_clear_axis_collections(axes[0])
axes[0].fill_between(x_values, y_prob, color='gray', alpha=0.2, zorder=1)
axes[0].set_xlim([0, 0.9])
axes[0].set_ylabel("Probability")
axes[0].relim()
axes[0].autoscale_view()
ymax_list.append(np.nanmax(y_prob))

# --------------------------------------
# Scatter Jittered Data for Control (unchanged)
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
# FIG 1C: Experimental categories
# --------------------------------------
# KDE for ratios (unchanged; Δx cancels in the ratio)
kde_control = gaussian_kde(whistle_songs_control.d_median, bw_method=0.5)
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=0, vmax=2)

scatter = None  # will hold the last scatter for the colorbar
for idx, (category, color) in enumerate(zip(CATEGORY, COLORS)):
    data = filtered_data_1[filtered_data_1["category"] == category]
    print(category, "N =", len(data.d_median))

    # KDE line (density -> probability per bin)
    kde_ax = axes[2 * (idx + 1)]
    kde_line = sns.kdeplot(
        data.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.4, ax=kde_ax
    )

    x_vals_exp = kde_line.lines[0].get_xdata()
    y_vals_exp = kde_line.lines[0].get_ydata()
    y_prob_exp = y_vals_exp * dx_ref  # convert using common Δx

    kde_line.lines[0].set_ydata(y_prob_exp)
    _clear_axis_collections(kde_ax)
    kde_ax.fill_between(x_vals_exp, y_prob_exp, color=color, zorder=1)
    kde_ax.set_ylabel("Probability")
    kde_ax.relim()
    kde_ax.autoscale_view()

    ymax_list.append(np.nanmax(y_prob_exp))

    # KDE ratio for scatter coloring (unchanged)
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

# --------------------------------------
# Enforce consistent y-limits across all KDE panels (shared y-axis)
# --------------------------------------
ymax = float(max(ymax_list)) * 1.05  # small headroom
for ax in axes[::2]:  # KDE rows: 0,2,4,6,8
    ax.set_ylim(0, ymax)

# --------------------------------------
# Colorbar for KDE ratio (unchanged)
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
#FIG 1F
#--------------------------------------

# -------------------------
# 1. Prepare data (Hz → kHz)
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

# Plot boundaries (edges)
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
    print(region, len(data.d_median))
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

print(' Mean: ', np.mean(control.duration), " STD: ", np.std(control.duration))
print(' Median: ', np.median(control.duration), " MAD: ", np.median(np.abs(control.duration - np.median(control.duration))))
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
for i in range(2, 5, 2):  # Indices of KDE plots (excluding the first one)
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


for cluster in range(best_k):
    subset = control_clean[control_clean.cluster == cluster]
    center_x = subset['duration'].mean()
    center_y = subset['pitch_whistles'].mean() / 1000  # Convert to kHz
    std_x = subset['duration'].std()
    std_y = (subset['pitch_whistles'].std()) / 1000  # Convert to kHz

    print(f"Cluster {cluster}: Center at ({center_x:.3f} ± {std_x:.3f}, {center_y:.3f} ± {std_y:.3f} kHz)")
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
print(' Median: ', np.median(whistle_songs_control.d_first_to_last), " MAD: ", np.median(np.abs(whistle_songs_control.d_first_to_last - np.median(whistle_songs_control.d_first_to_last))))
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
print(' Median: ', np.nanmedian(control.temp_distance), " MAD: ", np.nanmedian(np.abs(control.temp_distance - np.nanmedian(control.temp_distance))))
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
#FIG S1G
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
    print(category, ' Median: ', np.median(data.d_median), " MAD: ", np.median(np.abs(data.d_median - np.median(data.d_median))))
    print(category, 'Mean: ', np.mean(data.d_median), 'std: ', np.std(data.d_median))
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
        print(bird, title, len(data.d_median))
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

# Create meshgrid for decision boundaries
x_min, x_max = control_clean['duration'].min() - 0.1, control_clean['duration'].max() + 0.1
y_min, y_max = control_clean['pitch_whistles'].min() - 0.1, control_clean['pitch_whistles'].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predict cluster labels for each point in the meshgrid
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points) 
Z = kmeans.predict(mesh_points_scaled)
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Purples_r')  # Decision regions
for i in range(3):
    subset = control_clean[control_clean['cluster'] == i]
    plt.scatter(subset['duration'], subset['pitch_whistles'], label=f'C{i + 1}', color = purple_shades[i])
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
simulated_counts = simulate_cluster_counts(filtered_data_2, scaler, kmeans, shuffle_data, N=1)

# Convert to NumPy arrays
counts_A = np.array(simulated_counts['A'])
counts_B = np.array(simulated_counts['B'])
counts_C = np.array(simulated_counts['C'])

# Define settings
regions = ['A', 'B', 'C']
cluster_labels = ['C1', 'C2', 'C3']

data_list = [counts_A, counts_B, counts_C]
real_counts = [real_counts_A, real_counts_B, real_counts_C]


# Define figure size with extra row (for regions) and extra column (for clusters)
fig, axes = plt.subplots(len(cluster_labels) + 1, len(regions) + 1, figsize=(8, 8))
# Fill in the extra row (region labels and contours with playbacks)
for i, region in enumerate(regions):
    ax = axes[0, i + 1]  # Top row, skipping first column
    
    # Plot the outside contour for the region
    outside = sns.kdeplot(x=whistle_songs_control.d_median, 
                       y=whistle_songs_control.f_median/1000, 
                       cmap="Reds_r", fill=False,
                       bw_adjust=0.7, levels=1, thresh=0.1, ax=ax)
    # Scatter plot the playback points for each region (A, B, or C)
    if region == 'A':
        ax.plot(playback_exp2_d_A, playback_exp2_f_A, 
                color="#12B568", marker='+', linestyle='None', 
                markersize=10, markeredgewidth=4)
    elif region == 'B':
        ax.plot(playback_exp2_d_B, playback_exp2_f_B, 
                color="#12B568", marker='+', linestyle='None', 
                markersize=10, markeredgewidth=4)
    elif region == 'C':
        ax.plot(playback_exp2_d_C, playback_exp2_f_C, 
                color="#12B568", marker='+', linestyle='None', 
                markersize=10, markeredgewidth=4)
        
    # Labeling the region
    ax.text(0.5, 0.5, f"Region {region}", fontsize=12, ha='center', va='center', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)  # Hide all spines

# Fill in the extra row (region labels)
for i, region in enumerate(regions):
    ax = axes[0, i + 1]  # Top row, skipping first column
    ax.text(0.5, 0.5, f"Region {region}", fontsize=12, ha='center', va='center', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)  # Hide all spines

# Fill in the extra column (contour plots)
for j, cluster in enumerate(cluster_labels):
    ax = axes[j + 1, 0]  # Leftmost column, skipping first row

    # Plot decision boundaries with transparency
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Purples_r')
    outside = sns.kdeplot(x=whistle_songs_control.d_median, 
                       y=whistle_songs_control.f_median, 
                       cmap="Reds_r", fill=False,
                       bw_adjust=0.7, levels=1, thresh=0.1, ax = ax)
    # Plot clusters with only the current one highlighted
    for i in range(3):
        subset = control_clean[control_clean['cluster'] == i]
        alpha_value = 1 if i == j else 0# Highlight only the current cluster
        ax.scatter(subset['duration'], subset['pitch_whistles'], label=f'C{i + 1}', s = 1, color=purple_shades[i], alpha=alpha_value)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Duration')
    ax.set_ylabel('Pitch Whistles')
    ax.set_title(f'Cluster {cluster}')
    ax.spines[:].set_visible(False)  # Hide all spines

# Loop through regions and clusters for histograms (these will share axes)
for i, region in enumerate(regions):  # Columns (regions)
    total_counts_per_iteration = np.sum(data_list[i], axis=1)  # Sum across clusters

    for j, (cluster, color) in enumerate(zip(cluster_labels, purple_shades)):  # Rows (clusters)
        ax = axes[j + 1, i + 1]  # Shifted by 1 row and 1 column
        
        # Compute percentages
        cluster_percentages = (data_list[i][:, j] / total_counts_per_iteration) * 100
        
        # Plot histogram
        ax.hist(cluster_percentages, bins=20, color=color, edgecolor=color)

        # Compute p-value if applicable
        p_value = None
        if cluster in real_counts[i]:
            observed_value = (real_counts[i][cluster] / np.sum(real_counts[i])) * 100
            ax.axvline(observed_value, color='black', linestyle='-', linewidth=2)
            more_extreme = np.sum(cluster_percentages >= observed_value)
            p_value = (2 * min(more_extreme, len(cluster_percentages) - more_extreme)) / len(cluster_percentages)

        # Set title with p-value
        p_text = f" (p = {p_value:.3f})" if p_value is not None else ""
        ax.set_title(f"C{cluster} - R{region}{p_text}", fontsize=10)

        ax.set_box_aspect(1)  # Keep subplot square

# Ensure only histograms share x and y axes
for i in range(1, len(cluster_labels) + 1):
    for j in range(1, len(regions) + 1):
        if i == 1:
            axes[i, j].sharex(axes[1, 1])
        if j == 1:
            axes[i, j].sharey(axes[1, 1])

plt.tight_layout(rect=[0, 0, 1, 0.96])  
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Exp2_p-values_table_with_contours.pdf', transparent=True)
plt.show()

#------------------------------------------------------------------------------------------
#ADITIONAL ANALYSIS
#------------------------------------------------------------------------------------------
#-------------------------------
#-------------------------------
#-------------------------------
#FIRST EXPERIMENT
#-------------------------------
#-------------------------------
#-------------------------------

filtered_data_1['duration_error'] = filtered_data_1['d_median'] - filtered_data_1['d0']
filtered_data_1['pitch_error'] = filtered_data_1['f_median'] - filtered_data_1['f0']
filtered_data_1['abs_duration_error'] = np.abs(filtered_data_1['d_median'] - filtered_data_1['d0'])
filtered_data_1['abs_pitch_error'] = np.abs(filtered_data_1['f_median'] - filtered_data_1['f0'])

filtered_data_2['duration_error'] = filtered_data_2['d_median'] - filtered_data_2['d0']
filtered_data_2['pitch_error'] = filtered_data_2['f_median'] - filtered_data_2['f0']
filtered_data_2['abs_duration_error'] = np.abs(filtered_data_2['d_median'] - filtered_data_2['d0'])
filtered_data_2['abs_pitch_error'] = np.abs(filtered_data_2['f_median'] - filtered_data_2['f0'])


shuffled_responses_1['duration_error'] = shuffled_responses_1['d_median'] - shuffled_responses_1['d0']
shuffled_responses_1['pitch_error'] = shuffled_responses_1['f_median'] - shuffled_responses_1['f0']
shuffled_responses_1['abs_duration_error'] = np.abs(shuffled_responses_1['d_median'] - shuffled_responses_1['d0'])
shuffled_responses_1['abs_pitch_error'] = np.abs(shuffled_responses_1['f_median'] - shuffled_responses_1['f0'])

#-------------------------------
#DURATION AND PITCH ERROR IN THE FIRST EXPERIMENT
#-------------------------------
fig, axes = plt.subplots(2, len(CATEGORY), figsize=(1 * len(CATEGORY), 8), sharey='row')

for row_idx, (error_col, ylabel) in enumerate(zip(['duration_error', 'pitch_error'],
                                                  ['Duration Error (s)', 'Pitch Error (Hz)'])):
    all_densities = []

    # First pass: collect all densities to get global norm
    for category in CATEGORY:
        data = filtered_data_1[filtered_data_1['category'] == category][error_col].dropna()
        kde = gaussian_kde(data, bw_method = 'scott')
        density = kde(data)
        all_densities.extend(density)

    norm = plt.Normalize(min(all_densities), max(all_densities))

    for col_idx, category in enumerate(CATEGORY):
        data = filtered_data_1[filtered_data_1['category'] == category][error_col].dropna()
        kde = gaussian_kde(data, bw_method = 'scott')
        density = kde(data)
        colors = cm.cividis(norm(density))

        x_jitter = np.random.normal(loc=col_idx, scale=0.1, size=len(data))

        ax = axes[row_idx, col_idx]
        ax.scatter(x_jitter, data, color=colors, s=40, alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title(f"{category}")
        ax.set_xticks([])
        if col_idx == 0:
            ax.set_ylabel(ylabel)

        # Compute mode of KDE (most likely error)
        median_value = np.median(data)

        # Draw horizontal line at mode
        ax.axhline(y=median_value, color='red', linestyle='-', linewidth=2)
        print(category + ": " + str(median_value) )

    # Colorbar for each row
    cbar_ax = fig.add_axes([0.91, 0.55 if row_idx == 0 else 0.1, 0.015, 0.35])
    sm = cm.ScalarMappable(cmap='cividis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='KDE Density')

fig.subplots_adjust(right=0.88, hspace=0.3)
plt.show()
#-------------------------------
#PITCH AND/OR DURATION MATCH
#-------------------------------
# Define match conditions
filtered_data_1['pitch_match'] = filtered_data_1['abs_pitch_error'] < 100
filtered_data_1['duration_match'] = filtered_data_1['abs_duration_error'] < 0.055

# Define match type for each row
def get_match_type(row):
    pitch_match = row['abs_pitch_error'] < 100
    duration_match = row['abs_duration_error'] < 0.055
    if pitch_match and duration_match:
        return 'Both'
    elif pitch_match:
        return 'Pitch'
    elif duration_match:
        return 'Duration'
    else:
        return 'None'

filtered_data_1['match_type'] = filtered_data_1.apply(get_match_type, axis=1)

# Count total responses per category (including 'None')
total_per_category = filtered_data_1.groupby('category').size()

# Count match types per category
match_counts = (
    filtered_data_1
    .groupby(['category', 'match_type'])
    .size()
    .unstack(fill_value=0)
)

# Calculate percentages relative to total (including 'None')
match_percentages = match_counts.div(total_per_category, axis=0) * 100

# Drop 'None' only from plotting (not from percentage calc)
plot_data = match_percentages.drop(columns='None', errors='ignore')

# Ensure column order
ordered_columns = ['Pitch', 'Duration', 'Both']
plot_data = plot_data.reindex(columns=ordered_columns, fill_value=0)

colors = {
    'Both': '#3e4a89',         # Dark Blue
    'Pitch': '#577dbf',   # Mid Blue
    'Duration': '#a3c664',# Olive-Yellow
    'None': '#fdea45'          # Bright Yellow
}

# Create plot
fig, ax = plt.subplots(figsize=(4, 4))

bottom = np.zeros(len(plot_data))
bar_width = 0.8
x = np.arange(len(plot_data))

for i, col in enumerate(plot_data.columns):
    values = plot_data[col].values
    bars = ax.bar(x, values, bottom=bottom,
                  label=col, color=colors[col], edgecolor='black')
    
    for j, (value, bar) in enumerate(zip(values, bars)):
        label = f"{value:.1f}%"
        if value >= 5:
            # Large enough: label inside bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[j] + value / 2,
                label,
                ha='center', va='center', fontsize=10, color='white'
            )
        elif value > 0:
            # Too small: label outside with line
            y_pos = bottom[j] + value
            ax.plot([bar.get_x() + bar.get_width() / 2]*2, [y_pos, y_pos + 3],
                    color='black', lw=0.5)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + 3,
                label,
                ha='center', va='bottom', fontsize=9
            )
    bottom += values

# Final touches
ax.set_xticks(x)
ax.set_xticklabels(plot_data.index)
ax.set_ylabel('Percentage of All Responses')
ax.legend(bbox_to_anchor=(1, 1), frameon = False)
plt.tight_layout()
plt.show()
#----------
#SOUND SPACE OF EXP 1 RESPONSES
#---------
duration_match_only = filtered_data_1[filtered_data_1.match_type == 'Duration']
no_duration_match= filtered_data_1[filtered_data_1.match_type != 'Duration']

plt.figure(figsize=(3, 3))
plt.scatter(no_duration_match.d_median, no_duration_match.f_median, color = 'gray', alpha = 0.2, s = 10, label = "other")
plt.scatter(duration_match_only.d_median, duration_match_only.f_median, color = 'r', s = 10, label = 'duration match')
outside = sns.kdeplot(x=whistle_songs_control.d_median, 
                   y=whistle_songs_control.f_median, 
                   cmap="Reds_r", fill=False,
                   bw_adjust=0.7, levels=1, thresh=0.1)
plt.xlabel('median duration (s)')
plt.ylabel('median frequency (Hz)')
plt.show()

#-------------------------------
#ERROR ACROSS CATEGORIES
#-------------------------------
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
axes = axes.flatten()

# Define the same color mapping as before
match_colors = {
    'Both': colors['Both'],
    'Pitch': colors['Pitch'],
    'Duration': colors['Duration'],
    'None': colors['None']
}

# Loop over each category
for idx, category in enumerate(CATEGORY):
    ax = axes[idx]
    cat_data = filtered_data_1[filtered_data_1['category'] == category]
    
    # Plot points by match type so that the colors are consistent.
    for mt in ['Pitch', 'Duration']:
        subset = cat_data[cat_data['match_type'] == mt]
        ax.scatter(
            subset['duration_error'],   # x: signed duration error
            subset['pitch_error'],      # y: signed pitch error
            label=mt,
            color=match_colors[mt],
            edgecolor='none',
            alpha=0.7,
            s=20
        )
    
    # Draw threshold lines for visual reference.
    # Vertical lines at ±0.05 seconds (duration error thresholds)
    ax.axvline(x=0.055, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=-0.055, color='gray', linestyle='--', linewidth=1)
    # Horizontal lines at ±100 Hz (pitch error thresholds)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=-100, color='gray', linestyle='--', linewidth=1)
    
    ax.set_title(f"{category}")
    ax.set_xlabel("Duration Error (s)")
    ax.set_ylabel("Pitch Error (Hz)")

# Since each subplot uses the same labels for match types, create a single global legend.
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, title='Match Type', loc='center right', frameon=False)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to leave space for the legend
plt.show()


#----
#ERROR SPACE IN EXP 1
#----

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
axes = axes.flatten()

# Loop over each category
for idx, category in enumerate(CATEGORY):
    ax = axes[idx]
    cat_data = filtered_data_1[filtered_data_1['category'] == category]
    X = cat_data[['duration_error', 'pitch_error']].dropna().values
    ax.scatter(
        cat_data['duration_error'],   # x: signed duration error
        cat_data['pitch_error'],      # y: signed pitch error
        s=40
    )
    if len(X) >= 2:
        pca = PCA(n_components=2).fit(X)
        center = X.mean(axis=0)
        vector = pca.components_[0] * pca.explained_variance_ratio_[0]

        # Plot the PC1 direction vector
        ax.plot(
            [center[0] - vector[0], center[0] + vector[0]],
            [center[1] - vector[1], center[1] + vector[1]],
            color='red', linestyle='--', label='PC1 direction'
            )

    angle_rad = np.arctan2(pca.components_[0,1], pca.components_[0,0])
    angle_deg = np.degrees(angle_rad)
    print(f"{category} trade-off angle: {angle_deg:.1f}°")
    # Draw threshold lines for visual reference.
    # Vertical lines at ±0.05 seconds (duration error thresholds)
    ax.axvline(x=0.055, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=-0.055, color='gray', linestyle='--', linewidth=1)
    # Horizontal lines at ±100 Hz (pitch error thresholds)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=-100, color='gray', linestyle='--', linewidth=1)
    
    ax.set_title(f"{category}")
    ax.set_xlabel("Duration Error (s)")
    ax.set_ylabel("Pitch Error (Hz)")

# Since each subplot uses the same labels for match types, create a single global legend.
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, title='Match Type', loc='center right', frameon=False)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to leave space for the legend
plt.show()

#-------------------------------
#EXAMPLE OF SHUFFLED DATA RESULTS
#-------------------------------
fig, axes = plt.subplots(2, len(CATEGORY), figsize=(1 * len(CATEGORY), 8), sharey='row')

for row_idx, (error_col, ylabel) in enumerate(zip(['duration_error', 'pitch_error'],
                                                  ['Duration Error (s)', 'Pitch Error (Hz)'])):
    all_densities = []

    # First pass: collect all densities to get global norm
    for category in CATEGORY:
        data = shuffled_responses_1[shuffled_responses_1['category'] == category][error_col].dropna()
        kde = gaussian_kde(data, bw_method = 'scott')
        density = kde(data)
        all_densities.extend(density)

    norm = plt.Normalize(min(all_densities), max(all_densities))

    for col_idx, category in enumerate(CATEGORY):
        data = shuffled_responses_1[shuffled_responses_1['category'] == category][error_col].dropna()
        kde = gaussian_kde(data, bw_method = 'scott')
        density = kde(data)
        colors = cm.cividis(norm(density))

        x_jitter = np.random.normal(loc=col_idx, scale=0.1, size=len(data))

        ax = axes[row_idx, col_idx]
        ax.scatter(x_jitter, data, color=colors, s=40, alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title(f"{category}")
        ax.set_xticks([])
        if col_idx == 0:
            ax.set_ylabel(ylabel)

        # Compute mode of KDE (most likely error)
        median_value = np.median(data)

        # Draw horizontal line at mode
        ax.axhline(y=median_value, color='red', linestyle='-', linewidth=2)
        print(category + ": " + str(median_value) )

    # Colorbar for each row
    cbar_ax = fig.add_axes([0.91, 0.55 if row_idx == 0 else 0.1, 0.015, 0.35])
    sm = cm.ScalarMappable(cmap='cividis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='KDE Density')

fig.subplots_adjust(right=0.88, hspace=0.3)
plt.show()

#-------------------------------
#BOOTSTRAPING
#-------------------------------
num_shufflings = 10000

def plot_random_errors(num_iterations):
    median_null = {category: [] for category in CATEGORY}
    for iteration in tqdm(range(num_iterations), desc="Shuffling Progress"):
        shuffled_data = shuffle_data(filtered_data_1) 
        shuffled_data['duration_error'] = shuffled_data['d_median'] - shuffled_data['d0']
        
        for category in CATEGORY:
            data = shuffled_data[shuffled_data['category'] == category]['duration_error'].dropna()
            if len(data) > 1:  # KDE needs at least two points
                median_value = np.median(data)
                median_null[category].append(median_value)
            else:
                median_null[category].append(np.nan)  # Fill with NaN or skip
    return median_null

median_null = plot_random_errors(num_shufflings)

fig, axes = plt.subplots(1, len(CATEGORY), figsize=(1 * len(CATEGORY), 3), sharey=True)

for col_idx, category in enumerate(CATEGORY):
    ax = axes[col_idx]

    # Null modes from shuffling
    null_medians = median_null[category]
    x_jitter = np.random.normal(loc=col_idx, scale=0.1, size=len(null_medians))
    ax.scatter(x_jitter, null_medians, color='gray', s=10, alpha=0.5)

    # Real mode
    data = filtered_data_1[filtered_data_1['category'] == category]['duration_error'].dropna()
    median_real = np.median(data)

    ax.axhline(y=median_real, color='red', linestyle='-', linewidth=2)

    # Formatting
    ax.set_title(category)
    ax.set_xticks([])
    if col_idx == 0:
        ax.set_ylabel('Duration Error (s)')

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, len(CATEGORY), figsize=(1 * len(CATEGORY), 4), sharey=True)

for col_idx, category in enumerate(CATEGORY):
    ax = axes[col_idx]

    # Null medians from shuffling
    null_medians = median_null[category]
    x_jitter = np.random.normal(loc=col_idx, scale=0.1, size=len(null_medians))
    ax.scatter(x_jitter, null_medians, color='gray', s=10, alpha=0.5)

    # Real median
    data = filtered_data_1[filtered_data_1['category'] == category]['duration_error'].dropna()
    median_real = np.median(data)
    print(category, ": ", median_real)
    # Red line: real median
    ax.axhline(y=median_real, color='red', linestyle='-', linewidth=2)

    # One-sided p-value
    if median_real > 0:
        p_val = 1 - np.mean(np.array(null_medians) >= median_real)
    else:
        p_val = 1 - np.mean(np.array(null_medians) <= median_real)

    # Print p-value in console
    print(f"{category}: real median = {median_real:.4f}, one-sided p = {p_val:.4f}")

    # Determine significance stars
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = ''

    # Annotate p-value and stars
    ax.text(0.5, 0.95, f"p = {p_val:.4f}\n{stars}", transform=ax.transAxes,
            ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    # Formatting
    ax.set_title(category)
    ax.set_xticks([])
    if col_idx == 0:
        ax.set_ylabel('Duration Error (s)')

fig.tight_layout()
plt.show()

# -------------------------------
# BOOTSTRAPPING (ABSOLUTE ERROR)
# -------------------------------
num_shufflings = 10000

def plot_random_errors_abs(num_iterations):
    median_null = {category: [] for category in CATEGORY}
    for iteration in tqdm(range(num_iterations), desc="Shuffling Progress"):
        shuffled_data = shuffle_data(filtered_data_1)
        shuffled_data['duration_error'] = np.abs(shuffled_data['d_median'] - shuffled_data['d0'])

        for category in CATEGORY:
            data = shuffled_data[shuffled_data['category'] == category]['duration_error'].dropna()
            if len(data) > 1:  # Ensure KDE/statistics can run
                median_value = np.median(data)
                median_null[category].append(median_value)
            else:
                median_null[category].append(np.nan)
    return median_null

median_null = plot_random_errors_abs(num_shufflings)

# -------------------------------
# PLOT + P-VALUES
# -------------------------------
fig, axes = plt.subplots(1, len(CATEGORY), figsize=(1 * len(CATEGORY), 4), sharey=True)

for col_idx, category in enumerate(CATEGORY):
    ax = axes[col_idx]

    # Null medians from shuffling
    null_medians = median_null[category]
    x_jitter = np.random.normal(loc=col_idx, scale=0.1, size=len(null_medians))
    ax.scatter(x_jitter, null_medians, color='gray', s=20, alpha=0.2)

    # Real median (absolute error)
    data = filtered_data_1[filtered_data_1['category'] == category]
    real_error = np.abs(data['d_median'] - data['d0']).dropna()
    median_real = np.median(real_error)
    print(category, ": ", median_real)
    # Red line: real median
    ax.axhline(y=median_real, color='red', linestyle='-', linewidth=1)

    # One-sided p-value: how many null medians >= real median
    p_val = 1 - np.mean(np.array(null_medians) >= median_real)

    # Print to console
    print(f"{category}: real median (abs error) = {median_real:.4f}, one-sided p = {p_val:.4f}")

    # Significance stars
    if p_val < 0.001:
        stars = '***'
    elif p_val < 0.01:
        stars = '**'
    elif p_val < 0.05:
        stars = '*'
    else:
        stars = ''

    # Annotate
    ax.text(0.5, -0.2, f"p = {p_val:.4f}\n{stars}", transform=ax.transAxes,
        ha='center', va='top', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Format
    ax.set_title(category)
    ax.set_xticks([])
    if col_idx == 0:
        ax.set_ylabel('|Duration Error| (s)')

fig.tight_layout()
plt.show()
#-------------------------------
#-------------------------------
#-------------------------------
#SECOND EXPERIMENT
#-------------------------------
#-------------------------------
#-------------------------------
plt.style.use('default')

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

regions = {
    'A': (birds_songs_info[zone_A_positive & bird_filter_2], playback_exp2_d_A, playback_exp2_f_A),
    'B': (birds_songs_info[zone_B_positive & bird_filter_2], playback_exp2_d_B, playback_exp2_f_B),
    'C': (birds_songs_info[zone_C_positive & bird_filter_2], playback_exp2_d_C, playback_exp2_f_C)
}

# Storage for results
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

# Common axis labels
fig.text(0.5, 0.04, "Error Duration (s)", ha='center', fontsize=12)
fig.text(0.06, 0.5, "Error Pitch (kHz)", va='center', rotation='vertical', fontsize=12)

mpl.rcParams['pdf.fonttype'] = 42
# Save and display the plot
plt.savefig('Plots/Exp_2_errors.pdf', transparent=True)
plt.show()
    
#------
#TRADE OFF
#-----

# Thresholds for max error tolerance
T_d = 0.2  # seconds
T_f = 500  # Hz
epsilon = 1e-8  # to avoid divide-by-zero

regions_masks = {
    'A': zone_A_positive & bird_filter_2,
    'B': zone_B_positive & bird_filter_2,
    'C': zone_C_positive & bird_filter_2
}

# Containers to store computed values
regions = []
duration_alignments = []
pitch_alignments = []
tradeoff_indices = []

for region, mask in regions_masks.items():
    data = birds_songs_info[mask]
    
    # Absolute errors
    duration_error = np.abs(data.d_median - data.d0)
    pitch_error = np.abs(data.f_median - data.f0)
    
    # Alignment scores, clipped between 0 and 1
    duration_alignment = np.clip(1 - duration_error / T_d, 0, 1)
    pitch_alignment = np.clip(1 - pitch_error / T_f, 0, 1)
    
    # Mean alignment per region
    A_d = duration_alignment.mean()
    A_f = pitch_alignment.mean()
    
    # Trade-off index
    tradeoff_index = 1 - np.abs(A_d - A_f) / (A_d + A_f + epsilon)
    
    print(f"Region {region}:")
    print(f"  Mean duration alignment: {A_d:.2f}")
    print(f"  Mean pitch alignment: {A_f:.2f}")
    print(f"  Trade-off index: {tradeoff_index:.2f}")
    print()
    
    # Store for plotting
    regions.append(region)
    duration_alignments.append(A_d)
    pitch_alignments.append(A_f)
    tradeoff_indices.append(tradeoff_index)

# Convert lists to numpy arrays for plotting
duration_alignments = np.array(duration_alignments)
pitch_alignments = np.array(pitch_alignments)
tradeoff_indices = np.array(tradeoff_indices)

# Calculate angles clipped between 0 and pi/2 (quarter circle)
angles = np.arctan2(pitch_alignments, duration_alignments)
angles = np.clip(angles, 0, np.pi/2)

radii = tradeoff_indices

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4,4))

# Define different marker styles for each region (no fill)
markers = ['o', 's', '^']  # circle, square, triangle
sizes = 100  # marker size

for i, region in enumerate(regions):
    ax.scatter(angles[i], radii[i], 
               s=sizes, 
               marker=markers[i], 
               edgecolor='k', 
               facecolors='none',  # no fill
               linewidth=2,
               alpha=0.8, 
               label=f'Region {region}')

# Adjust label positions manually to reduce overlap
label_offsets = [0.06, 0.06, 0.06]
angle_offsets = [0, 0.05, -0.05]  # small angular shifts for spacing

for i, region in enumerate(regions):
    ax.text(angles[i] + angle_offsets[i], radii[i] + label_offsets[i], region,
            ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

ax.set_rlim(0, 1)
ax.set_rlabel_position(-22.5)

ax.set_thetamin(0)
ax.set_thetamax(90)

ax.set_xticks([0, np.pi/4, np.pi/2])
ax.set_xticklabels(['Duration\n(0°)', 'Balanced\n(45°)', 'Pitch\n(90°)'])

ax.grid(True)

plt.show()

T_d = 0.2  # seconds
T_f = 2000  # Hz

fig, axs = plt.subplots(1, 3, figsize=(6, 2), sharex=True, sharey=True)

for i, region in enumerate(regions):
    ax = axs[i]
    
    # Get region data again
    mask = regions_masks[region]
    data = birds_songs_info[mask]
    
    # Compute alignment scores
    duration_error = np.abs(data.d_median - data.d0)
    pitch_error = np.abs(data.f_median - data.f0)
    
    duration_alignment = np.clip(1 - duration_error / T_d, 0, 1)
    pitch_alignment = np.clip(1 - pitch_error / T_f, 0, 1)
    
    # Scatter plot
    ax.scatter(duration_alignment, pitch_alignment, 
               s=50, alpha=0.6, edgecolor='k', facecolors='none')
    
    # Identity line
    ax.plot([0, 1], [0, 1], 'r--', lw=1)
    
    ax.set_title(f'Region {region}', fontsize=12)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')  # square axes
    if i == 0:
        ax.set_ylabel('Pitch alignment')
    ax.set_xlabel('Duration alignment')

plt.tight_layout()
plt.show()



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

# Print key values
print(f"mu_d = {mu_d:.4f}, beta_d = {beta_d:.4f}, duration_sim_at_zero = {duration_sim_at_zero:.4f}")
print(f"mu_f = {mu_f:.4f}, beta_f = {beta_f:.4f}, pitch_sim_at_zero = {pitch_sim_at_zero:.4f}")

# Setup figure
fig = plt.figure(figsize=(8.2, 4))
gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.3, hspace=0.4)
axs = np.empty((2, 3), dtype=object)

for i in range(3):
    axs[0, i] = fig.add_subplot(gs[0, i])
    axs[1, i] = fig.add_subplot(gs[1, i])

last_im = None

for i, region in enumerate(regions):
    mask = regions_masks[region]
    data = birds_songs_info[mask]

    duration_error = np.abs(data.d_median - data.d0)
    pitch_error = np.abs(data.f_median - data.f0)

    duration_similarity = sigmoid_similarity(duration_error, mu_d, beta_d)
    pitch_similarity = sigmoid_similarity(pitch_error, mu_f, beta_f)

    # -------- Scatter plot --------
    ax = axs[0, i]
    ax.scatter(duration_similarity, pitch_similarity, s=10, color='k')
    ax.plot([0, 1], [0, 1], 'r--', lw=1)
    ax.axvline(duration_sim_at_zero, color='green', linestyle='--', lw=1)
    ax.axhline(pitch_sim_at_zero, color='green', linestyle='--', lw=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('Duration similarity')
    if i == 0:
        ax.set_ylabel('Pitch similarity')
    ax.set_title(f'Region {region}', fontsize=12)

    # -------- Density plot --------
    ax = axs[1, i]
    xy = np.vstack([duration_similarity, pitch_similarity])
    kde = gaussian_kde(xy, bw_method=0.4)

    xgrid, ygrid = np.mgrid[0:1:100j, 0:1:100j]
    grid_coords = np.vstack([xgrid.ravel(), ygrid.ravel()])

    z = kde(grid_coords).reshape(100, 100)

    dx = 1 / 100
    dy = 1 / 100
    bin_area = dx * dy
    z_prob = z * bin_area

    im = ax.imshow(z_prob.T, origin='lower', extent=[0, 1, 0, 1],
                   cmap='Blues', aspect='equal', vmin=0, vmax=np.max(z_prob))
    last_im = im

    ax.plot([0, 1], [0, 1], 'r--', lw=1)
    ax.axvline(duration_sim_at_zero, color='green', linestyle='--', lw=1)
    ax.axhline(pitch_sim_at_zero, color='green', linestyle='--', lw=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Duration similarity')
    if i == 0:
        ax.set_ylabel('Pitch similarity')

# Colorbar
cbar_ax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(last_im, cax=cbar_ax)
cbar.set_label('Probability per 2D bin')

mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/Exp2_similarities.pdf', transparent=True)
plt.show()

# Setup figure and GridSpec with 4 columns (last one for colorbar)
fig = plt.figure(figsize=(8.2, 2.8))
gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
axs = []

last_im = None  # to hold the last density image for colorbar

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

    # Reference lines
    ax.plot([0, 1], [0, 1], 'r--', lw=1)
    ax.axvline(0.5, color='k', linestyle='--', lw=1)
    ax.axhline(0.5, color='k', linestyle='--', lw=1)

    # Axes settings
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_box_aspect(1)  # Ensure square plotting area

    ticks = np.linspace(0, 1, 5)  # Same ticks for both axes
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_title(f'Region {region}', fontsize=12)
    ax.set_xlabel('Duration similarity')
    if i == 0:
        ax.set_ylabel('Pitch similarity')

# Add colorbar
cbar_ax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(last_im, cax=cbar_ax)
cbar.set_label('Probability per 2D bin')

# Adjust layout without distorting aspect ratios
fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.15, wspace=0.3)
fig.align_labels()

mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/Exp2_similarities_with_filter.pdf', transparent=True)
plt.show()


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

    # Set limits
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 9)

    # Force square axis box
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


fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True, sharex=True)

for i, region in enumerate(regions):
    # Filter data
    mask = regions_masks[region]
    data = birds_songs_info[mask]

    # Compute similarities
    duration_error = np.abs(data.d_median - data.d0)
    pitch_error = np.abs(data.f_median - data.f0)
    duration_similarity = sigmoid_similarity(duration_error, mu_d, beta_d)
    pitch_similarity = sigmoid_similarity(pitch_error, mu_f, beta_f)

    # Signed distance to y = x
    signed_dist = (pitch_similarity - duration_similarity) / np.sqrt(2)

    # Wilcoxon signed-rank test (test if median of signed_dist differs from 0)
    stat, p_value = wilcoxon(signed_dist)
    print(f"Region {region}: Wilcoxon test statistic = {stat:.2f}, p-value = {p_value:.4f}")

    # Compute histogram manually to get probabilities
    counts, bins = np.histogram(signed_dist, bins=20)
    probs = counts / counts.sum()  # Normalize to probability
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot
    ax = axs[i]
    ax.bar(bin_centers, probs, width=np.diff(bins), color='black', align='center', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.set_title(f'Region {region}\nWilcoxon p = {p_value:.4f}')
    ax.set_xlabel('Signed distance to y = x')
    if i == 0:
        ax.set_ylabel('Probability')

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
# Save and display the plot
plt.savefig('Plots/Exp2_similarities_distances.pdf', transparent=True)
plt.show()


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
#-------------------------------
#BOOTSTRAP FOR THE ERROR IN DURATION IN REGION A EXP 2
#-------------------------------
# Step 1: Real data median for zone A
zone_A_real = filtered_data_2[filtered_data_2.d0 < 0.2]
real_median_abs_error = np.median(np.abs(zone_A_real.d_median - zone_A_real.d0))

# Step 2: Shuffle and compute medians
n_iterations = 10000
shuffled_medians = []


for _ in range(n_iterations):
    shuffled = shuffle_data(filtered_data_2)
    zone_A_shuffled = shuffled[shuffled.d0 < 0.2]
    median_abs_error = np.median(np.abs(zone_A_shuffled.d_median - zone_A_shuffled.d0))
    shuffled_medians.append(median_abs_error)

# Convert list to NumPy array for efficiency
shuffled_medians = np.array(shuffled_medians)

# One-sided p-value: proportion of shuffled medians <= real median
p_value = np.mean(shuffled_medians <= real_median_abs_error)

print(f"Real median absolute duration error (Zone A): {real_median_abs_error:.4f} s")
print(f"One-sided p-value: {p_value:.4f}")

# Step 3: Plot
plt.figure(figsize=(5, 3))
plt.hist(shuffled_medians, bins=30, color='gray', edgecolor='gray')
plt.axvline(real_median_abs_error, color='red', linewidth=2, label='Real Median')
plt.xlabel('Median Absolute Duration Error (s)')
plt.ylabel('Count')
plt.title('Zone A')
plt.legend(frameon = False)
plt.tight_layout()
plt.show()

#-------------------------------
#PITCH MATCH IN REGION B AND C
#-------------------------------
# Theoretical playback frequencies
true_f0s = np.array([1000, 2000, 3000, 6000, 7000, 8000])

# Setup figure
fig, ax = plt.subplots(figsize=(3, 3))
cmap = cm.get_cmap('cividis')

# Go through regions B and C
for region_label in ['B', 'C']:
    data = regions[region_label][0].copy()

    # Snap each f0 to the nearest theoretical value
    data['f0_snapped'] = data['f0'].apply(lambda x: true_f0s[np.argmin(np.abs(true_f0s - x))])

    for f0_val in true_f0s:
        group = data[data.f0_snapped == f0_val]
        y_vals = group.f_median.values

        if len(y_vals) > 1:
            kde = gaussian_kde(y_vals, bw_method='scott')
            densities = kde(y_vals)
            # Get mode of KDE
            y_grid = np.linspace(y_vals.min(), y_vals.max(), 500)
            mode_y = y_grid[np.argmax(kde(y_grid))]
        elif len(y_vals) == 1:
            densities = np.array([1.0])
            mode_y = y_vals[0]
        else:
            continue  # skip if no points

        norm = Normalize(vmin=np.min(densities), vmax=np.max(densities))
        colors = cmap(norm(densities))

        x_jitter = np.random.uniform(-200, 200, size=len(group))
        x_vals = np.full(len(group), f0_val) + x_jitter

        ax.scatter(x_vals, y_vals, c=colors, s=50, alpha=0.8, edgecolor='k', linewidth=0.2)

        # Draw one red line for KDE mode
        ax.hlines(mode_y, f0_val - 250, f0_val + 250, color='red', linewidth=2)
        print("playback: ", f0_val, "Mode response: ", mode_y)
# Labels and style
ax.axhline(y = 3000, color = 'gray', linestyle='--')
ax.axhline(y = 6000, color = 'gray', linestyle='--')
ax.set_xlabel("Playback Frequency (Hz)", fontsize=10)
ax.set_ylabel("Response Median Frequency (Hz)", fontsize=10)
plt.tight_layout()
plt.show()

#-------------------------------
#BOOTSTRAP FOR THE ERROR IN PITCH IN REGION B AND C EXP 2
#-------------------------------

# Step 2: Shuffle and compute medians
n_iterations = 10000
# Dictionary to store modes per f0_val
shuffled_modes = {f0_val: [] for f0_val in true_f0s}


for _ in range(n_iterations):
    shuffled = shuffle_data(filtered_data_2)

    # Define regions B and C in shuffled data
    zone_B_shuffled = shuffled[(shuffled.f0 > 5000) & (shuffled.d0 >= 0.5)]
    zone_C_shuffled = shuffled[shuffled.f0 < 4000]

    regions_shuffled = {
        "B": zone_B_shuffled,
        "C": zone_C_shuffled
    }

    for region_label in ['B', 'C']:
        data = regions_shuffled[region_label].copy()

        # Snap each f0 to the nearest theoretical playback frequency
        data['f0_snapped'] = data['f0'].apply(lambda x: true_f0s[np.argmin(np.abs(true_f0s - x))])

        for f0_val in true_f0s:
            group = data[data.f0_snapped == f0_val]
            y_vals = group.f_median.values

            if len(y_vals) > 1:
                kde = gaussian_kde(y_vals, bw_method='scott')
                y_grid = np.linspace(y_vals.min(), y_vals.max(), 500)
                mode_y = y_grid[np.argmax(kde(y_grid))]
                shuffled_modes[f0_val].append(mode_y)
            elif len(y_vals) == 1:
                # Just use the single value as the "mode"
                shuffled_modes[f0_val].append(y_vals[0])
            # else: do nothing if len == 0
# Step 3: Get real modes from actual data
real_modes = {}

for region_label in ['B', 'C']:
    data = regions[region_label][0].copy()
    data['f0_snapped'] = data['f0'].apply(lambda x: true_f0s[np.argmin(np.abs(true_f0s - x))])

    for f0_val in true_f0s:
        group = data[data.f0_snapped == f0_val]
        y_vals = group.f_median.values

        if len(y_vals) > 1:
            kde = gaussian_kde(y_vals, bw_method='scott')
            y_grid = np.linspace(y_vals.min(), y_vals.max(), 500)
            mode_y = y_grid[np.argmax(kde(y_grid))]
        elif len(y_vals) == 1:
            mode_y = y_vals[0]
        else:
            continue

        real_modes[f0_val] = mode_y

# Step 4: Plot histograms and compute p-values
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
axes = axes.ravel()

for i, f0_val in enumerate(true_f0s):
    modes = np.array(shuffled_modes[f0_val])
    real_mode = real_modes.get(f0_val, None)

    ax = axes[i]
    ax.hist(modes, bins=30, color='gray', edgecolor='black')

    if real_mode is not None:
        ax.axvline(real_mode, color='red', linewidth=2, label='Real Mode')

        # Two-sided p-value
        median_mode = np.median(modes)
        abs_diff_real = np.abs(real_mode - median_mode)
        abs_diff_shuffled = np.abs(modes - median_mode)
        p_val = np.mean(abs_diff_shuffled >= abs_diff_real)

        ax.set_title(f'{f0_val} Hz\np = {p_val:.4f}')
    else:
        ax.set_title(f'{f0_val} Hz\n(No real data)')

    ax.set_xlabel("Mode of Response Frequencies (Hz)")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()

#----
#DURATION IN REGION C
#-----
data_zone_C = birds_songs_info[zone_C_positive & bird_filter_2]
plt.style.use('default')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 3), gridspec_kw={'height_ratios': [1, 0.4]})
# KDE plot (First Subplot)
ax1 = axes[0]
kde = sns.kdeplot(data_zone_C.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.4, ax=ax1)
# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()  # Get the latest plotted KDE line

# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)

print(' Mean: ', np.mean(control.duration), " STD: ", np.std(control.duration))
print(' Median: ', np.median(control.duration), " MAD: ", np.median(np.abs(control.duration - np.median(control.duration))))
ax1.set_ylabel('Normalized counts', fontsize=10)
x_values = kde.lines[0].get_xdata()
y_values = kde.lines[0].get_ydata()

ax1.set_xlim([0, 0.9])
# Scatter plot with jittered data (Second Subplot)
ax2 = axes[1]
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(data_zone_C.d_median))
ax2.scatter(data_zone_C.d_median, np.zeros_like(data_zone_C.d_median) + y_jittered, 
                color='black', alpha=0.2, s=4)
ax2.set_xlabel('Whistle duration (s)')
ax2.set_yticks([])
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.show()

# Step 1: Real data median for zone C
zone_C_real = birds_songs_info[zone_C_positive & bird_filter_2]
real_median_C = np.median(zone_C_real.d_median)

# Step 2: Shuffle and compute medians
n_iterations = 10000
shuffled_medians_C = []

for _ in tqdm(range(n_iterations)):
    shuffled = shuffle_data(filtered_data_2)
    zone_C_shuffled = shuffled[shuffled.f0 < 4000]
    median_d = np.median(zone_C_shuffled.d_median)
    shuffled_medians_C.append(median_d)

# Convert list to NumPy array for efficiency
shuffled_medians_C = np.array(shuffled_medians_C)

# One-sided p-value: proportion of shuffled medians <= real median
p_value = np.mean(shuffled_medians_C >= real_median_C)

print(f"Real median duration (Zone C): {real_median_C:.4f} s")
print(f"One-sided p-value: {p_value:.4f}")

# Step 3: Plot
plt.figure(figsize=(5, 3))
plt.hist(shuffled_medians_C, bins=30, color='gray', edgecolor='gray')
plt.axvline(real_median_C, color='red', linewidth=2, label='Real Median')
plt.xlabel('Median Duration (s)')
plt.ylabel('Count')
plt.title('Zone C')
plt.legend(frameon = False)
plt.tight_layout()
plt.show()

#same but excluding pitch match
zone_C_real = birds_songs_info[zone_C_positive & bird_filter_2]
zone_C_real = zone_C_real[np.abs(zone_C_real.f_median - zone_C_real.f0) > 100]

# Compute real median duration
real_median_C = np.median(zone_C_real.d_median)

# Step 2: Shuffle and compute medians
n_iterations = 10000
shuffled_medians_C = []

for _ in tqdm(range(n_iterations)):
    shuffled = shuffle_data(filtered_data_2)
    zone_C_shuffled = shuffled[shuffled.f0 < 4000]
    zone_C_shuffled = zone_C_shuffled[np.abs(zone_C_shuffled.f_median - zone_C_shuffled.f0) > 100]

    if len(zone_C_shuffled) > 0:
        median_d = np.median(zone_C_shuffled.d_median)
        shuffled_medians_C.append(median_d)

# Convert to NumPy array
shuffled_medians_C = np.array(shuffled_medians_C)

# One-sided p-value: proportion of shuffled medians >= real median
p_value = np.mean(shuffled_medians_C >= real_median_C)

# Print results
print(f"Real median duration (Zone C): {real_median_C:.4f} s")
print(f"One-sided p-value (excluding ±100 Hz): {p_value:.4f}")

# Step 3: Plot
plt.figure(figsize=(5, 3))
plt.hist(shuffled_medians_C, bins=30, color='gray', edgecolor='gray')
plt.axvline(real_median_C, color='red', linewidth=2, label='Real Median')
plt.xlabel('Median Duration (s)')
plt.ylabel('Count')
plt.title('Zone C (Excluding ±100 Hz Matches)')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
#---
#RESULTS WITHOUT THE PITCH MATCH
#---
plt.style.use('default')
fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(5, 9), sharex=True, 
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
no_pitch_match= filtered_data_1[filtered_data_1.match_type != 'Pitch']
kde_control = gaussian_kde(whistle_songs_control.d_median, bw_method=0.5)
cmap = plt.cm.RdGy_r
norm = plt.Normalize(vmin=0, vmax=2)

for idx, (category, color) in enumerate(zip(CATEGORY, COLORS)):
    data = no_pitch_match[no_pitch_match["category"] == category]
    print(category, len(data.d_median))
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
    
    print(category, ' Mean: ', np.mean(data.d_median), " STD: ", np.std(data.d_median))
    print(category, ' Median: ', np.median(data.d_median), " MAD: ", np.median(np.abs(data.d_median - np.median(data.d_median))))
    
    axes[2 * (idx + 1) + 1].vlines(median_value, -0.08, -0.2, color='black', lw=2, linestyle='-', zorder=6)
    
    axes[2 * (idx + 1) + 1].set_yticks([])
    axes[2 * (idx + 1) + 1].set_ylabel(category)

# Add colorbar for KDE ratio visualization
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(scatter, cax=cbar_ax)
cbar.set_label("KDE Ratio")

plt.tight_layout(rect=[0, 0, 0.9, 1])
mpl.rcParams['pdf.fonttype'] = 42
plt.show()

median_real = plot_responses(no_pitch_match, 'd_median', 0.6)
num_shufflings = 10000
def plot_random_shufflings(num_iterations, parameter):
    median_null = {category: [] for category in CATEGORY}
    for iteration in tqdm(range(num_iterations), desc="Shuffling Progress"):
        shuffled_data = shuffle_data(no_pitch_match) 
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
plt.show()

#--------------
#error position
#-----------

# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# Loop through each region and plot
for i, (cat, color) in enumerate(zip(CATEGORY, COLORS)):
    ax = axes[i]
    region_data = filtered_data_1[filtered_data_1.category == cat]
    
    # All points in black
    ax.scatter(filtered_data_1.abs_duration_error, filtered_data_1.ix_median, c='lightgray', alpha=0.3)
    
    # Highlight region in its color
    ax.scatter(region_data.abs_duration_error, region_data.ix_median, c=color, alpha=0.7)
    
    ax.axvline(x=0.055, color='r', linestyle='--')
    ax.set_title(cat)
    ax.set_xlabel('Abs duration error (s)')
    if i == 0:
        ax.set_ylabel('syllable for the median')
    ax.set_xlim(0, filtered_data_1.abs_duration_error.max())

plt.tight_layout()
plt.show()

# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# Loop through each region and plot
for i, (cat, color) in enumerate(zip(CATEGORY, COLORS)):
    ax = axes[i]
    region_data = filtered_data_1[filtered_data_1.category == cat]
    
    # All points in black
    ax.scatter(filtered_data_1.abs_duration_error, filtered_data_1.ix_best, c='lightgray', alpha=0.3)
    
    # Highlight region in its color
    ax.scatter(region_data.abs_duration_error, region_data.ix_best, c=color, alpha=0.7)
    
    ax.axvline(x=0.055, color='r', linestyle='--')
    ax.set_title(cat)
    ax.set_xlabel('Abs duration error (s)')
    if i == 0:
        ax.set_ylabel('Closest syllable in duration')
    ax.set_xlim(0, filtered_data_1.abs_duration_error.max())

plt.tight_layout()
plt.show()

plt.figure(figsize=(3, 3))
plt.scatter(filtered_data_2.abs_duration_error, filtered_data_2.ix_median, c = 'k', alpha = 0.5);
plt.xlabel('Absolute duration error (s)')
plt.ylabel('Syllable for the median')
plt.axvline(x = 0.055, c = 'r', linestyle = '--' )
plt.title("EXP 2")
plt.show()

# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# Loop through each region and plot
for i, (cat, color) in enumerate(zip(CATEGORY, COLORS)):
    ax = axes[i]
    region_data = filtered_data_1[filtered_data_1.category == cat]
    
    # All points in black
    ax.scatter(filtered_data_1.abs_duration_error, filtered_data_1.loc_median, c='lightgray', alpha=0.3)
    
    # Highlight region in its color
    ax.scatter(region_data.abs_duration_error, region_data.loc_median, c=color, alpha=0.7)
    
    ax.axvline(x=0.055, color='r', linestyle='--')
    ax.set_title(cat)
    ax.set_xlabel('Abs duration error (s)')
    if i == 0:
        ax.set_ylabel('Normalized position\nof the median')
    ax.set_xlim(0, filtered_data_1.abs_duration_error.max())
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# Loop through each region and plot
for i, (cat, color) in enumerate(zip(CATEGORY, COLORS)):
    ax = axes[i]
    region_data = filtered_data_1[filtered_data_1.category == cat]
    
    # All points in black
    ax.scatter(filtered_data_1.abs_duration_error, filtered_data_1.loc_best, c='lightgray', alpha=0.3)
    
    # Highlight region in its color
    ax.scatter(region_data.abs_duration_error, region_data.loc_best, c=color, alpha=0.7)
    
    ax.axvline(x=0.055, color='r', linestyle='--')
    ax.set_title(cat)
    ax.set_xlabel('Abs duration error (s)')
    if i == 0:
        ax.set_ylabel('Normalized position for the best')
    ax.set_xlim(0, filtered_data_1.abs_duration_error.max())
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# Loop through each region and plot
for i, (cat, color) in enumerate(zip(CATEGORY, COLORS)):
    ax = axes[i]
    region_data = filtered_data_1[filtered_data_1.category == cat]
    
    # All points in light gray
    ax.scatter(filtered_data_1.abs_duration_error, filtered_data_1.loc_best, c='lightgray', alpha=0.3)
    
    # Highlight region in its color
    ax.scatter(region_data.abs_duration_error, region_data.loc_best, c=color, alpha=0.7)
    
    # Add red threshold line
    ax.axvline(x=0.055, color='r', linestyle='--')
    
    # Compute and plot center of mass (mean)
    mean_x = region_data.abs_duration_error.mean()
    mean_y = region_data.loc_best.mean()
    ax.scatter(mean_x, mean_y, c='black', s=300, marker='o', edgecolors='white', zorder=10)  # Big black dot
    print(mean_y)
    # Labels and aesthetics
    ax.set_title(cat)
    ax.set_xlabel('Abs duration error (s)')
    if i == 0:
        ax.set_ylabel('Normalized position for the best')
    ax.set_xlim(0, filtered_data_1.abs_duration_error.max())
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# Loop through each region and plot
for i, (cat, color) in enumerate(zip(CATEGORY, COLORS)):
    ax = axes[i]
    region_data = filtered_data_1[filtered_data_1.category == cat]
    
    # All points in light gray
    ax.scatter(filtered_data_1.abs_duration_error, filtered_data_1.loc_median, c='lightgray', alpha=0.3)
    
    # Highlight region in its color
    ax.scatter(region_data.abs_duration_error, region_data.loc_median, c=color, alpha=0.7)
    
    # Add red threshold line
    ax.axvline(x=0.055, color='r', linestyle='--')
    
    # Compute and plot center of mass (mean)
    mean_x = region_data.abs_duration_error.mean()
    mean_y = region_data.loc_median.mean()
    ax.scatter(mean_x, mean_y, c='black', s=300, marker='o', edgecolors='white', zorder=10)  # Big black dot
    
    # Labels and aesthetics
    ax.set_title(cat)
    ax.set_xlabel('Abs duration error (s)')
    if i == 0:
        ax.set_ylabel('Normalized position for the median')
    ax.set_xlim(0, filtered_data_1.abs_duration_error.max())
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# Loop through each region and plot histogram
for i, (cat, color) in enumerate(zip(CATEGORY, COLORS)):
    ax = axes[i]
    region_data = filtered_data_1[filtered_data_1.category == cat]
    
    ax.hist(region_data.loc_best, bins=5, color=color, alpha=0.8, edgecolor='k')
    
    ax.set_title(cat)
    ax.set_xlabel('Normalized best position')
    if i == 0:
        ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()


# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# Loop through each region and plot histogram
for i, (cat, color) in enumerate(zip(CATEGORY, COLORS)):
    ax = axes[i]
    region_data = filtered_data_1[filtered_data_1.category == cat]
    
    ax.hist(region_data.loc_median, bins=10, color=color, alpha=0.8, edgecolor='k')
    
    ax.set_title(cat)
    ax.set_xlabel('Normalized position')
    if i == 0:
        ax.set_ylabel('Count')
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()

plt.figure(figsize=(3, 3))
plt.scatter(filtered_data_2.abs_duration_error, filtered_data_2.loc_median, c = 'k', alpha = 0.5);
plt.xlabel('Absolute duration error (s)')
plt.ylabel('Normalized position of the median')
plt.axvline(x = 0.055, c = 'r', linestyle = '--' )
plt.title("EXP 2")
plt.show()

# Histogram for filtered_data_1
plt.figure(figsize=(3, 3))
plt.hist(filtered_data_1.loc[filtered_data_1.abs_duration_error < 0.055, 'loc_median'], 
         bins=10, color='k', alpha=0.7)
plt.xlabel('Normalized position of the median')
plt.ylabel('Count')
plt.title('Filtered Data 1')
plt.xlim(0, 1)
plt.show()

# Histogram for filtered_data_2
plt.figure(figsize=(3, 3))
plt.hist(filtered_data_2.loc[filtered_data_2.abs_duration_error < 0.055, 'loc_median'], 
         bins=10, color='k', alpha=0.7)
plt.xlabel('Normalized position of the median')
plt.ylabel('Count')
plt.title('Filtered Data 2')
plt.xlim(0, 1)
plt.show()


#--------------------------------------
#DURATION ANALYSIS FOR THE PITCH DATA
#--------------------------------------
region_high = whistle_songs_pitch[whistle_songs_pitch.f0 > 4000 ]
region_low = whistle_songs_pitch[whistle_songs_pitch.f0 <= 4000 ]

plt.style.use('default')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 1]})
# KDE plot (First Subplot)
ax1 = axes[0]
kde = sns.kdeplot(whistle_songs_pitch.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)

# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()
y_values = ax1.lines[-1].get_ydata()   # Get the latest plotted KDE line
# Fill the area under the curve with green
ax1.fill_between(x_values, y_values, color='lightseagreen', alpha=0.3)
# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)
kde_control = sns.kdeplot(whistle_songs_control['d_median'], color='red', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)
print(' Mean: ', np.mean(whistle_songs_pitch.d_median), " STD: ", np.std(whistle_songs_pitch.d_median))
print(' Median: ', np.median(whistle_songs_pitch.d_median), " MAD: ", np.median(np.abs(whistle_songs_pitch.d_median - np.median(whistle_songs_pitch.d_median))))
ax1.set_ylabel('Normalized counts per bin', fontsize=12)

ax1.set_xlim([0, 0.9])
# Scatter plot with jittered data (Second Subplot)
ax2 = axes[1]
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(whistle_songs_pitch.d_median))
ax2.scatter(whistle_songs_pitch.d_median, np.zeros_like(whistle_songs_pitch.d_median) + y_jittered, 
                color='black', alpha=0.2, s=2)
ax2.set_xlabel('Median whistle duration (s)')
ax2.vlines(0.440, -0.08, -0.1, color='lightseagreen', lw=2, zorder=5)
ax2.set_yticks([])
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/whistle_durations_pitch_exp.pdf", transparent=True)
plt.show()

#FOR THE LOW PITCH
plt.style.use('default')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 1]})
# KDE plot (First Subplot)
ax1 = axes[0]
kde = sns.kdeplot(region_low.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)

# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()
y_values = ax1.lines[-1].get_ydata()   # Get the latest plotted KDE line
# Fill the area under the curve with green
ax1.fill_between(x_values, y_values, color='lightseagreen', alpha=0.3)
# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)
kde_control = sns.kdeplot(whistle_songs_control['d_median'], color='red', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)
print(' Mean: ', np.mean(region_low.d_median), " STD: ", np.std(region_low.d_median))
print(' Median: ', np.median(region_low.d_median), " MAD: ", np.median(np.abs(region_low.d_median - np.median(region_low.d_median))))
ax1.set_ylabel('Normalized counts per bin', fontsize=12)

ax1.set_xlim([0, 0.9])
# Scatter plot with jittered data (Second Subplot)
ax2 = axes[1]
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(region_low.d_median))
ax2.scatter(region_low.d_median, np.zeros_like(region_low.d_median) + y_jittered, 
                color='black', alpha=0.2, s=2)
ax2.set_xlabel('Median whistle duration (s)')
ax2.vlines(0.440, -0.08, -0.1, color='lightseagreen', lw=2, zorder=5)
ax2.set_yticks([])
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/whistle_durations_low_pitch_exp.pdf", transparent=True)
plt.show()
#FOR THE HIGH PITCH
plt.style.use('default')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 1]})
# KDE plot (First Subplot)
ax1 = axes[0]
kde = sns.kdeplot(region_high.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)

# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()
y_values = ax1.lines[-1].get_ydata()   # Get the latest plotted KDE line
# Fill the area under the curve with green
ax1.fill_between(x_values, y_values, color='lightseagreen', alpha=0.3)
# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)
kde_control = sns.kdeplot(whistle_songs_control['d_median'], color='red', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)
print(' Mean: ', np.mean(region_high.d_median), " STD: ", np.std(region_high.d_median))
print(' Median: ', np.median(region_high.d_median), " MAD: ", np.median(np.abs(region_high.d_median - np.median(region_high.d_median))))
ax1.set_ylabel('Normalized counts per bin', fontsize=12)

ax1.set_xlim([0, 0.9])
# Scatter plot with jittered data (Second Subplot)
ax2 = axes[1]
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(region_high.d_median))
ax2.scatter(region_high.d_median, np.zeros_like(region_high.d_median) + y_jittered, 
                color='black', alpha=0.2, s=2)
ax2.set_xlabel('Median whistle duration (s)')
ax2.vlines(0.440, -0.08, -0.1, color='lightseagreen', lw=2, zorder=5)
ax2.set_yticks([])
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/whistle_durations_high_pitch_exp.pdf", transparent=True)
plt.show()
#---
#PLOT FOR THE OCCURENCE INDEX IN PITCH EXP
#---
plt.figure(figsize=(1, 4))

kde_pitch_exp = gaussian_kde(whistle_songs_pitch['d_median'], bw_method=0.5)
kde_control = gaussian_kde(whistle_songs_control['d_median'], bw_method=0.5)
ratios = kde_pitch_exp(whistle_songs_pitch['d_median']) / (kde_control(whistle_songs_pitch['d_median']) + 0.0001)

# Create a color map based on KDE ratios
cmap = plt.cm.RdGy_r  # Red-White-Blue colormap
norm = plt.Normalize(vmin=0, vmax=2)  # Normalize ratios to [0, 2]
# Horizontal jitter for scatter plot
jitter = np.random.uniform(-0.1, 0.1, size=whistle_songs_pitch['d_median'].shape)  # Random jitter
plt.scatter(jitter, whistle_songs_pitch['d_median'], edgecolor='none', c=ratios, cmap=cmap, norm=norm, s=10)
plt.xticks([])
plt.ylabel('Median duration')
plt.hlines(y=0.440, xmin=0.1, xmax=-0.1, color='lightseagreen', linestyle='-', linewidth=2)

# Find the median
median_value = np.median(np.sort(whistle_songs_pitch['d_median']))
# Plot median as a horizontal line
plt.hlines(y=median_value, xmin=0.1, xmax=-0.1, color='k', linestyle='-', linewidth=3)
plt.show()
#-------------
#p-value
#-------------

real_median_pitch_exp= np.median(whistle_songs_pitch['d_median'])

# Step 2: Shuffle and compute medians
n_iterations = 10000
shuffled_medians_pitch_exp = []

for _ in tqdm(range(n_iterations)):
    shuffled = shuffle_data(whistle_songs_pitch)
    if len(shuffled) > 0:
        median_d = np.median(shuffled.d_median)
        shuffled_medians_pitch_exp.append(median_d)

# Convert to NumPy array
shuffled_medians_pitch_exp = np.array(shuffled_medians_pitch_exp)

# One-sided p-value: proportion of shuffled medians >= real median
p_value = np.mean(shuffled_medians_pitch_exp >= real_median_pitch_exp)

# Print results
print(f"Real median duration (Pitch exp): {real_median_C:.4f} s")
print(f"One-sided p-value (Pitch exp): {p_value:.4f}")

# Step 3: Plot
plt.figure(figsize=(5, 3))
plt.hist(shuffled_medians_pitch_exp, bins=30, color='lightseagreen', edgecolor='lightseagreen')
plt.axvline(real_median_pitch_exp, color='red', linewidth=2, label='Real Median')
plt.xlabel('Median Duration (s)')
plt.ylabel('Count')
plt.title('Pitch experiment')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


def get_kde_mode(data, bw=0.5, grid=None):
    """Compute the mode of a KDE distribution."""
    kde = gaussian_kde(data, bw_method=bw)
    if grid is None:
        grid = np.linspace(0, 1.0, 1000)  # adjust based on expected range
    kde_vals = kde(grid)
    mode_value = grid[np.argmax(kde_vals)]
    return mode_value

# Step 1: Compute the mode of the real distribution
real_mode_pitch_exp = get_kde_mode(whistle_songs_pitch['d_median'])

# Step 2: Shuffle and compute modes
n_iterations = 10000
shuffled_modes_pitch_exp = []

for _ in tqdm(range(n_iterations)):
    shuffled = shuffle_data(whistle_songs_pitch)
    if len(shuffled) > 0:
        mode_d = get_kde_mode(shuffled['d_median'])
        shuffled_modes_pitch_exp.append(mode_d)

shuffled_modes_pitch_exp = np.array(shuffled_modes_pitch_exp)

# One-sided p-value: proportion of shuffled modes >= real mode
p_value_mode = np.mean(shuffled_modes_pitch_exp >= real_mode_pitch_exp)

# Print results
print(f"Real mode of KDE (Pitch exp): {real_mode_pitch_exp:.4f} s")
print(f"One-sided p-value (Pitch exp, using mode): {p_value_mode:.4f}")

# Step 3: Plot
plt.figure(figsize=(5, 3))
plt.hist(shuffled_modes_pitch_exp, bins=30, color='lightseagreen', edgecolor='lightseagreen')
plt.axvline(real_mode_pitch_exp, color='red', linewidth=2, label='Real Mode')
plt.xlabel('Mode of KDE (s)')
plt.ylabel('Count')
plt.title('Pitch experiment (KDE Mode)')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


#--------------------------------------
#DURATION ANALYSIS FOR THE PITCH DATA WITHOUT PITCH MATCH
#--------------------------------------
whistle_songs_pitch_match = whistle_songs_pitch[np.abs(whistle_songs_pitch.f_median - whistle_songs_pitch.f0) < 100]
region_high_match = whistle_songs_pitch_match[whistle_songs_pitch_match.f0 > 4000 ]
region_low_match = whistle_songs_pitch_match[whistle_songs_pitch_match.f0 <= 4000 ]
plt.style.use('default')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 1]})
# KDE plot (First Subplot)
ax1 = axes[0]
kde = sns.kdeplot(whistle_songs_pitch_match.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)
# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()  # Get the latest plotted KDE line
y_values = ax1.lines[-1].get_ydata()  
# Fill the area under the curve with green
ax1.fill_between(x_values, y_values, color='lightseagreen', alpha=0.3)
# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)
kde_control = sns.kdeplot(whistle_songs_control['d_median'], color='red', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)
print(' Mean: ', np.mean(whistle_songs_pitch_match.d_median), " STD: ", np.std(whistle_songs_pitch_match.d_median))
print(' Median: ', np.median(whistle_songs_pitch_match.d_median), " MAD: ", np.median(np.abs(whistle_songs_pitch_match.d_median - np.median(whistle_songs_pitch_match.d_median))))
ax1.set_ylabel('Normalized counts per bin', fontsize=12)

ax1.set_xlim([0, 0.9])
# Scatter plot with jittered data (Second Subplot)
ax2 = axes[1]
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(whistle_songs_pitch_match.d_median))
ax2.scatter(whistle_songs_pitch_match.d_median, np.zeros_like(whistle_songs_pitch_match.d_median) + y_jittered, 
                color='black', alpha=0.2, s=2)
ax2.set_xlabel('Median whistle duration (s)')
ax2.vlines(0.440, -0.08, -0.1, color='lightseagreen', lw=2, zorder=5)
ax2.set_yticks([])
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/whistle_durations_pitch_exp_match.pdf", transparent=True)
plt.show()

#FOR THE LOW PITCH
plt.style.use('default')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 1]})
# KDE plot (First Subplot)
ax1 = axes[0]
kde = sns.kdeplot(region_low_match.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)

# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()
y_values = ax1.lines[-1].get_ydata()   # Get the latest plotted KDE line
# Fill the area under the curve with green
ax1.fill_between(x_values, y_values, color='lightseagreen', alpha=0.3)
# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)
kde_control = sns.kdeplot(whistle_songs_control['d_median'], color='red', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)
print(' Mean: ', np.mean(region_low_match.d_median), " STD: ", np.std(region_low_match.d_median))
print(' Median: ', np.median(region_low_match.d_median), " MAD: ", np.median(np.abs(region_low_match.d_median - np.median(region_low_match.d_median))))
ax1.set_ylabel('Normalized counts per bin', fontsize=12)

ax1.set_xlim([0, 0.9])
# Scatter plot with jittered data (Second Subplot)
ax2 = axes[1]
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(region_low_match.d_median))
ax2.scatter(region_low_match.d_median, np.zeros_like(region_low_match.d_median) + y_jittered, 
                color='black', alpha=0.2, s=2)
ax2.set_xlabel('Median whistle duration (s)')
ax2.vlines(0.440, -0.08, -0.1, color='lightseagreen', lw=2, zorder=5)
ax2.set_yticks([])
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/whistle_durations_low_pitch_exp_match.pdf", transparent=True)
plt.show()
#FOR THE HIGH PITCH
plt.style.use('default')
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 1]})
# KDE plot (First Subplot)
ax1 = axes[0]
kde = sns.kdeplot(region_high_match.d_median, color='black', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)

# Extract x-values from the KDE plot
x_values = ax1.lines[-1].get_xdata()
y_values = ax1.lines[-1].get_ydata()   # Get the latest plotted KDE line
# Fill the area under the curve with green
ax1.fill_between(x_values, y_values, color='lightseagreen', alpha=0.3)
# Compute the spacing between consecutive x-values
x_spacing = np.diff(x_values)
kde_control = sns.kdeplot(whistle_songs_control['d_median'], color='red', linestyle='-', lw=2, bw_adjust=0.5, ax=ax1)
print(' Mean: ', np.mean(region_high_match.d_median), " STD: ", np.std(region_high_match.d_median))
print(' Median: ', np.median(region_high_match.d_median), " MAD: ", np.median(np.abs(region_high_match.d_median - np.median(region_high_match.d_median))))
ax1.set_ylabel('Normalized counts per bin', fontsize=12)

ax1.set_xlim([0, 0.9])
# Scatter plot with jittered data (Second Subplot)
ax2 = axes[1]
y_jittered = np.random.uniform(low=-0.075, high=0.075, size=len(region_high_match.d_median))
ax2.scatter(region_high_match.d_median, np.zeros_like(region_high_match.d_median) + y_jittered, 
                color='black', alpha=0.2, s=2)
ax2.set_xlabel('Median whistle duration (s)')
ax2.vlines(0.440, -0.08, -0.1, color='lightseagreen', lw=2, zorder=5)
ax2.set_yticks([])
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("Plots/whistle_durations_high_pitch_exp_match.pdf", transparent=True)
plt.show()

#---
#DENSITY PLOT FOR THE PITCH MATCH
#---
# Define a custom colormap from white to yellowgreen
custom_cmap = LinearSegmentedColormap.from_list("white_to_olivedrab", ["white", "olivedrab"])

fig = plt.figure(figsize=(5, 5))  # Making figure square

# Scatter plot for playback points (marked with red '+')
plt.plot(0.440* np.ones_like(whistle_songs_pitch_match.f0.unique()), whistle_songs_pitch_match.f0.unique(), 
         color="lightseagreen", marker='+', linestyle='None', 
         markersize=10, markeredgewidth=2)

# KDE plot using the custom colormap
kde = sns.kdeplot(x=whistle_songs_pitch_match.d_median, 
                   y=whistle_songs_pitch_match.f_median, 
                   cmap=custom_cmap, fill=True, 
                   bw_adjust=0.7, levels=20, thresh=0.1, zorder=-1)

# Outer contour with a reversed red colormap
outside = sns.kdeplot(x=whistle_songs_control.d_median, 
                      y=whistle_songs_control.f_median, 
                      cmap="Reds_r", fill=False,
                      bw_adjust=0.7, levels=1, thresh=0.1)

# Scatter plot for whistle songs control (small black points)
plt.scatter(whistle_songs_pitch_match.d_median, whistle_songs_pitch_match.f_median, 
            c='k', s=2, zorder=1)

# Labels and axis limits
plt.xlabel('Median of whistle durations per song (s)')
plt.ylabel('Median frequency (kHz)')

# Square aspect ratio
plt.gca().set_box_aspect(1)

plt.tight_layout()

# Set font type for PDF export
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('Plots/pitchexp_meanvspitch.pdf', transparent=True)
plt.show()

#RATIO PLOT
data = whistle_songs_pitch_match
x_exp, y_exp = data.d_median, data.f_median
kde_exp = gaussian_kde(np.vstack([x_exp, y_exp]), bw_method=0.4)

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
plt.contourf(X, Y, ratio, levels=10, cmap=cmap, norm=norm)

# Overlay control KDE
sns.kdeplot(x=whistle_songs_control.d_median, y=whistle_songs_control.f_median, 
            cmap="Reds_r", fill=False, bw_adjust=0.7, levels=1, thresh=0.1)
plt.show()


#--------------------------------------
#PITCH MATCH PLOT ANALYSIS
#--------------------------------------

plt.style.use('default')

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 4),
                         sharex=True,
                         gridspec_kw={'height_ratios': [2, 1.5, 2, 1.5]})

# --------------------------------------
# Control density -> probability
# --------------------------------------
kde = sns.kdeplot(whistle_songs_control.d_median, color='black', linestyle='-', lw=2,
                  bw_adjust=0.4, common_norm=True, ax=axes[0])

# Extract x and y values from Seaborn KDE line
x_vals = kde.lines[0].get_xdata()
y_vals = kde.lines[0].get_ydata()
dx = np.mean(np.diff(x_vals))

# Convert density to probability
y_vals_prob = y_vals * dx

# Remove Seaborn KDE line and plot the probability version
axes[0].lines[-1].remove()
axes[0].plot(x_vals, y_vals_prob, color='black', linestyle='-', lw=2)
axes[0].fill_between(x_vals, y_vals_prob, color='gray', alpha=0.2)
axes[0].set_xlim([0, 0.9])
axes[0].set_ylabel("Probability")

# Scatter jittered data for Control
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

# Convert this KDE to probability
x_vals_pitch = kde.lines[0].get_xdata()
y_vals_pitch = kde.lines[0].get_ydata()
dx_pitch = np.mean(np.diff(x_vals_pitch))
y_vals_prob_pitch = y_vals_pitch * dx_pitch

# Remove Seaborn KDE line and plot the probability version
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

# Print statistics
print(category, ' Mean: ', np.mean(data.d_median), " STD: ", np.std(data.d_median))
print(category, ' Median: ', np.median(data.d_median),
      " MAD: ", np.median(np.abs(data.d_median - np.median(data.d_median))))

# Set y-limits based on max probability in each plot
ylim_max = max(max(y_vals_prob), max(y_vals_prob_pitch)) * 1.05
axes[0].set_ylim(0, ylim_max)
axes[2].set_ylim(0, ylim_max)

axes[3].set_xlabel('Whistle duration (s)')

plt.tight_layout(rect=[0, 0, 0.9, 1])
mpl.rcParams['pdf.fonttype'] = 42
fig.savefig('Plots/Results exp_pitch.pdf', transparent=True)
plt.show()

# Extract durations
control_durations = whistle_songs_control.d_median.values
playback_durations = data.d_median.values  # Already filtered by pitch match

# Observed difference in medians
observed_diff = np.median(playback_durations) - np.median(control_durations)

# Combine datasets for permutation
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

# ----------------------------------
# Plot histogram of null distribution
# ----------------------------------
plt.figure(figsize=(4, 4))  # Square figure physically

counts, bins, patches = plt.hist(null_distribution, bins=40, color='mediumaquamarine')  # This draws it with counts by default

plt.cla()  # Clear the current axes before re-plotting

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