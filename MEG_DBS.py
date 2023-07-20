#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:01:40 2019

@author: Nithya Ramakrishnan
"""

#### MEG ASSR PRE-PROCESSING PIPELINE 
import mne
import numpy as np
import scipy
from scipy import signal, stats
import os.path as op
import matplotlib.pyplot as plt
import copy
import random
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing.ica import ICA
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.preprocessing import maxwell_filter
from mne.viz import plot_projs_topomap
from mne.io import RawArray
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
from mne.time_frequency import csd_morlet
import bad_data as fbc

#%% Loading Directory

# in this section we will begin the FIRST LOOP, this will be the outer loop
# which loads up each individual subject.
data_path = '/raid5/goodman/DBS_GOODMAN/MEG/' #subject, raw, ssaep
patient =  'aDBS004' #change this number as per the subjectid 
megid ='0489'
session=1
run='EYESOPEN1'#change this number as per the runid
subjects_dir=op.join(data_path, '%s/fs_test/' %(patient))
raw_dir='Raw_files'


#raw_tsss_fname = op.join(data_path, '%s/session%s/%sDB_%s_tsss_mc.fif' %(patient, session, megid, run)) #ssaep.fif
#raw_tsss_elekta= mne.io.read_raw_fif(raw_tsss_fname,allow_maxshield=True,preload=True)

raw_fname = op.join(data_path, '%s/session%s/%s/%sDB_%s.fif' %(patient, session, raw_dir, megid, run)) 
icaname = op.join(data_path, '%s/session%s/ica/%s_%s_rawica.fif' %(patient, session, megid, run)) 
saveall = op.join(data_path, '%s/session%s/epochs/%s_%s_all-epo.fif' %(patient, session, megid, run))
reportsdir=op.join(data_path, '%s/session%s/reports/' %(patient, session))
fname_report=op.join(reportsdir,'%s_%s_report.html' %(megid, run))
fname_report_h5=op.join(reportsdir,'%s_%s_report.h5' %(megid, run))
fname_csd=op.join(data_path, '%s/session%s/epochs/%s_%s_csd.h5' %(patient, session, megid, run))


##### LOAD DATA AND PREPARE MONTAGE
raw = mne.io.read_raw_fif(raw_fname,allow_maxshield=True,preload=True)
raw_filt=raw.copy().filter(1, 50,l_trans_bandwidth='auto',h_trans_bandwidth='auto', 
                         filter_length='auto', phase='zero',fir_window='hamming', fir_design='firwin', n_jobs=20)
# Bandpass the data.
#raw_filt = raw.copy().filter(bandpass_fmin, bandpass_fmax, l_trans_bandwidth='auto',
#                   h_trans_bandwidth='auto', filter_length='auto', phase='zero',
#                   fir_window='hamming', fir_design='firwin', n_jobs=n_jobs)
#raw_filt.notch_filter(np.arange(60,241,60), filter_length='auto',phase='zero')

# Highpass the EOG channels to > 1Hz, regardless of the bandpass-filter
# applied to the other channels
#picks_eog = mne.pick_types(raw_sss.info, meg=False, eog=True)
#raw_filt.filter(
#    1., None, picks=picks_eog, l_trans_bandwidth='auto',
#    filter_length='auto', phase='zero', fir_window='hann',
#    fir_design='firwin', n_jobs=n_jobs)

## Make a plot of the PSD before and after filtering
#figs_before=raw.plot_psd(fmax=200,show=False)
#figs_after=raw_filt.plot_psd(fmax=200,show=False)



#raw_sss.plot_psd(proj=True
#bandpass_fmin=0.01
#bandpass_fmax=100
#n_jobs=20
n_components_1=0.97
n_components_2=0.999
n_ecg_components=1
n_eog_components=1
epoch_tmin, epoch_tmax= 0, 3
#baseline=(-0.2, 0)
csd_tmin, csd_tmax = 0, 3
spacing = 'ico4'
#max_sensor_dist = 0.07
#min_skull_dist = 0
reg = 0.05
freq_bands = [
    (3, 7),     # theta
    (7, 13),    # alpha
    (13, 17),   # low beta
    (17, 25),   # high beta 1
    (25, 31),   # high beta 2
    (31, 40),   # low gamma
]
con_fmin = 31
con_fmax = 40

# Minimum distance between sources to compute connectivity for (in meters)
min_pair_dist = 0.04
eogch = [e for e in raw.info['ch_names'] if "EOG" in e][0]
ecgch = [e for e in raw.info['ch_names'] if "ECG" in e][0]
 
#%% Pre-Processing Step 1
#raw.filter(.01, 60.0, method='iir')
#reject = dict(mag=4e-12,grad=4000e-13, eog=150e-6)
raw.pick_types(meg=True, eeg=True, eog=True, ecg=True, stim=False)

#raw.plot()
# Keep track of PSD plots before and after filtering
figs_before = []
figs_after = []

# Append PDF plots to report
report=mne.open_report(fname_report_h5)
#    
#report.add_figs_to_section(
#        figs_before,
#        captions='PSD before filtering',
#        section='Sensor-level',
#        replace=True)
#report.add_figs_to_section(
#        figs_after,
#        captions='PSD after filtering',
#        section='Sensor-level',
#        replace=True)
#report.save(fname_report, open_browser=False, overwrite=True)


#%% Pre-Processing Step 2

# in this section we run the ICA algorithm to remove EOG, ECG, and EMG artifacts.
# we have two options for how to approach this:
#   1) Run the ICA using the raw, continuous data.  
#      In order to do this we would fit the infomax ICA parameters to the raw object
#      and set the rejection criteria for bad data.

# here we set the random state seed, and apply extended infomax with PCA reduction
# to 99% of the variance. Data is rejected if it exceeds the peak to peak 
# amplitude thresholds set in the "reject" dictionary


print('fitting ica')
ica = ICA(method='extended-infomax', random_state=42, n_components=n_components_1)
ica.fit(raw, decim=11)

#ica.fit(raw_filt, reject=dict(grad=4000e-13, mag=4e-12), decim=3)
print('Fit %d components (explaining at least %0.1f%% of the variance)'
      % (ica.n_components_, 100 * n_components_1))

# Find onsets of heart beats and blinks. Create epochs around them
ecg_epochs = create_ecg_epochs(raw, ch_name=ecgch, tmin=-.3, tmax=.3, preload=False)
eog_epochs = create_eog_epochs(raw, ch_name=eogch, tmin=-.5, tmax=.5, preload=False)

# Find ICA components that correlate with heart beats.
ecg_epochs.decimate(5)
ecg_epochs.load_data()
ecg_epochs.apply_baseline((None, None))
ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, method='correlation')
ecg_scores = np.abs(ecg_scores)
rank = np.argsort(ecg_scores)[::-1]
if rank.sum() > 0:
    rank = [r for r in rank if ecg_scores[r] > 0.05]
    ica.exclude = rank[:n_ecg_components]
print('    Found %d ECG indices' % (len(ecg_inds),))

# Find ICA components that correlate with eye blinks
eog_epochs.decimate(5)
eog_epochs.load_data()
eog_epochs.apply_baseline((None, None))
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
eog_scores = np.max(np.abs(eog_scores), axis=0)
# Remove all components with a correlation > 0.1 to the EOG channels and that
# have not already been flagged as ECG components
rank = np.argsort(eog_scores)[::-1]
if rank.sum() > 0:
    rank = [r for r in rank if eog_scores[r] > 0.1 and r not in ecg_inds]
    ica.exclude += rank[:n_eog_components]
print('    Found %d EOG indices' % (len(eog_inds),))

#plot ica component time series to manually select components 
ica.get_sources(raw_sss)
ica.plot_sources(raw_sss)

#icafig = ica.plot_sources(raw_filt,show=False)
#report.add_figs_to_section(icafig,captions='ICAsources',
#                           section='Sensor-level',replace=True)
## Save plots of the ICA components to the report
compfigs = ica.plot_components(show=True)

#report.add_slider_to_section(
#        compfigs,
#        ['ICA components %d' % i for i in range(len(compfigs))],
#        title='ICA components',
#        section='Sensor-level',
#        replace=True)
#
#report.save(fname_report, overwrite=True,
#            open_browser=False)
ica.apply(raw,n_pca_components=0.99,exclude = ica.exclude) # apply the ICA correction

# ica again
print('fitting ica 2nd time')
ica3 = ICA(method='extended-infomax', random_state=42, n_components=n_components_2)
ica3.fit(raw, decim=11)

ica3.get_sources(raw)
ica3.plot_sources(raw)
compfigs3 = ica3.plot_components(show=True)

icafig3=ica3.plot_sources(raw, show=True)

report.add_figs_to_section(icafig3,captions='ICAsources',
                           section='Sensor-level',replace=True)
#report.add_figs_to_section(compfigs3,captions='ICAsources',
#                           section='Sensor-level',replace=True)

report.add_slider_to_section(compfigs3,
        ['ICA components %d' % i for i in range(len(compfigs3))],
        title='ICA components',
        section='Sensor-level',
        replace=True)

report.save(fname_report, overwrite=True,
            open_browser=False)
#ica.fit(raw_filt2, reject=dict(grad=4000e-13, mag=4e-12), decim=3)
print('Fit %d components (explaining at least %0.1f%% of the variance)'
      % (ica3.n_components_, 100 * n_components_2))

ica3.apply(raw,exclude=ica3.exclude) # apply the ICA correction

#if np.shape(ica3.exclude)[0]>0:
#    ica3.apply(raw_filt,n_pca_components=0.99,exclude = ica2.exclude) # apply the ICA correction
#
# Save the ICA decomposition
#ica2.save(icaname)
import gcmi 
from gcmi import gcmi_cc

eegchan=raw.get_data(picks='EEG010')
eegchan2=raw.get_data(picks='EEG015')

fig = plt.figure()
ax2d = fig.add_subplot(121)
#ax3d = fig.add_subplot(122, projection='3d')
raw.plot_sensors(ch_type='eeg', axes=ax2d)
#raw_sss.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d')
#ax3d.view_init(azim=70, elev=15)

icachans=ica3.get_sources(raw).get_data(picks='all')
miall=[]
mi=[]
for j in icachans:
    mi=gcmi_cc(eegchan,j)
    miall=np.append(miall,mi)
mean=np.mean(miall)
std=np.std(miall)
threshold=mean+std
comprem=np.where(miall>threshold)


#mi=[gcmi_cc(j,i) for i in icachans for j in eegchan]
#
#for i in icachans:
#    for e in eegchan:
#        mi.append(gcmi_cc(e,i))
#        
#    mi=gcmi_cc(eegchan,j)
#    miall=np.append(miall,mi)
#eegchan2=raw_sss4.get_data(picks='BIO003')

miall2=[]
mi2=[]
for j in icachans:
    mi2=gcmi_cc(eegchan2,j)
    miall2=np.append(miall2,mi2)
mean2=np.mean(miall2)
std2=np.std(miall2)
threshold2=mean2+2*std2
comprem2=np.where(miall2>threshold2)


raw_new=raw.copy()
ica3.exclude=[34,35]
ica3.apply(raw_new)

#n_components_2=0.96
#print('fitting ica')
#ica = ICA(method='extended-infomax', random_state=42, n_components=n_components_2)
#
#ica.fit(raw_sss,decim=11)


#bad channel removal
raw_filt_new = fbc.process_third_stage(raw_filt)
raw_filt_new.plot()
print(raw_filt_new.info['bads'])
badchan=raw_filt_new.info['bads']



#%% EVENT PROCESSING & BAD TRIAL REJECTION
picks2 = mne.pick_types(raw.info, meg=True, eeg=True, eog=False,ecg=False,
                        stim=False, exclude=badchan)
events=mne.make_fixed_length_events(raw_sss, id=1, duration=4.0)
epochs=mne.Epochs(raw,events,1,proj=False,picks=picks2,tmin=0,tmax=3, preload=True)

reject=get_rejection_threshold(epochs,ch_types=['grad'],decim=6)
print('The rejection dictionary is %s' % reject)
# Drop epochs that have too large signals (most likely due to the subject
# moving or muscle artifacts)
epochs.drop_bad(reject=reject)
print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))
epochs.plot()
print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))

# Save evoked plot to report

report.add_figs_to_section(
    [epochs.average().plot(show=False)],
    ['Evoked with ICA and bad epochs dropped'],
    section='Sensor-level',
    replace=True
)
report.save(fname_report, overwrite=True,
            open_browser=False)


print('  Writing to disk')
epochs.save(saveall)
report.save(fname_report_h5, overwrite=True)


fmin = freq_bands[0][0]
fmax = freq_bands[-1][1]
frequencies = np.arange(fmin, fmax + 1, 2)
csd = csd_morlet(epochs, frequencies=frequencies, tmin=csd_tmin,
                     tmax=csd_tmax, decim=10, n_jobs=20, verbose=True)

# Save the CSD matrices
csd.save(fname_csd)
report.add_figs_to_section(csd.plot(show=False),
                           ['CSD for %s' % run],section='Sensor-level', replace=True)
report.save(fname_report, overwrite=True,
            open_browser=False)
report.save(fname_report_h5, overwrite=True)
