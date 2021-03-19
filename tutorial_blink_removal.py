import numpy as np
import mne
from mne.datasets import sample


data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)
raw.set_eeg_reference('average', projection=True)


raw.info['bads'] = ['MEG 2443']


# Reading data with a bad channel marked as bad:
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname, condition='Left Auditory',
                          baseline=(None, 0))

# restrict the evoked to EEG and MEG channels
evoked.pick_types(meg=True, eeg=True, exclude=[])

# plot with bads
evoked.plot(exclude=[])

print(evoked.info['bads'])


events = mne.find_events(raw)
ecg_event_id = 999
eog_event_id = 998

eog_events = mne.preprocessing.find_eog_events(raw)
ecg_events = mne.preprocessing.find_ecg_events(raw)
ecg_events = np.asarray(ecg_events[0])

n_blinks = len(eog_events)
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
description = ['blink'] * n_blinks
#annotations = mne.Annotations(onset, duration, description)
#raw.set_annotations(annotations)
epochs_blink = mne.Epochs(raw, eog_events, eog_event_id, reject=None, preload=True)
raw.plot(events=eog_events)  # To see the annotated segments.
epochs_blink = epochs_blink[:-6]


n_heartbeat = len(ecg_events)
onset = ecg_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_heartbeat)
description = ['heartbeat'] * n_heartbeat
#annotations = mne.Annotations(onset, duration, description)
#raw.set_annotations(annotations)
epochs_heartbeat = mne.Epochs(raw, ecg_events, ecg_event_id, reject=None, preload=True)
raw.plot(events=ecg_events)  # To see the annotated segments.



#epochs_all = mne.Epochs(raw, np.append(ecg_events, eog_events, axis=0), [ecg_event_id, eog_event_id],
#                        reject=None, preload=True)

print('asdf')



# Now start creating the NIFTI files
import mne_bids
from export_epoch_to_nifti import write_trials_bids






single_epoch = epochs_blink[0]
#single_epoch.resample(sfreq=10)
epochs_blink['998'].plot_image(picks='meg')
epochs_heartbeat['999'].plot_image(picks='meg')


for iSubject in range(1, 15):
    annotated_event_for_gt = '998'  # This is the event that will be used to create the derivatives
                                    # 999 Heartbeats
                                    # 998 Blinks
    #epochs_ = epochs_heartbeat[(iSubject-1)*3:iSubject*3]
    epochs_ = epochs_blink[(iSubject-1)*3:iSubject*3]

    # Create bids folder
    bids_path = mne_bids.BIDSPath(subject='IVADOMEDSubjTest' + str(iSubject), session='IVADOMEDSession1', task='testing',
                                   acquisition='01', run='01', root='/home/nas/Desktop/test_BIDS')

    bid_path_return = write_trials_bids(epochs_, bids_path, annotated_event_for_gt, overwrite=True)

'''
single_epoch = epochs_[0]
trial = single_epoch.average()
trial.plot_topomap(epochs_.times[0], size=8, extrapolate='head', colorbar=False, cmap='Greys',
                                 outlines=None, contours=0, show=True, sensors=True, show_names=True)  # res = int selects res
'''