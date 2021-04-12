from autoreject import AutoReject
import numpy as np
import sys

try:
    import mne
except ImportError as error:
    sys.path.append("/home/nas/PycharmProjects/mne-python/")
    import mne

import mne_bids

from config import export_folder

data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)
raw.set_eeg_reference('average', projection=True)


raw.info['bads'] = ['MEG 2443']

# Assign line frequency - Required for BIDS export
raw.info['line_freq'] = 60


# Reading data with a bad channel marked as bad:
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname, condition='Left Auditory',
                          baseline=(None, 0))

# restrict the evoked to EEG and MEG channels
evoked.pick_types(meg=True, eeg=True, exclude=[])

# plot with bads
#evoked.plot(exclude=[])

print(evoked.info['bads'])



events = mne.find_events(raw)
ecg_event_id = 999
eog_event_id = 998

eog_events = mne.preprocessing.find_eog_events(raw)
ecg_events = mne.preprocessing.find_ecg_events(raw)
ecg_events = np.asarray(ecg_events[0])

epochs_all = mne.Epochs(raw, np.concatenate((events, ecg_events, eog_events), axis=0), 
                        reject=None, preload=True, event_repeated='drop')

ar = AutoReject(random_state=97, n_jobs=1)
epochs_ar, reject_log = ar.fit_transform(epochs_all, return_log=True)


n_blinks = len(eog_events)
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
description = ['blink'] * n_blinks
#annotations = mne.Annotations(onset, duration, description)
#raw.set_annotations(annotations)
epochs_blink = mne.Epochs(raw, eog_events, eog_event_id, reject=None, preload=True)
#raw.plot(events=eog_events)  # To see the annotated segments.
epochs_blink = epochs_blink[:-6]


n_heartbeat = len(ecg_events)
onset = ecg_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_heartbeat)
description = ['heartbeat'] * n_heartbeat
#annotations = mne.Annotations(onset, duration, description)
#raw.set_annotations(annotations)
epochs_heartbeat = mne.Epochs(raw, ecg_events, ecg_event_id, reject=None, preload=True)
#raw.plot(events=ecg_events)  # To see the annotated segments.

#epochs_all = mne.Epochs(raw, np.append(ecg_events, eog_events, axis=0), [ecg_event_id, eog_event_id],
#                        reject=None, preload=True)


# Now start creating the NIFTI files
import export_epoch_to_nifti_small

#single_epoch = epochs_blink[0]
#single_epoch.resample(sfreq=10)
#epochs_blink['998'].plot_image(picks='meg')
#epochs_heartbeat['999'].plot_image(picks='meg')

# Select channel type to create the topographies on
ch_type = 'grad'

for iSubject in range(1, 15):
    annotated_event_for_gt = '998'  # This is the event that will be used to create the derivatives
                                    # 999 Heartbeats
                                    # 998 Blinks
    #epochs_ = epochs_heartbeat[(iSubject-1)*3:iSubject*3]
    epochs_ = epochs_blink[(iSubject-1)*3:iSubject*3]

    # Create bids folder
    bids_path = mne_bids.BIDSPath(subject='IVADOMEDSubjTest' + str(iSubject), session='IVADOMEDSession1',
                                  task='testing', acquisition='01', run='01', root=export_folder)

    # Use the raw object that the trials came from in order to build the BIDS tree
    mne_bids.write_raw_bids(raw, bids_path, overwrite=True, verbose=True)

    # Export trials into .nii files
    export_epoch_to_nifti_small.run_export(epochs_, ch_type, annotated_event_for_gt, bids_path)
