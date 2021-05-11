# ivadomedEEG

This repo utilizes mne-python to create nifti files for each mne epoch.

As an example, one of the tutorial datasets is loaded and epochs are segmented around blink events.
Sequentially, channels that are mostly affected around the blinks are annotated as the ground truth for the IVADOMED training.

The files follow the BIDS tree that IVADOMED uses + extra .csv files per subject (and its derivatives) that show channels' locations in NIFTI pixel coordinates, and the time that each slice corresponds to.
