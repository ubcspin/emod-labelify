# Emotion-Modelling Labelify

These are the set of modules used for emotion label generation

# Guide (Raw emotion "feeltrace" extraction)
- unzip trial_data_split-anon.zip from the server into the a folder called participant folder (participant/p1 participant/p2 etc should be visible)
- From the src directory run ```python3 main.py``` or ```python main.py``` on non unix machines. This will create the csv files for each subject and will take some time since the dataset is large (~5GB)
- The csv files containing the feeltrace signals are located in 'feeltrace' by default

# Notes
    - The feeltrace signals are sampled at ~30Hz but other signals, for example, EEG are sampled at 1000Hz. Further processing is required to account for this
    - The feeltrace signals are cropped with the EEG
