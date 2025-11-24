# LPCVC-Track-2

# Preprocessing QEVD Dataset
We ran refactor_dataset.py with appropriate main() directory names to build the formatted QEVD dataset 


# Clean Dataset
We used check_videos.py, a generated script, to check and clean for corrupted videos

# Note on dynamic frame selection
We modified torchvision/datasets/video_utils.py

# To train the model: 

run references/video_classification/train.py, with appropriate arguments for 
  
  --data-path(path to dataset root, being root/ train|val / action categories
  
  --resume(include if testing/resuming training from checkpoint) 
  
  --start-epoch(include if resuming training from a checkpoint
  
  --weights(we used 'KINETICS400_V1' for initial weights)
  
  --cache-dataset(caches processed dataset videos, saves time)

  
**  
 More details on parameters can be found in train.py in the ArgParse**



# To validate the model: 

run train.py with --test-only and --resume






