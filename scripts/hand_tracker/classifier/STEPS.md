# Hand Pose Classifier Training and Evaluation Steps

## Data

- Correct the annotations to make sure there is no mismatch between the start of `PICK` sequence to the end of `NOT_PICK` sequence
- Run `utils/process_gt.py`. Make sure that the number of frames match with the number of annotated frames
- Run `utils/convert_video2image.py` to convert each frame of the video to an image and obtain the image list text file.
- Run `utils/filter_data.py` to filter out confusing data instances and balance the dataset
- [Optional] Run `utils/visualize_labels.py` with `gt=True` to make sure the labels match up with the images using the annotated CSV file

## Training

- Run `train.py` and obtain the saved weights as a `.pth` file.

## Evaluation

- Run `evaluate.py` and get class-wise accuracy and overall accuracy. Also obtain a `.npy` file with the filename, prediction and ground truth.
- Run `utils/pick_onset_evaluator.py` with the `.npy` file obtained from `evaluate.py` to evaluate how many onsets of picks were correctly detected by the model. Also, obtain another `.npy` files with the ground-truth and prediction sequence lengths for a given `PICK` sequence.
- Run `utils/visualize_labels.py` with `gt=False` with the `.npy` files obtained from the previous two steps to visualize the ground-truth, predictions and their respective `PICK` sequence lengths on all the images and save them to another folder