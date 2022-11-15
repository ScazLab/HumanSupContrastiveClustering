import cv2
import moviepy.editor as mpy
import os
import numpy as np

ROOT = os.path.join(os.path.expanduser("~"), "hand_pose_classification_dataset")
# P_ID = ['P_7', 'P_20', 'P_22', 'P_23', 'P_24', 'P_25', 'P_27', 'P_42', 'P_74', 'P_77']
P_ID = ['P_3', 'P_5', 'P_1']
IMAGE_DIR_NAME = 'Images/'
FILE_LIST = "all_eval_file_list.text"

def convert_video2frames(video_path, image_save_dir, p_id):
    print(p_id)
    vid = mpy.VideoFileClip(video_path)
    file_array = np.array([])
    for i, (tstamp, frame) in enumerate(vid.iter_frames(with_times=True)):
            frame_name = p_id + '_frame_' + str(i) + '.png'

            file_path = os.path.join(image_save_dir, frame_name)
            file_array = np.append(file_array, file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, frame)
    return file_array

def main():
    files = np.array([])
    text_save_path = os.path.join(ROOT, FILE_LIST)
    for p_id in P_ID:
        data_dir = os.path.join(ROOT, p_id)
        video_path = os.path.join(data_dir, p_id+'.webm')
        if p_id == 'P_3':
                video_path = os.path.join(data_dir, p_id+'.mp4')
        image_save_dir = os.path.join(data_dir, IMAGE_DIR_NAME)
        
        if not os.path.exists(image_save_dir):
            os.mkdir(image_save_dir)
        file_array = convert_video2frames(video_path, image_save_dir, p_id)
        files = np.append(files, file_array)
    np.savetxt(text_save_path, files,  fmt='%s')

if __name__ == "__main__":
    main()