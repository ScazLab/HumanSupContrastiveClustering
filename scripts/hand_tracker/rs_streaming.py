# coding: utf-8
import cv2
import numpy as np
from realsensecv import RealsenseCapture

cap = RealsenseCapture()
# Property setting
cap.WIDTH = 640
cap.HEIGHT = 480
cap.FPS = 30
# Unlike cv2.VideoCapture (), do not forget cap.start ()
cap.start()

while True:

    ret, frames = cap.read() # RGB to # frames [0], frames [1] in the image that contains the ndarray of Depth
    
    color_frame = frames[0]
    depth_frame = frames[1]

    # in the heat map conversion
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.08), cv2.COLORMAP_JET)

    # rendering
    images = np.hstack((color_frame, depth_colormap))  # display side by side RGB and Depth next to
    cv2.imshow('RealSense', images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Streaming Stop
cap.release()
cv2.destroyAllWindows()
 