import os.path
import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('Dance_1_down.mov')
success,image = vidcap.read()
count = 0
success = True
while success:
  if os.path.exists("./Video_frame/frame_%d.jpg" % count) == False:
    cv2.imwrite("./Video_frame/frame_%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  # print 'Read a new frame: ', success
  count += 1