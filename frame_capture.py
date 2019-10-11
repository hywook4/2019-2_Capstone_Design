import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--dirpath', type=str, required=True, help="Directory path")
parser.add_argument('-cl', '--cutlength', type=int, required=True, help="Clip length")
args = parser.parse_args()




vidcap = cv2.VideoCapture('twice3-1.mp4')
success,image = vidcap.read()
count = 0
success = True

while success:
  cv2.imwrite("test/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

