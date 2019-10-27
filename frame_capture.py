import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--dirpath', type=str, required=True, help="Directory paht)
args = parser.parse_args()

dir_path = args.dirpath

string = os.popen('ls' + dir_path + ' | grep.mp4').read()

mp4_list = string.split("\n")

video_list = []
                    
for v in mp4_list:
    if v[-4:] == ".mp4":
        video_list.append(v)

print(video_list)

for i in range(0, len(video_list))                  
  vidcap = cv2.VideoCapture(video_list[i])
  success,image = vidcap.read()
  count = 0
  success = True
  while success:
    success,image = vidcap.read()
    if(count%9==0):
        imageName = name[:-4] + str(count) + '.jpg'
        cv2.imwrite(imageName, image)     # save frame as JPEG file
    count += 1
