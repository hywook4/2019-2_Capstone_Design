import os
from datetime import timedelta
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--dirpath', type=str, required=True, help="Directory path")
args = parser.parse_args()

dir_path = args.dirpath

string = os.popen('ls ' + dir_path + ' | grep .mp4').read()

mp4_list = string.split("\n")

video_list = []
video_len = []

for v in mp4_list:
    if v[-4:] == ".mp4":
        video_list.append(v)

print(video_list)

for v in video_list:
    cmd = "ffmpeg -i " + dir_path + "/" + v + "  2>&1 | grep \"Duration\"| cut -d ' ' -f 4 | sed s/,// | sed 's@\..*@@g' | awk '{ split($1, A, \":\"); split(A[3], B, \".\"); print 3600*A[1] + 60*A[2] + B[1] }'"
    print(v)
    print(os.popen(cmd).read())
    video_len.append(int(os.popen(cmd).read()))

for i in range(0, len(video_list)):
    print(video_list[i], video_len[i])

for i in range(0, len(video_len)):
    cmd = "scenedetect -i " + video_list[i] + " -o ./ list-scenes split-video detect-threshold --threshold 60"
    os.system(cmd)