import os
from datetime import timedelta
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--dirpath', type=str, required=True, help="Directory path")
parser.add_argument('-cl', '--cutlength', type=int, required=True, help="Clip length")
args = parser.parse_args()


dir_path = args.dirpath
cut_len = args.cutlength


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


for i in range(0, len(video_list)):
    # cut_len 길이 이상이면 cut_len 길이의 클립 여러개로 자르기
    print("Cutting video : ", video_list[i])
    if video_len[i] > cut_len:
        cur_start = 0
        filename = video_list[i].split('.')[0]
        count = 1

        while(cur_start + cut_len <= video_len[i]):
            start_time = "0" + str(timedelta(seconds=cur_start))
            end_time = "0" + str(timedelta(seconds=cur_start+cut_len))

            cmd =  "ffmpeg -t " + end_time 
            cmd += " -i " + dir_path + "/" + video_list[i]
            cmd += " -ss " + start_time
            cmd += " " + dir_path + "/" + filename + "-" + str(count) + ".mp4"

            os.system(cmd)

            cur_start += cut_len
            count += 1

        cmd = "rm " + dir_path + "/" + video_list[i]
        os.system(cmd)

    # cut_len 길이면 그대로 사용
    elif video_len[i] == cut_len:
        pass
    # cut_len 이하 영상들 삭제
    else:
        cmd = "rm " + dir_path + "/" + video_list[i]
        os.system(cmd)
