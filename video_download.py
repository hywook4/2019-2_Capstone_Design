from pytube import YouTube
import csv

dataset_file = "data_url.csv"

url_list = []

with open(dataset_file, 'r') as f:
    reader = csv.reader(f)
    csv_list = list(reader)

for u in csv_list:
    if u[1]=='':
        pass
    else:
        url_list.append(u)


url_list = url_list[991:]
save_path = "./test/"

check_list = []

for a in url_list:
    url = a[1]
    
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4', resolution='360p').all()

    info = []
    info.append(a[0])
    info.append(url)

    if len(stream) == 0:
        info.append('x')
        info.append(None)
        info.append(None)
        check_list.append(info)
        print("failed : ", info)
    else:
        info.append('o') 
        info.append(stream[0].video_codec)
        info.append(stream[0].audio_codec)
        check_list.append(info)
        
        
        stream[0].download(save_path)

        print("success : ", info)
        
           

    

print("==================")
for a in check_list:
    print(a)

