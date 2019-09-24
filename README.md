# Matching Network for Music and Video

## Dependency

moviepy
numpy
librosa
matplotlib
ffmpeg

## How to Use

### get_melspectro.py
* Create melspectrogram image for input file(mp4, wav)

python3 get_melspectro.py -fp input_file.mp4 -sp output_file.jpg


> if -sp parameter is not given, the output file name will be same as input except for extension.

### video_clipper.py
* Split videos in directory into clips of given length 

python3 video_clipper.py -dp ./directory_path -cl clip_length


### frame_capture.py
* Capture frames from video and save it to directory



