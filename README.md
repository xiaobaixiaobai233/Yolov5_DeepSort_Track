# 1. Introduction  
This article will introduce how to use YOLOv5 and DeepSort for object detection and tracking, with the addition of trajectory line display. Improvements in this article include matching trajectory line colors with bounding boxes, optimizing trajectory lines to show only a segment, and hiding trajectory lines when objects disappear.  

# 2. Effect Demonstration  
![man-min.gif](https://z4a.net/images/2023/08/25/man-min.gif)  

# 3. How to Use  
### (1) First, download the code:  
`https://github.com/xiaobaixiaobai233/Yolov5_DeepSort_Track.git`  

The YOLOv5 original pre-trained model `yolov5s.pt` and the pedestrian re-identification model `ckpt.t7` are required; file size limits prevent uploading these files.  

### (2) Set up the virtual environment and configurations according to the process.txt  

### (3) Next, modify parameters in the `track_time_power.py` file:  
Update parameters to your own video file path and YOLOv5 pre-trained model path. Set the encoding format to `mp4v` to ensure saved MP4 files are playable. Run with the following command:  

`python track_time_power.py --source ./group_walk3.mp4 --fourcc mp4v`  

Finally, you can view the object detection and tracking results, with each object’s trajectory line matching its bounding box. Trajectory lines disappear when objects vanish, ensuring clearer visualization of each object’s movement path.  

# 4. Reference Code Links  
Special thanks to the author of this code:  
[Deepsort tracking algorithm to draw object motion trajectories](https://blog.csdn.net/qq_35832521/article/details/115124521?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169269914116800222876736%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169269914116800222876736&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-115124521-null-null.142%5Ev93%5EchatgptT3_2&utm_term=deepsort%20%E8%BD%A8%E8%BF%B9&spm=1018.2226.3001.4187)
