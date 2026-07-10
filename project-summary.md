Project title: Crowd Panic Detection System.

It will be made using OpenCV for object detection. There is a library named Ultralytics YOLOv8, which is popular for the deep‑learning part. PyTorch can be used, although I believe it is automatically installed since it is a dependency of Ultralytics.

If I want to get a server ready instead of just a scratch ML‑only model, I can use FastAPI, and a basic front‑end can also be made.

A useful feature is that we can connect multiple devices using the same room ID, and on the dashboard all of them can be seen. If any camera detects panic or any anomaly in the video feed, the system can send alerts, show an alert on the visual feed, and trigger an alarm if the camera or device supports it. Testing on a phone shows that the phone obviously supports alarm.

The video feed is captured; the source can be anything: a file, webcam, or phone camera. People are detected with bounding boxes, and around them the optical flow can be computed, which is implemented using YOLOv8. YOLOv8 calculates how fast or in which direction every pixel is moving.
The model can be trained on the UMN Unusual Crowd dataset. 

As far as I know, the crowd feed can give us a few features or data such as:

1. Average speed
2. Direction variance
3. Crowd density
4. Flow entropy

These metrics tell us how exactly packed and chaotic the crowd is. A small ML model can simply be trained to output “panic” or “normal.” I believe polling can be used to constantly feed frames to the model and get the result, or maybe WebSockets can be used. All of this basically runs at about 15–30 FPS or higher, depending on system capabilities. Based on my ChatGPT search, this is the approx FPS rate I can expect.

The important frameworks to be used are:

- YOLOv8 – a single neural network that looks at each frame once and outputs a bounding box plus confidence for a person. It basically helps us identify the person. I don’t have much experience with this, but it is pretty easy to use.
- Optical flow – part of OpenCV; it helps calculate pixel movement between two frames using mathematics.

A single frame can be misleading, so we should make decisions based on at least ten consecutive frames, or frames within a short time span of each other.

A few datasets are available, though I haven’t tested them properly. They do contain actual footage:

- The University of Minnesota’s UMN Unusual Crowd Activity dataset.
- A Kaggle dataset that contains footage of Times Square, the Las Vegas shooting, panic situations, parade disasters, and a few others.
- The UCSD Anomaly Detection dataset, which is considered one of the best starting points on the internet for this task.
