# AI_HELMET

基于树莓派 4B 的安全帽识别项目，使用双 ONNX 模型进行本地实时识别。  
A Raspberry Pi 4B safety helmet recognition project using two ONNX models for local real-time detection.

## 功能 / Features

- 使用人体检测模型识别人  
  Detect persons with a person detection model

- 使用安全帽模型识别 `helmet` 和 `no_helmet`  
  Detect `helmet` and `no_helmet` with a helmet detection model

- 根据人体头部区域匹配安全帽  
  Match helmets to persons based on the head region

- 支持树莓派摄像头和普通 OpenCV 摄像头  
  Support Raspberry Pi camera and standard OpenCV cameras

- 本地实时识别并显示检测框  
  Run local real-time recognition and display detection boxes

本版本已移除网页服务、抓拍保存、数据库记录、云端上传和 MQTT 推送，只保留本地识别功能。  
This version removes the web service, snapshot saving, database records, cloud upload, and MQTT push. Only local recognition is kept.

## 模型说明 / Model Notes

- `yolov5n_int8.onnx`: 人体检测模型，使用 YOLOv5 自带的人体检测模型转换而来  
  Person detection model converted from the original YOLOv5 model

- `helmetv2_int8.onnx`: 安全帽检测模型，本地训练数据约 3300 张图片  
  Helmet detection model trained locally with about 3,300 images

如果觉得安全帽识别精度不够，可以自行重新训练安全帽模型，并替换 `helmetv2_int8.onnx`。  
If the helmet recognition accuracy is not good enough, you can train your own helmet model and replace `helmetv2_int8.onnx`.

当前模型已量化为 INT8，推理速度更快，但精度会有一定下降。建议在合适的距离、光照和摄像头角度下进行测试。  
The models are quantized to INT8 for faster inference, but accuracy may decrease. Please test at a suitable distance with proper lighting and camera angle.

## 设备与性能 / Device and Performance

测试设备：  
Test device:

- Raspberry Pi 4B

实际识别帧率大约为 10 FPS，具体速度会受到摄像头分辨率、散热、供电、系统负载和模型参数影响。  
The actual recognition speed is about 10 FPS. The speed may vary depending on camera resolution, cooling, power supply, system load, and model parameters.

## 参数说明 / Parameter Notes

程序中默认最大检测数量为：  
The default maximum detection numbers are:

```python
MAX_DET_PERSON = 3
MAX_DET_HELMET = 5
```

也就是说，默认最多识别 3 个人和 5 个安全帽目标。  
This means the program detects up to 3 persons and 5 helmet targets by default.

如果实际场景中人数或安全帽数量更多，可以在 `server.py` 中自行修改这些参数。其他识别阈值、推理间隔、匹配范围等参数也可以根据实际环境调整。  
If there are more persons or helmets in your scene, you can modify these parameters in `server.py`. Other thresholds, inference intervals, and matching parameters can also be adjusted for your environment.

## 文件说明 / Files

- `server.py`: 主程序 / Main program
- `yolov5n_int8.onnx`: 人体检测模型 / Person detection model
- `helmetv2_int8.onnx`: 安全帽检测模型 / Helmet detection model
- `requirements.txt`: Python 依赖 / Python dependencies

## 安装依赖 / Installation

```bash
pip install -r requirements.txt
```

树莓派使用 CSI 摄像头时，还需要安装并配置 `picamera2`。  
If you use a Raspberry Pi CSI camera, you also need to install and configure `picamera2`.

## 运行 / Run

```bash
python server.py
```

如果没有图形界面，可以关闭本地窗口显示，只在终端输出识别结果：  
If there is no graphical desktop, you can disable the local display window and print recognition results in the terminal only:

```bash
SHOW_LOCAL_WINDOW=0 python server.py
```

## 说明 / Notes

程序默认从 `source=0` 打开摄像头。按 `Esc` 可以退出本地显示窗口。  
The program opens the camera from `source=0` by default. Press `Esc` to exit the local display window.
