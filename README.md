# ESP32-CAM 石头剪刀布识别

用手机自拍的少量视频，训练一个 3 类手势 CNN，量化成 **114 KB** 的 int8 TFLite，
部署到 ESP32-CAM 上实时识别。整套流程在本机完成，不依赖 Edge Impulse。

```
素材视频 → 5fps 抽帧 → 人工时间段标签 → 96×96 CNN → int8 TFLite → model.h → ESP32-CAM
```

## 数据

| 项 | 数值 |
| --- | --- |
| 视频 | 8 段，约 63 秒 |
| 帧（5fps） | 470 帧 + 1 张照片 |
| 类别 | paper 185 / rock 108 / scissors 178 |
| 背景 | 木纹门、灰墙、白墙（3 种） |
| 袖子 | 浅蓝、深绿、灰色 |

标签不是按整段视频给的——同一个人在同一段视频里会连续换手势，
所以 `make_dataset.py` 里的 `SEGMENTS` 按秒写死区间，`build_labels.py` 按
`t=(帧号-1)/5` 落区间，离边界 0.15 秒内的帧丢掉。

## 模型

- 输入 96×96×3，卷积 32/64/128 + GAP + Dense64 + Dense3
- 训练增强：水平翻转、±36° 旋转、缩放、平移、随机 90° 旋转（摄像头可能横装）、
  亮度/对比度/饱和度抖动
- 类别权重按 `sqrt(N/(3n))` 计算，上限 2.5
- int8 量化，TFLite **114,648 字节**

## 结果

**帧级随机划分（80/20）**——这是同源验证，数字偏乐观：

| | 准确率 |
| --- | --- |
| float | 96.8% |
| int8 | **96.8%（92/95）** |

混淆矩阵（行=真实，列=预测，paper/rock/scissors）：

```
[[36  0  1]
 [ 0 22  0]
 [ 2  0 34]]
```

**按视频留一法（跨背景）**——每次留一整段视频出来测试，其余训练：

| 留出的视频 | 手势 | 帧数 | 准确率 |
| --- | --- | --- | --- |
| 32777379 | paper | 110 | 61.8% |
| 359b5063 | scissors | 50 | 90.0% |
| a928747a | scissors | 61 | 0.0% |
| c0cfc796 | paper | 69 | 46.4% |
| **平均** | | | **49.5%** |

这两个数字差得很大，而且是真的：**模型学到的是背景和袖子，不只是手势**。
留出 `a928747a` 时训练集里没有"白墙 + 绿袖子 + 剪刀"的组合，
它就全判成了 paper。随机划分里帧与帧只差零点几秒，所以显得很准。

`demo_predictions.jpg` 是同一批验证帧上的预测示例，12 张里错 1 张。

## 已知限制

- 背景和袖子与类别相关，跨场景泛化弱（见上面的留一结果）
- 数据来自同一个人、同一部手机
- 验证帧与训练帧同源，帧级准确率不代表实际使用

想让它真的能用，下一步不是调模型，是补数据：

1. 每个手势换 3 个以上背景（桌面、白墙、地板、窗边）
2. 距离 30cm / 50cm / 80cm 各拍一段
3. 换人拍一段（至少 1 个人）
4. 补完重跑下面三步

## 目录

| 路径 | 内容 |
| --- | --- |
| `data/dataset.csv` | 带标签的帧清单 |
| `make_dataset.py` | ffmpeg 5fps 抽帧 + `SEGMENTS` 时间段表 |
| `build_labels.py` | 按时间段生成 `dataset.csv` |
| `train_gesture.py` | 训练、int8 量化、导出 `model.h` |
| `group_holdout.py` | 按视频留一法评估 |
| `model/gesture_model.tflite` | 量化模型（114,648 字节） |
| `model/model.h` | Arduino PROGMEM 数组 |
| `sketch/gesture_cam/` | ESP32-CAM 推理程序 |
| `烧录指南.md` | 接线、Arduino 配置、常见问题 |

## 重新训练

```bat
cd /d D:\esp32-gesture
.venv\Scripts\python.exe make_dataset.py
.venv\Scripts\python.exe build_labels.py
.venv\Scripts\python.exe train_gesture.py
```

补拍后要在 `make_dataset.py` 的 `SRC` / `SEGMENTS` 里加新视频和对应时间段。

## 部署

1. Arduino IDE 装库 **TensorFlowLite_ESP32**（tanakamasayuki）。
2. 开发板选 AI Thinker ESP32-CAM，Partition Scheme 选 Huge APP，**PSRAM 打开**。
3. 打开 `sketch/gesture_cam/gesture_cam.ino` 烧录，串口 115200 看输出：

```
rock  paper=0.01 rock=0.97 scissors=0.02
```

## 环境

- Python 3.12 + TensorFlow 2.21 CPU（`.venv`）
- imageio-ffmpeg 提供 ffmpeg 二进制；pip 走清华镜像
- 没在真板上编译验证过（本机没有 Arduino/ESP32 工具链），第一次烧录可能要微调
