# FlashDetect

**低延迟 YOLO26 目标检测引擎推理实现，专为实时视频流优化。仅支持 YOLO26 端到端导出（`[N,6]` 格式，无 NMS）。**

基于 TensorRT + CUDA Graph，全 GPU 流水线。

## 安装

```bash
pip install flashdetect-trt111-cu12x
```

首次 `import flashdetect` 会自动检测 NVIDIA 驱动版本，从 GitHub 下载最适合的原生库。

> 原生库地址：https://github.com/toever/flashdetect/releases/tag/v1.0.0

**手动安装：**

选择对应版本下载安装
```bash
https://github.com/toever/flashdetect/releases/tag/v1.0.0 
```

### 系统要求

- Windows x64 / Linux x64
- NVIDIA GPU (Compute Capability ≥ 7.5)
- Python ≥ 3.9
- **NVIDIA 驱动 ≥ 525**

> 轮子内置对应 CUDA 版本 + TensorRT 11.1 运行时，无需额外安装 CUDA Toolkit。

> **注意**：当前FlashDetect 仅包含推理运行时，请自行构建 engine，后续会增加构建模块

> **注意**：若同时使用 PyTorch 等 GPU 库，请将 `import flashdetect` 放在最前面，避免 CUDA 版本冲突。

### 导出引擎

> 该版本必须用支持 CUDA12.x 的TensorRT 11.1 构建 `.engine` 文件（使用其他版本引擎可能无法加载），且NMS必须为False

## 激活授权

首次使用时需要联系作者获取 license.key 文件，联系方式在最后：

```python
import flashdetect

# 1. 获取本机 ID
print(flashdetect.get_machine_id())

# 2. 将 license.key 放到 flashdetect 包目录下，或使用自动安装：
flashdetect.install_license("license.key路径")
```

## 快速开始

```python
from flashdetect import FlashDetect
import cv2

img = cv2.imread("bus.jpg")
model = FlashDetect("best.engine", device_id=0, bgr2rgb=True, letterbox=True)
result = model.detect(img)

for d in result:
    x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"cls={int(d.class_id)} {d.conf:.2f}"
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("bus_result.jpg", img)
```

## API 参考

### 模块级工具

| 函数 | 说明 |
|------|------|
| `flashdetect.get_machine_id()` | 获取本机硬件 ID（用于申请授权） |
| `flashdetect.install_license(path)` | 安装 license.key 到正确位置 |

### `FlashDetect(engine_path, **kwargs)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `engine_path` | `str` | 必填 | TensorRT .engine 文件路径 |
| `device_id` | `int` | `0` | GPU 设备编号 |
| `bgr2rgb` | `bool` | `True` | 输入为 BGR（OpenCV 默认），GPU 自动转 RGB |
| `letterbox` | `bool` | `True` | 任意尺寸输入，GPU 自动缩放填充 |

> 初始置信度 0.25，检测所有类别。通过 `detect(conf=..., classes=...)` 帧间动态调整。

### `detect(image, conf=None, classes=None) -> List[Detection]`

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | `np.ndarray` | (H, W, 3) uint8 图像 |
| `conf` | `float` | 可选，覆盖当前帧的置信度阈值 |
| `classes` | `List[int]` | 可选，覆盖当前帧的目标类别 |

### `Detection`

| 属性 | 类型 | 说明 |
|------|------|------|
| `x1, y1, x2, y2` | `float` | 边界框坐标（原图像素） |
| `conf` | `float` | 置信度 |
| `class_id` | `float` | 类别 ID（强转 int 使用） |
| `xyxy` | `tuple` | `(x1, y1, x2, y2)` 快捷属性 |

### 其他方法

```python
detector.input_width, detector.input_height   # 引擎输入尺寸
detector.close()                               # 释放 GPU 资源
# 或使用 with 语句自动释放：
# with FlashDetect("model.engine") as detector:
#     dets = detector.detect(frame)
```

## 性能

实测（RTX 4050 Laptop, YOLO26s 128×128 引擎, 256×256 BGR 输入, 1000 图 × 10 次）：

| 指标 | FlashDetect (CUDA Graph) | YOLO 官方 TRT (无 Graph) |
|------|------------------------:|--------------------------:|
| 平均延迟 | **685 us** | 2477 us |
| P50 延迟 | **673 us** | 1898 us |
| P99 延迟 | **855 us** | 3993 us |
| FPS | **~1459** | ~404 |
| 加速比 | **3.6×** | 1× |

CUDA Graph 确保每次推理耗时稳定，无 CPU 端 kernel launch 开销。

## 适用场景

- 游戏画面实时分析（DX 采集）
- 工业相机检测
- 视频流目标跟踪

**不适用**：图片文件 批量 识别（无需 GPU Graph 优化）。

## 许可 & 联系

需要 `license.key` 文件才能运行。无授权时 `FlashDetect()` 会提示本机 ID。

- 📧 邮箱：`2169431623@qq.com`
- 💬 微信：`13709056129`