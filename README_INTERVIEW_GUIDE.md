# YOLOv8-Pose 跌倒检测系统 - 面试知识全攻略

> 本文档基于项目所有源码，系统性整理每个模块的核心知识点，按面试深度要求分级。
> 每个模块包含：**核心概念 → 代码实现 → 面试高频问题 → 回答要点**。

---

## 目录

1. [项目整体架构](#一项目整体架构)
2. [OpenCV/RGA 预处理](#二opencvrga-预处理)
3. [RKNN NPU 推理引擎](#三rknn-npu-推理引擎)
4. [后处理（检测框+关键点解码）](#四后处理检测框关键点解码)
5. [人体跌倒检测算法](#五人体跌倒检测算法)
6. [多线程编程](#六多线程编程)
7. [视频输入/输出模块](#七视频输入输出模块)
8. [MJPEG 网络推流](#八mjpeg-网络推流)
9. [结果可视化](#九结果可视化)
10. [异常处理与日志](#十异常处理与日志)
11. [内存管理](#十一内存管理)
12. [CMakeLists 构建系统](#十二cmakelists-构建系统)
13. [模型训练与转换](#十三模型训练与转换)
14. [面试高频问题汇总](#十四面试高频问题汇总)

---

## 一、项目整体架构

### 1.1 架构分层图

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application)                        │
│  yolov8_fall_detect_usb.cpp / yolov8_fall_detect_tp.cpp      │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐              │
│  │ 读流线程  │──→│ 推理线程/线程池│──→│ 结果线程  │              │
│  └──────────┘   └──────────────┘   └──────────┘              │
├─────────────────────────────────────────────────────────────┤
│                    任务层 (Task)                               │
│  Yolov8Custom: 预处理→推理→后处理                              │
│  FallDetector: 6特征提取+状态机                                │
│  Yolov8ThreadPool: 多NPU核心并行推理                           │
├─────────────────────────────────────────────────────────────┤
│                    引擎层 (Engine)                             │
│  NNEngine (抽象接口) → RKEngine (RKNN实现)                     │
├─────────────────────────────────────────────────────────────┤
│                    处理层 (Process)                             │
│  Preprocess: letterbox + resize (OpenCV/RGA)                  │
│  Postprocess: 检测框解码 + 17关键点解码 + NMS                   │
├─────────────────────────────────────────────────────────────┤
│                    硬件加速层 (Hardware)                        │
│  RKNN NPU (6TOPS)  │  RGA (预处理加速)  │  OpenCV (I/O)        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 数据流

```
USB摄像头 → cv::VideoCapture → cv::Mat (BGR 640×480)
    → letterbox (RGA/OpenCV) → 640×640 填充图像
    → BGR2RGB → memcpy/tensor → INT8 输入张量
    → RKNN NPU 推理 → 9个输出张量 (3×reg + 3×cls + 3×pose)
    → 反量化 + 检测框解码 + 关键点解码 + NMS
    → vector<Detection> + vector<map<int, KeyPoint>>
    → FallDetector::Update() → FallResult
    → 绘制 + MJPEG推流 + 录像
```

### 1.3 面试回答要点

**Q: 介绍一下你的项目架构？**

> 项目采用分层架构，从上到下分为应用层、任务层、引擎层、处理层和硬件加速层。应用层负责多线程流水线调度，任务层封装了 YOLOv8 推理和跌倒检测逻辑，引擎层抽象了 RKNN NPU 接口（方便后续替换为其他后端），处理层实现了 letterbox 预处理和后处理解码，硬件加速层利用 RK3588 的 NPU（推理）和 RGA（预处理）实现端到端加速。

---

## 二、OpenCV/RGA 预处理

### 2.1 核心概念

预处理的目标：将输入图像转换为模型需要的格式。

```
输入图像 (BGR, 任意尺寸)
    → letterbox (保持宽高比填充到正方形)
    → resize (缩放到 640×640)
    → BGR → RGB 转换
    → 内存拷贝到输入张量
```

### 2.2 Letterbox 原理

```cpp
// src/process/preprocess.cpp
// 计算 letterbox 尺寸
float wh_ratio = 640.0f / 640.0f;  // 模型输入宽高比
if (img_width / img_height > wh_ratio) {
    // 图片更宽，上下填充
    letterbox_width = img_width;
    letterbox_height = img_width / wh_ratio;
    padding_ver = (letterbox_height - img_height) / 2;
} else {
    // 图片更高，左右填充
    letterbox_width = img_height * wh_ratio;
    letterbox_height = img_height;
    padding_hor = (letterbox_width - img_width) / 2;
}
```

**面试要点**：Letterbox 保持原始宽高比，用黑色像素填充空白区域，避免图像变形导致检测精度下降。

### 2.3 OpenCV vs RGA 预处理

| 特性 | OpenCV 版 | RGA 硬件加速版 |
|------|-----------|---------------|
| 实现 | `cv::resize()` + `cv::cvtColor()` | `imresize()` + `immakeBorder()` |
| 执行位置 | CPU | RGA 硬件（独立于 CPU/NPU） |
| 耗时 | ~8-15ms | ~2-5ms |
| 适用场景 | 任何平台 | 仅 RK3568/RK3588 |

### 2.4 关键代码

```cpp
// RGA 版本 letterbox
rga_buffer_t src = wrapbuffer_virtualaddr(
    (void*)img.data, img.cols, img.rows, RK_FORMAT_RGB_888);
rga_buffer_t dst = wrapbuffer_virtualaddr(
    (void*)img_letterbox.data, letterbox_width, letterbox_height, RK_FORMAT_RGB_888);
immakeBorder(src, dst, padding_ver, padding_ver, padding_hor, padding_hor, 0, 0, 0);

// RGA 版本 resize + 写入 tensor
rga_buffer_t dst = wrapbuffer_virtualaddr(
    (void*)tensor.data, width, height, RK_FORMAT_RGB_888);
imresize(src, dst);
```

### 2.5 面试高频问题

**Q: 为什么需要 Letterbox？直接 resize 不行吗？**

> 直接 `cv::resize()` 会改变图像的宽高比，导致人体变形（变胖或变瘦），影响关键点检测精度。Letterbox 先按比例缩放，再用黑色像素填充到目标尺寸，保持原始比例不变。

**Q: RGA 加速的原理是什么？**

> RGA（Rockchip Graphics Acceleration）是瑞芯微的 2D 图形加速器，独立于 CPU 和 NPU。它通过 DMA 直接在内存上执行 resize、色彩转换、格式转换等操作，不占用 CPU 资源。在 RK3588 上，RGA 可以将预处理耗时从 10ms 降到 3ms。

**Q: BGR 和 RGB 的区别？为什么要转换？**

> OpenCV 默认使用 BGR 通道顺序，而大多数深度学习模型（包括 YOLOv8）训练时使用 RGB 顺序。如果输入顺序错误，模型输出会完全错误。转换只需要 `cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB)`。

---

## 三、RKNN NPU 推理引擎

### 3.1 核心概念

RKNN（Rockchip Neural Network）是瑞芯微的 NPU 推理框架，支持 INT8/FP16 量化模型。

### 3.2 推理流程

```cpp
// src/engine/rknn_engine.cpp

// 1. 加载模型
rknn_init(&rknn_ctx, model_data, model_len, 0, NULL);

// 2. 查询输入输出信息
rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));

// 3. 设置输入
rknn_input rknn_inputs[1];
rknn_inputs[0].type = RKNN_TENSOR_UINT8;
rknn_inputs[0].fmt = RKNN_TENSOR_NHWC;
rknn_inputs[0].buf = input_data;
rknn_inputs_set(ctx, 1, rknn_inputs);

// 4. 运行推理
rknn_run(ctx, nullptr);

// 5. 获取输出
rknn_output outputs[9];
rknn_outputs_get(ctx, 9, outputs, NULL);

// 6. 释放
rknn_outputs_release(ctx, 9, outputs);
rknn_destroy(ctx);
```

### 3.3 引擎抽象层设计

```cpp
// src/engine/engine.h - 抽象接口
class NNEngine {
public:
    virtual ~NNEngine() {}
    virtual nn_error_e LoadModelFile(const char* model_file) = 0;
    virtual const std::vector<tensor_attr_s>& GetInputShapes() = 0;
    virtual const std::vector<tensor_attr_s>& GetOutputShapes() = 0;
    virtual nn_error_e Run(std::vector<tensor_data_s>& inputs,
                           std::vector<tensor_data_s>& outputs,
                           bool want_float) = 0;
};

// src/engine/rknn_engine.h - RKNN 实现
class RKEngine : public NNEngine {
    // 实现所有纯虚函数
};

// 工厂函数
std::shared_ptr<NNEngine> CreateRKNNEngine();
```

### 3.4 面试高频问题

**Q: 为什么要设计引擎抽象层？**

> 策略模式的应用。通过抽象接口 `NNEngine`，上层代码不依赖具体的推理后端。如果需要从 RKNN 迁移到其他 NPU（如地平线、寒武纪），只需新增一个 `Engine` 子类，不需要修改上层代码。这也是**开闭原则**的体现。

**Q: INT8 量化是什么？有什么优缺点？**

> INT8 量化将模型权重和激活值从 FP32（32位浮点）压缩到 INT8（8位整数），模型大小减少 4 倍，推理速度提升 2-4 倍（NPU 对 INT8 有硬件加速）。缺点是精度会有 1-3% 的下降。量化公式：`real_value = (int8_value - zero_point) × scale`。

**Q: NPU 的三个核心怎么利用？**

> RK3588 有 3 个 NPU 核心，每个核心可以独立运行一个 RKNN context。通过创建 3 个 `Yolov8Custom` 实例（每个调用 `rknn_init` 创建独立 context），RKNN 运行时会自动将不同 context 分配到不同核心，实现真正的并行推理。

---

## 四、后处理（检测框+关键点解码）

### 4.1 YOLOv8 输出结构

```
模型输出：9个张量
├── reg1: [1, 1, 4, 6400]   ← stride=8 的回归（框坐标偏移）
├── cls1: [1, 1, 80, 80]    ← stride=8 的分类（类别置信度）
├── reg2: [1, 1, 4, 1600]   ← stride=16 的回归
├── cls2: [1, 1, 40, 40]    ← stride=16 的分类
├── reg3: [1, 1, 4, 400]    ← stride=32 的回归
├── cls3: [1, 1, 20, 20]    ← stride=32 的分类
├── ps1:  [1, 51, 80, 80]   ← stride=8 的关键点 (51=17×3)
├── ps2:  [1, 51, 40, 40]   ← stride=16 的关键点
└── ps3:  [1, 51, 20, 20]   ← stride=32 的关键点
```

### 4.2 解码流程

```cpp
// 1. 生成网格坐标（meshgrid）
// 对于 stride=8 的特征图 (80×80)，每个网格中心坐标为 (j+0.5, i+0.5)

// 2. 检测框解码
xmin = (grid_x - reg[0]) * stride
ymin = (grid_y - reg[1]) * stride
xmax = (grid_x + reg[2]) * stride
ymax = (grid_y + reg[3]) * stride

// 3. 关键点解码（17个COCO关键点）
for (int kc = 0; kc < 17; kc++) {
    kp.x = (pose[kc*3+0] * 2 + (grid_x - 0.5)) * stride / input_w;
    kp.y = (pose[kc*3+1] * 2 + (grid_y - 0.5)) * stride / input_h;
    kp.score = sigmoid(pose[kc*3+2]);
}

// 4. NMS（非极大值抑制）
// 按置信度排序，IOU > 阈值的重叠框被抑制
```

### 4.3 INT8 反量化

```cpp
static float DeQnt2F32(int8_t qnt, int zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}
```

### 4.4 面试高频问题

**Q: YOLOv8 的 Anchor-Free 机制和 YOLOv5 有什么区别？**

> YOLOv5 使用预定义的 anchor 框，检测时预测相对于 anchor 的偏移。YOLOv8 去掉了 anchor，直接预测框的 4 个边界距离（ltrb：left, top, right, bottom），每个特征图位置就是一个候选点。这样减少了超参数，也更适合不同尺度的目标。

**Q: NMS 的原理是什么？为什么需要它？**

> 一个目标可能被多个检测框检出（相邻网格都检测到了同一个人）。NMS 按置信度排序，依次取出最高分的框，删除与它 IOU 超过阈值（如 0.45）的其他框。这确保每个目标只保留一个最佳检测框。

**Q: 17 个 COCO 关键点分别是什么？**

> 鼻子(0)、左眼(1)、右眼(2)、左耳(3)、右耳(4)、左肩(5)、右肩(6)、左肘(7)、右肘(8)、左腕(9)、右腕(10)、左髋(11)、右髋(12)、左膝(13)、右膝(14)、左踝(15)、右踝(16)。

---

## 五、人体跌倒检测算法

### 5.1 6 维特征提取

| 特征 | 计算方式 | 站立值 | 跌倒值 | 权重 |
|------|----------|--------|--------|------|
| 躯干角度 | `atan2(\|肩中x-髋中x\|, \|肩中y-髋中y\|)` | ~0° | >60° | 0.30 |
| 身体宽高比 | `bbox宽 / bbox高` | <1.0 | >1.2 | 0.20 |
| 头脚距离 | `\|头y - 脚y\|`（归一化） | ~1.0 | ~0.3 | 0.20 |
| 重心高度 | `mean(所有关键点y)` | ~0.4 | ~0.7 | 0.15 |
| 肩膀倾斜 | `atan2(\|左肩y-右肩y\|, \|左肩x-右肩x\|)` | ~0° | >45° | 0.10 |
| 身体展平 | `std(所有关键点y)` | >0.2 | <0.1 | 0.05 |

### 5.2 得分融合

```cpp
float fall_score = 0.30 * angle_score +     // 躯干角度
                   0.20 * ratio_score +     // 宽高比
                   0.20 * dist_score +      // 头脚距离
                   0.15 * gravity_score +   // 重心高度
                   0.10 * tilt_score +      // 肩膀倾斜
                   0.05 * flatness_score;   // 身体展平
```

### 5.3 状态机

```
STANDING ──(连续3帧score>0.55)──→ FALLING ──(达到3帧)──→ FALLEN (报警!)
    ↑                                                        │
    └──(连续8帧score<0.55)── RECOVERING ←─────────────────────┘
```

### 5.4 面试高频问题

**Q: 为什么用多特征融合而不是单一特征？**

> 单一特征容易误判。例如蹲下时躯干角度也会增大，但宽高比和头脚距离不会像跌倒那样变化。多特征互补可以降低误报率：只有多个特征同时异常时才判定为跌倒。

**Q: 状态机的作用是什么？**

> 状态机实现**时序平滑**，过滤短暂的异常姿态。比如人弯腰捡东西只有 1-2 帧，不会触发跌倒确认（需要连续 3 帧）。恢复也需要连续 8 帧正常，避免反复报警。

**Q: 如果有人被遮挡怎么办？**

> 当前实现的局限：遮挡时关键点缺失，`CalcTrunkAngle` 等函数会返回默认值 0，导致跌倒得分偏低（误判为站立）。改进方向：1) 基于时序的关键点补全；2) 多摄像头融合；3) 引入运动特征（帧间差分）。

---

## 六、多线程编程

### 6.1 项目中的线程模型

#### 方案A：3线程流水线（yolov8_fall_detect_usb）

```
线程1: 读流 → SafeQueue<FrameData>
线程2: 推理 → SafeQueue<ResultData>
线程3: 显示 + MJPEG + 录像
```

#### 方案B：线程池（yolov8_fall_detect_tp）

```
线程1: 读流 → 任务队列
线程池(3): Worker各自独立YOLOv8实例 → 结果Map
线程4: 结果处理 + 跌倒检测 + 绘制 + MJPEG
```

### 6.2 线程安全队列

```cpp
template <typename T>
class SafeQueue {
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cond_;

    void push(T item) {
        std::lock_guard<std::mutex> lock(mtx_);
        while (queue_.size() >= max_size_)
            queue_.pop();  // 丢弃旧帧，保持低延迟
        queue_.push(std::move(item));
        cond_.notify_one();
    }

    bool pop(T& item, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cond_.wait_for(lock, chrono::milliseconds(timeout_ms),
                            [this]{ return !queue_.empty() || g_stop; }))
            return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
};
```

### 6.3 面试高频问题

**Q: mutex、condition_variable、atomic 分别用在什么场景？**

> - **mutex**：保护共享数据的读写（如帧缓冲区、结果Map），保证同一时刻只有一个线程访问
> - **condition_variable**：线程间等待/通知（如队列为空时消费者等待，生产者push后通知）
> - **atomic**：无锁的简单变量操作（如停止标志`g_stop`、客户端计数`client_count_`），比mutex轻量

**Q: 生产者-消费者模式在你项目中怎么用的？**

> 读流线程是生产者，不断往 SafeQueue 推入帧数据；推理线程是消费者，从队列取帧推理。队列设 max_size=2，满了就丢弃旧帧（保持低延迟）。condition_variable 让消费者在队列为空时休眠，不浪费 CPU。

**Q: 线程池有什么好处？**

> 1) 避免频繁创建/销毁线程的开销；2) 控制并发数，防止线程过多导致 CPU 争抢；3) RK3588 有 3 个 NPU 核心，创建 3 个 YOLOv8 实例分别绑定，实现真正的并行推理，吞吐量提升 3 倍。

---

## 七、视频输入/输出模块

### 7.1 视频读取

```cpp
cv::VideoCapture cap(0);  // USB摄像头
cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // 最小缓冲，降低延迟

cv::Mat frame;
cap >> frame;  // 读取一帧
```

### 7.2 视频写入

```cpp
int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
cv::VideoWriter writer("output.mp4", fourcc, 25, cv::Size(640, 480));
writer << frame;
```

### 7.3 面试高频问题

**Q: `CAP_PROP_BUFFERSIZE` 设为 1 有什么作用？**

> 默认缓冲区大小为 1-3 帧。设为 1 意味着 OpenCV 只缓冲最新一帧，读取时拿到的是最新的摄像头画面，降低端到端延迟。如果不设置，可能会读到几帧前的旧画面。

**Q: VideoWriter 的编解码器怎么选？**

> `mp4v` 是 MPEG-4 编码，兼容性好但压缩率一般。`XVID` 是另一种选择。在嵌入式设备上，如果 FFmpeg 支持硬件编码（如 RK3588 的 MPP），可以用 `H264` 获得更好的压缩率和性能。

---

## 八、MJPEG 网络推流

### 8.1 MJPEG 协议原理

```
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg
Content-Length: 12345

[JPEG数据]
--frame
Content-Type: image/jpeg
Content-Length: 12346

[JPEG数据]
...
```

浏览器原生支持，`<img src="/stream">` 即可显示。

### 8.2 核心实现

```cpp
// 编码：OpenCV → JPEG
std::vector<uchar> buf;
cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 75});

// 推流：HTTP multipart
std::string header = "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
send(client_fd, header.c_str(), header.size(), 0);

// 每帧发送
send(client_fd, "--frame\r\nContent-Type: image/jpeg\r\n\r\n", ...);
send(client_fd, jpeg_data.data(), jpeg_data.size(), 0);
send(client_fd, "\r\n", 2, 0);
```

### 8.3 面试高频问题

**Q: MJPEG 和 H.264 推流有什么区别？**

> MJPEG 每帧独立压缩（帧内编码），实现简单，浏览器原生支持，但带宽占用大（~200KB/帧）。H.264 利用帧间相似性压缩（帧间编码），带宽小（~20KB/帧），但需要硬件编码器，浏览器需要 MSE/WASM 支持。在嵌入式监控场景，MJPEG 的低延迟和简单实现是优势。

**Q: 为什么用 HTTP 而不是 WebSocket？**

> HTTP MJPEG 是最简单的方案，`<img src="/stream">` 一行代码就能显示。WebSocket 需要额外的握手协议和帧格式处理。对于单向视频流，HTTP MJPEG 足够且更简单。

---

## 九、结果可视化

### 9.1 绘制层次

```
1. 检测框 (cv::rectangle) + 类别标签 (cv::putText)
2. 关键点 (cv::circle, 按部位着色)
3. 骨架 (cv::line, 按部位着色)
4. 跌倒状态 (右上角)
5. 调试面板 (左下角, 6个特征值)
6. FPS (左上角)
7. 报警框 (跌倒时红色全屏边框)
```

### 9.2 半透明背景实现

```cpp
cv::Mat overlay = img.clone();
cv::rectangle(overlay, rect, cv::Scalar(0, 0, 0), -1);
cv::addWeighted(overlay, 0.6, img, 0.4, 0, img);
```

### 9.3 面试要点

**Q: `cv::addWeighted` 的公式是什么？**

> `dst = α × src1 + β × src2 + γ`。用于实现半透明效果：`α=0.6, β=0.4` 表示 overlay 占 60%，原图占 40%。

---

## 十、异常处理与日志

### 10.1 错误码体系

```cpp
typedef enum {
    NN_SUCCESS = 0,
    NN_LOAD_MODEL_FAIL = -1,
    NN_RKNN_INIT_FAIL = -2,
    NN_RKNN_QUERY_FAIL = -3,
    NN_RKNN_INPUT_SET_FAIL = -4,
    NN_RKNN_RUNTIME_ERROR = -5,
    NN_IO_NUM_NOT_MATCH = -6,
    NN_RKNN_OUTPUT_GET_FAIL = -7,
    NN_RKNN_INPUT_ATTR_ERROR = -8,
    NN_RKNN_OUTPUT_ATTR_ERROR = -9,
    NN_RKNN_MODEL_NOT_LOAD = -10,
    NN_TIMEOUT = -12,
} nn_error_e;
```

### 10.2 日志系统

```cpp
#define NN_LOG_ERROR(...)   do { if (g_log_level >= 1) { printf("[NN_ERROR] "); printf(__VA_ARGS__); printf("\n"); } } while(0)
#define NN_LOG_WARNING(...) do { if (g_log_level >= 2) { printf("[NN_WARNING] "); printf(__VA_ARGS__); printf("\n"); } } while(0)
#define NN_LOG_INFO(...)    do { if (g_log_level >= 3) { printf("[NN_INFO] "); printf(__VA_ARGS__); printf("\n"); } } while(0)
#define NN_LOG_DEBUG(...)   do { if (g_log_level >= 4) { printf("[NN_DEBUG] "); printf(__VA_ARGS__); printf("\n"); } } while(0)
```

### 10.3 信号处理

```cpp
signal(SIGINT, signalHandler);   // Ctrl+C
signal(SIGTERM, signalHandler);  // kill 命令

void signalHandler(int signum) {
    g_stop = true;  // 优雅退出，所有线程检查此标志后退出
}
```

### 10.4 面试要点

**Q: 为什么用枚举错误码而不是异常？**

> 嵌入式系统中异常处理开销大（栈展开、RTTI），且 NPU 驱动层可能不支持异常。错误码方式轻量、可控，每一层都可以检查返回值并决定是否继续。

---

## 十一、内存管理

### 11.1 关键内存操作

```cpp
// 模型文件加载（手动 malloc/free）
unsigned char* model = (unsigned char*)malloc(model_len);
fread(model, 1, model_len, fp);
// 使用完后 free(model)

// 输入/输出张量（手动 malloc/free）
input_tensor_.data = malloc(input_tensor_.attr.size);
// 析构函数中 free(input_tensor_.data)

// RKNN 输出（框架分配，用户释放）
rknn_outputs_get(ctx, output_num_, outputs, NULL);
memcpy(data.data, output.buf, output.size);
free(output.buf);  // 必须释放
```

### 11.2 面试要点

**Q: 为什么不用智能指针管理张量内存？**

> 张量内存需要传递给 NPU 驱动（RKNN API），驱动可能对内存地址有对齐要求。`malloc` 返回的地址满足系统对齐要求，而智能指针的自定义删除器会增加复杂度。在嵌入式场景，手动管理更直接可控。

---

## 十二、CMakeLists 构建系统

### 12.1 核心结构

```cmake
# 1. 基本设置
cmake_minimum_required(VERSION 3.11)
project(rk3588-demo)
set(CMAKE_CXX_STANDARD 14)

# 2. 依赖库路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
find_package(OpenCV REQUIRED)

# 3. 头文件搜索路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 4. 构建共享库
add_library(yolov8_lib SHARED src/task/yolov8_custom.cpp)
target_link_libraries(yolov8_lib rknn_engine nn_process)

# 5. 构建可执行文件
add_executable(yolov8_fall_detect_tp
    src/yolov8_fall_detect_tp.cpp
    src/task/yolov8_thread_pool.cpp)
target_link_libraries(yolov8_fall_detect_tp
    draw_lib fall_draw_lib fall_detector_lib mjpeg_stream_lib yolov8_lib pthread)
```

### 12.2 面试要点

**Q: CMake 中 SHARED 和 STATIC 库的区别？**

> SHARED（.so）是动态链接，运行时加载，多个程序可以共享同一份库代码，节省内存。STATIC（.a）是静态链接，编译时合并到可执行文件中，不依赖外部库文件。嵌入式部署通常用 SHARED 方便更新。

**Q: `target_link_libraries` 和 `link_libraries` 的区别？**

> `target_link_libraries` 是现代 CMake 的推荐方式，只对指定 target 生效。`link_libraries` 是全局的，影响之后所有 target。推荐用前者，更精确可控。

> **面试建议**：CMake 部分不需要深入，**懂得整体流程即可**（设置版本→找依赖→include→add_library→add_executable→target_link_libraries）。能说清楚项目怎么编译、依赖哪些库就够了。

---

## 十三、模型训练与转换

### 13.1 训练流程（RTX 4070 上完成）

```python
# 1. 安装 ultralytics
pip install ultralytics

# 2. 训练 YOLOv8-Pose
from ultralytics import YOLO
model = YOLO('yolov8s-pose.pt')  # 预训练模型
model.train(data='coco-pose.yaml', epochs=100, imgsz=640, device=0)

# 3. 微调自定义数据集（如跌倒场景）
model.train(data='fall_pose.yaml', epochs=50, imgsz=640, device=0)
```

### 13.2 模型导出（ONNX）

```python
# 导出 ONNX
model = YOLO('best.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=11)
```

### 13.3 RKNN 转换（rknn-toolkit2）

```python
from rknn.api import RKNN

rknn = RKNN()

# 1. 配置模型
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
    quantized_dtype='asymmetric_quantized-8'  # INT8 量化
)

# 2. 加载 ONNX
ret = rknn.load_onnx(model='yolov8s-pose.onnx')

# 3. 构建（含量化校准）
ret = rknn.build(
    do_quantization=True,
    dataset='calibration_dataset.txt'  # 校准数据集列表
)

# 4. 导出 RKNN 模型
ret = rknn.export_rknn('yolov8s-pose.rknn')

# 5. 精度评估（可选）
ret = rknn.accuracy_analysis(
    inputs=['test_image.jpg'],
    output_dir='accuracy_result'
)
```

### 13.4 量化校准数据集

```
# calibration_dataset.txt
calib_img_001.jpg
calib_img_002.jpg
...
# 需要 100-500 张代表性图片，覆盖各种场景
```

### 13.5 面试高频问题

**Q: 为什么要用 INT8 量化？精度损失大吗？**

> INT8 量化将 FP32 的 4 字节压缩到 1 字节，模型大小减少 4 倍，NPU 推理速度提升 2-4 倍。精度损失通常在 1-3% 以内，通过校准数据集可以最小化损失。在边缘部署场景，速度比精度更重要。

**Q: 量化校准数据集怎么选？**

> 需要 100-500 张**代表性图片**，覆盖实际部署场景的各种情况（不同光照、角度、人数、姿态）。不能用随机图片，否则量化的 scale/zero_point 不准确，精度损失会更大。

**Q: ONNX 简化（simplify）有什么作用？**

> `onnxsim` 会去除冗余的算子（如连续的 Reshape），合并可以融合的层（如 Conv+BN），优化计算图结构。简化后的模型在 RKNN 转换时效率更高，推理速度更快。

**Q: `yolov8-pose-int.rknn` 和 `yolov8s-pose-int.rknn` 有什么区别？**

> `yolov8s-pose` 是 small 版本（约 11M 参数），`yolov8-pose` 可能是 nano 版本（约 3M 参数）。s 版本精度更高但速度更慢，n 版本速度更快但精度略低。选择取决于部署场景的精度/速度需求。

**Q: 模型从 PyTorch 到 RK3588 部署的完整流程是什么？**

> ```
> PyTorch (.pt) → ONNX (.onnx) → RKNN (.rknn)
>     训练          导出+简化      量化+转换
> 
> 1. RTX 4070 上用 ultralytics 训练/微调
> 2. export(format='onnx', simplify=True)
> 3. rknn-toolkit2 配置 + load_onnx + build(量化) + export_rknn
> 4. 将 .rknn 文件拷贝到 RK3588 板子
> 5. 代码中 rknn_init() 加载模型 → rknn_run() 推理
> ```

---

## 十四、面试高频问题汇总

### 14.1 项目介绍类（必问）

| 问题 | 回答要点 |
|------|----------|
| 介绍一下你的项目 | 分层架构 + 6特征跌倒检测 + 三核NPU并行 + MJPEG推流 |
| 你在项目中负责什么？ | 全栈：模型部署、跌倒算法设计、多线程架构、推流服务 |
| 项目的难点是什么？ | NPU多核并行调度、实时性优化、跌倒检测准确率调优 |

### 14.2 技术深度类

| 问题 | 回答要点 |
|------|----------|
| YOLOv8 和 YOLOv5 的区别？ | Anchor-Free、解耦头、Task-Aligned Assigner |
| INT8 量化原理？ | 对称/非对称量化，scale/zero_point，校准数据集 |
| NMS 原理？ | 按置信度排序，IOU抑制，Soft-NMS 改进 |
| Letterbox 作用？ | 保持宽高比填充，避免变形 |
| 多线程同步方式？ | mutex + condition_variable + atomic，适用场景对比 |
| 生产者-消费者模式？ | SafeQueue 实现，丢帧策略，延迟控制 |
| RGA 加速原理？ | 独立硬件单元，DMA操作，不占用CPU/NPU |
| RKNN 多核怎么利用？ | 多context绑定多NPU核心，线程池并行调度 |

### 14.3 工程实践类

| 问题 | 回答要点 |
|------|----------|
| 怎么优化 FPS？ | 多NPU并行、RGA预处理、降低分辨率、模型轻量化 |
| 怎么降低延迟？ | 最小缓冲区、丢帧策略、TCP_NODELAY、JPEG质量调优 |
| 怎么保证线程安全？ | mutex保护共享数据、atomic无锁标志、队列限长 |
| 嵌入式部署注意事项？ | 内存管理、热降频、驱动兼容性、交叉编译 |

### 14.4 算法设计类

| 问题 | 回答要点 |
|------|----------|
| 为什么选这6个特征？ | 覆盖角度/比例/距离/重心/倾斜/展平，互补降误报 |
| 状态机设计思路？ | 4状态时序平滑，帧数确认机制，避免短暂异常误报 |
| 如何处理多人场景？ | 独立状态机跟踪（当前取最高分，可扩展IoU跟踪） |
| 如何处理遮挡？ | 当前局限，可改进：时序补全、多摄像头融合 |

---

## 十五、学习路线建议

### 嵌入式 AI 部署方向

```
基础层：
├── C/C++ 基础（指针、内存、多线程）
├── OpenCV 基础（Mat、图像操作、视频IO）
├── Linux 编程（文件IO、进程/线程、Socket）
└── CMake 构建系统

核心层：
├── 深度学习基础（CNN、Backbone、检测/分割/姿态估计）
├── YOLO 系列原理（v5/v8 架构、Anchor机制、损失函数）
├── 模型量化（INT8/FP16、PTQ/QAT、校准方法）
├── 推理引擎（RKNN/TensorRT/ONNX Runtime）
└── 模型转换（PyTorch→ONNX→RKNN/TRT）

工程层：
├── 多线程编程（线程池、生产者消费者、无锁队列）
├── 性能优化（流水线并行、硬件加速、内存优化）
├── 网络编程（HTTP推流、RTSP、WebSocket）
├── 嵌入式系统（交叉编译、驱动、散热）
└── 工程规范（日志、错误处理、代码组织）
```

### 推荐学习资源

| 方向 | 资源 |
|------|------|
| YOLOv8 原理 | ultralytics 官方文档、YOLOv8 论文 |
| 模型量化 | RKNN-Toolkit2 用户指南、《深度学习模型压缩与加速》 |
| 多线程 | 《C++ Concurrency in Action》、cppreference.com |
| 嵌入式部署 | 瑞芯微开发者文档、Rockchip RKNN SDK |
| OpenCV | OpenCV 官方教程、《Learning OpenCV 4》 |
