# YOLOv8-Pose 跌倒检测系统 - 项目全面分析与面试准备指南

> 平台：OrangePi 5 Plus (RK3588)  
> 模型：YOLOv8-Pose INT8 量化  
> 性能：25 FPS，NPU三核并行，MJPEG实时推流  
> 预计学习时间：1-2天

---

## 一、项目一句话描述（面试开场白）

> "我在RK3588嵌入式平台上，基于YOLOv8-Pose模型实现了一个实时人体跌倒检测系统。通过RKNN API调用NPU三核并行推理，结合RGA硬件加速预处理和MJPEG HTTP推流，在640×480摄像头输入下实现了25FPS的实时检测。项目涉及多线程池架构设计、NPU核心绑定优化、RGA容错降级策略、基于关键点的跌倒状态机算法等。"

---

## 二、项目架构总览

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────┐
│                    yolov8_fall_detect_tp.cpp (主程序)         │
│                    线程编排 + 信号处理 + 参数解析              │
├──────────┬───────────────────┬───────────────────────────────┤
│ 读取线程  │   推理线程池(3)    │        结果线程               │
│ USB摄像头 │   每线程独立模型    │  跌倒检测+绘制+MJPEG推流      │
│ 640×480  │   绑定NPU Core 0/1/2│                              │
├──────────┴───────────────────┴───────────────────────────────┤
│                    Yolov8ThreadPool (线程池管理)              │
│          任务队列 + 结果队列 + 帧跳过 + 内存清理              │
├──────────────────────────────────────────────────────────────┤
│                    Yolov8Custom (模型封装)                    │
│          预处理(RGA/OpenCV) → NPU推理 → 后处理(NMS)          │
├──────────────┬───────────────┬───────────────────────────────┤
│   RKEngine   │  preprocess   │       postprocess             │
│  RKNN API    │  RGA/OpenCV   │    解码+NMS+关键点提取         │
├──────────────┼───────────────┼───────────────────────────────┤
│  FallDetector│   cv_draw     │       MjpegServer             │
│  跌倒状态机   │  检测框+骨架   │    HTTP MJPEG推流             │
└──────────────┴───────────────┴───────────────────────────────┘
```

### 2.2 模块代码地图

| 模块 | 文件 | 代码行数 | 核心职责 |
|------|------|---------|---------|
| **主程序** | `src/yolov8_fall_detect_tp.cpp` | ~370行 | 线程编排、参数解析、信号处理 |
| **线程池** | `src/task/yolov8_thread_pool.h/.cpp` | ~200行 | 任务队列、结果管理、帧跳过 |
| **模型封装** | `src/task/yolov8_custom.h/.cpp` | ~270行 | 预处理→推理→后处理全流程 |
| **NN引擎** | `src/engine/engine.h` (接口) + `rknn_engine.h/.cpp` (实现) | ~230行 | RKNN API封装 |
| **预处理** | `src/process/preprocess.h/.cpp` | ~190行 | letterbox + resize (RGA/OpenCV) |
| **后处理** | `src/process/postprocess.h/.cpp` | ~495行 | 解码 + NMS + 关键点提取 |
| **跌倒检测** | `src/task/fall_detector.h/.cpp` | ~520行 | 6特征状态机算法 |
| **绘制** | `src/draw/cv_draw.h/.cpp` + `fall_draw.h/.cpp` | ~260行 | 检测框+骨架+跌倒状态绘制 |
| **推流** | `src/stream/mjpeg_server.h/.cpp` | ~440行 | HTTP MJPEG推流服务器 |
| **数据类型** | `src/types/*.h` | ~180行 | 张量、检测、跌倒等数据结构 |

---

## 三、模块详解（按面试提问频率排序）

### 模块1：多线程池架构 ⭐⭐⭐⭐⭐（必问）

**面试问法**："你的多线程是怎么设计的？为什么用线程池？"

**核心设计**：
```
4类线程，3种队列：
1. 读取线程：从USB摄像头读帧 → 提交到任务队列
2. 工作线程(3)：从任务队列取帧 → 预处理+推理+后处理 → 存入结果队列
3. 结果线程：从结果队列取结果 → 跌倒检测+绘制+MJPEG推流
4. MJPEG服务器线程：HTTP accept + 每客户端一个线程
```

**关键代码**：
- [`submitTask()`](src/task/yolov8_thread_pool.cpp:88)：生产者提交任务，队列满时阻塞
- [`worker()`](src/task/yolov8_thread_pool.cpp:54)：消费者处理任务，condition_variable等待
- [`getTargetResultFull()`](src/task/yolov8_thread_pool.cpp:120)：按帧ID获取结果，超时机制

**要掌握的C++知识**：
- `std::mutex` + `std::condition_variable` 实现生产者-消费者模型
- `std::unique_lock` vs `std::lock_guard` 的区别
- `std::atomic` 用于无锁标志（`g_stop`、`g_read_finished`）
- `std::shared_ptr` 管理模型实例生命周期

**面试加分点**：
- 提到帧跳过优化（结果线程落后时跳到最新帧）
- 提到任务队列大小限制（防止内存溢出）
- 提到超时机制（防止死锁）

---

### 模块2：NPU核心绑定 ⭐⭐⭐⭐⭐（必问）

**面试问法**："你怎么利用RK3588的三核NPU？"

**核心设计**：
```
RK3588有3个NPU核心，通过rknn_set_core_mask()显式绑定：
  线程0 → RKNN_NPU_CORE_0 (mask=1)
  线程1 → RKNN_NPU_CORE_1 (mask=2)
  线程2 → RKNN_NPU_CORE_2 (mask=4)
```

**优化前 vs 优化后**：
```
优化前：Core0: 70%, Core1: 5%, Core2: 0%  (AUTO模式，负载严重不均)
优化后：Core0: 25%, Core1: 23%, Core2: 25% (显式绑定，负载均衡)
```

**要掌握的知识**：
- `rknn_init()` 的 flag 参数含义
- `rknn_set_core_mask()` API 的使用时机（必须在init之后）
- `RKNN_NPU_CORE_AUTO` vs 显式绑定的区别
- 分层设计：接口层(engine.h) → 实现层(rknn_engine.cpp) → 封装层(yolov8_custom.h) → 使用层(thread_pool)

**面试加分点**：
- 提到AUTO模式的不可靠性（实测数据证明）
- 提到分层架构设计（SetCoreMask通过虚函数层层透传）

---

### 模块3：预处理与RGA硬件加速 ⭐⭐⭐⭐

**面试问法**："你的预处理是怎么做的？为什么用RGA？"

**预处理流程**：
```
原始帧(640×480 BGR) → letterbox(加边框成640×640) → resize(640×640) → BGR2RGB → memcpy到tensor
```

**RGA vs OpenCV**：
| 操作 | OpenCV (CPU) | RGA (硬件) |
|------|-------------|-----------|
| letterbox | cv::copyMakeBorder ~5ms | immakeBorder ~2ms |
| resize | cv::resize ~8ms | imresize ~2ms |
| 颜色转换 | cv::cvtColor ~3ms | RGA内联处理 ~0ms |

**容错降级策略**：
```
优先RGA硬件加速 → 失败？→ 自动降级到OpenCV → 继续推理不中断
```

**要掌握的知识**：
- letterbox的作用（保持宽高比，填充黑边）
- RGA是Rockchip的2D图形加速器，独立于CPU/NPU
- `wrapbuffer_virtualaddr()` 将虚拟地址包装为RGA buffer
- `imcheck()` 检查参数合法性，`imresize()`/`immakeBorder()` 执行操作
- 容错降级的设计模式

---

### 模块4：后处理与NMS ⭐⭐⭐⭐

**面试问法**："YOLOv8的输出是怎么解码的？"

**YOLOv8-Pose输出结构**（9个tensor）：
```
3个检测头 × (reg + cls + pose)：
  reg1: [1, 1, 4, 6400]  → 边框回归 (80×80特征图)
  cls1: [1, 1, 80, 80]   → 类别置信度
  ps1:  [1, 51, 80, 80]  → 17个关键点×3(x,y,score)
  reg2/cls2/ps2: 40×40特征图
  reg3/cls3/ps3: 20×20特征图
```

**解码流程**：
```
1. 生成meshgrid（每个特征图位置的中心坐标）
2. 对每个位置：
   - cls_val = sigmoid(dequant(cls)) → 类别置信度
   - if cls_val > threshold:
     - xmin = (cx - dequant(reg[0])) × stride
     - ymin = (cy - dequant(reg[1])) × stride
     - xmax = (cx + dequant(reg[2])) × stride
     - ymax = (cy + dequant(reg[3])) × stride
     - 17个关键点 = dequant(pose) × 2 + offset
3. NMS去重（IOU > 0.45 的抑制）
```

**要掌握的知识**：
- INT8反量化公式：`float_val = (int8_val - zero_point) × scale`
- sigmoid函数的作用（将logits转为概率）
- NMS算法原理（按置信度排序，贪心抑制重叠框）
- letterbox逆变换（将坐标映射回原始图像）

---

### 模块5：跌倒检测算法 ⭐⭐⭐⭐

**面试问法**："你的跌倒检测是怎么做的？用了哪些特征？"

**6个特征**：
| 特征 | 计算方法 | 物理含义 |
|------|---------|---------|
| 躯干角度 | 肩膀中点→髋部中点连线与垂直方向夹角 | 身体倾斜程度 |
| 宽高比 | bbox宽度/bbox高度 | 跌倒时身体变宽 |
| 头脚距离 | 头部y-脚部y（归一化） | 跌倒时头脚接近同一水平线 |
| 重心高度 | 关键点y坐标均值 | 跌倒时重心下降 |
| 肩膀倾斜 | 左右肩连线与水平方向夹角 | 身体侧倾 |
| 身体展平度 | 关键点y坐标标准差 | 跌倒时身体水平展平 |

**状态机**：
```
STANDING ──(连续N帧score>阈值)──→ FALLING ──(连续N帧确认)──→ FALLEN(报警)
    ↑                                                            │
    └──────────(连续M帧score<阈值)────────── RECOVERING ←────────┘
```

**综合得分**：
```
fall_score = w1×normalize(trunk_angle) + w2×normalize(bbox_ratio) + 
             w3×normalize(head_foot_dist) + w4×normalize(center_of_gravity) + 
             w5×normalize(shoulder_tilt) + w6×normalize(body_flatness)
```

---

### 模块6：MJPEG推流服务器 ⭐⭐⭐

**面试问法**："你的推流是怎么实现的？"

**架构**：
```
MjpegServer (HTTP服务器)
├── ServerThread: accept客户端连接
├── ClientThread(每个客户端): 发送MJPEG流
│   └── 每帧: --boundary\r\n + Content-Length + JPEG数据
└── PushFrame(): 编码JPEG + 更新帧缓冲（非阻塞）
```

**MJPEG协议**：
```
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=--boundary

--boundary
Content-Type: image/jpeg
Content-Length: 12345

<JPEG数据>
--boundary
Content-Type: image/jpeg
Content-Length: 12346

<JPEG数据>
...
```

**要掌握的知识**：
- MJPEG vs H.264的区别（MJPEG无帧间压缩，简单但带宽大）
- `cv::imencode(".jpg", frame, buf, params)` 的JPEG编码
- HTTP chunked transfer encoding
- 多客户端连接管理（每个客户端独立线程）

---

### 模块7：RKNN引擎封装 ⭐⭐⭐

**面试问法**："你是怎么封装RKNN API的？"

**分层设计**：
```
NNEngine (抽象接口，纯虚函数)
    └── RKEngine (RKNN实现)
            ├── LoadModelFile() → rknn_init + rknn_query
            ├── SetCoreMask() → rknn_set_core_mask
            ├── Run() → rknn_inputs_set + rknn_run + rknn_outputs_get
            └── ~RKEngine() → rknn_destroy
```

**设计模式**：
- **策略模式**：NNEngine接口可以有不同的实现（RKNN、TensorRT、ONNX Runtime）
- **工厂模式**：`CreateRKNNEngine()` 创建具体实现
- **RAII**：析构函数自动释放rknn_context

---

## 四、面试高频问题清单

### 架构设计类
1. 为什么用线程池而不是单线程？→ 利用RK3588三核NPU并行推理
2. 线程间怎么通信？→ mutex + condition_variable + 生产者-消费者队列
3. 怎么处理线程池中的任务积压？→ 帧跳过机制 + 队列大小限制
4. 为什么用3个线程？→ 对应RK3588的3个NPU核心

### 性能优化类
5. 瓶颈在哪里？→ 结果线程的绘制+MJPEG编码（不是NPU推理）
6. NPU核心绑定怎么做？→ rknn_set_core_mask()，AUTO模式负载不均
7. RGA是什么？→ Rockchip 2D图形加速器，做resize/color convert比CPU快
8. RGA失败怎么办？→ 自动降级到OpenCV，不影响功能
9. FPS从17提升到25做了什么？→ NPU核心绑定 + RGA并行 + 帧跳过 + MJPEG优化

### 算法类
10. YOLOv8的输出怎么解码？→ meshgrid + 反量化 + sigmoid + NMS
11. 跌倒检测用了哪些特征？→ 6个：躯干角度、宽高比、头脚距离、重心高度、肩倾角、展平度
12. 为什么用状态机？→ 避免单帧误判，需要连续N帧确认
13. 关键点置信度怎么用？→ 低于阈值的关键点不参与特征计算

### 工程类
14. 怎么处理摄像头断开？→ 重试机制（30次×100ms）+ 多后端降级
15. 怎么推流到浏览器？→ HTTP MJPEG，浏览器直接用 `<img src="/stream">`
16. 模型量化有什么要注意的？→ INT8需要zero_point和scale做反量化
17. 内存管理怎么做？→ shared_ptr管理模型实例，手动free tensor数据

---

## 五、1-2天学习计划

### Day 1：理解架构和核心流程

**上午（3小时）**：
1. 读 `src/yolov8_fall_detect_tp.cpp` 主程序，理解4类线程的创建和协作
2. 读 `src/task/yolov8_thread_pool.h/.cpp`，理解生产者-消费者模型
3. 画出完整的数据流图

**下午（3小时）**：
1. 读 `src/task/yolov8_custom.cpp` 的 `Run()` 方法，理解预处理→推理→后处理流程
2. 读 `src/process/preprocess.cpp`，理解letterbox和resize
3. 读 `src/process/postprocess.cpp`，理解YOLOv8解码和NMS

**晚上（2小时）**：
1. 读 `src/task/fall_detector.cpp`，理解6特征跌倒检测算法
2. 读 `src/types/fall_datatype.h`，理解状态机和配置参数
3. 总结：用自己的话描述整个系统的工作流程

### Day 2：深入细节和面试准备

**上午（3小时）**：
1. 读 `src/engine/rknn_engine.cpp`，理解RKNN API调用流程
2. 读 `src/stream/mjpeg_server.cpp`，理解HTTP推流实现
3. 回顾所有优化文档（`docs/` 目录），理解每个优化的动机和效果

**下午（3小时）**：
1. 对照面试问题清单，逐一准备口头回答
2. 画架构图（白板上能画出来）
3. 准备3个"亮点故事"：
   - NPU核心绑定优化（从70%/5%/0%到25%/23%/25%）
   - RGA容错降级策略（移除互斥锁+自动fallback）
   - 帧跳过优化（结果线程不再积压）

**晚上（2小时）**：
1. 模拟面试：让朋友问你上面的问题
2. 整理不熟悉的知识点，重点复习
3. 准备项目亮点的量化数据（FPS提升、NPU利用率提升等）

---

## 六、项目亮点总结（简历/面试用）

### 量化成果
- 实现25FPS实时跌倒检测（640×480输入，YOLOv8-Pose INT8）
- NPU三核并行推理，利用率从单核70%均衡到三核各25%
- FPS从初始17提升到25（+47%），通过NPU绑定+RGA优化+帧跳过
- 支持MJPEG HTTP推流，浏览器实时查看检测结果

### 技术亮点
1. **多线程池架构**：3个推理线程绑定独立NPU核心，生产者-消费者模型
2. **NPU核心绑定**：通过rknn_set_core_mask()显式绑定，解决AUTO模式负载不均
3. **RGA容错降级**：移除互斥锁恢复并行，RGA失败自动降级到OpenCV
4. **帧跳过机制**：结果线程落后时跳到最新帧，防止积压
5. **跌倒状态机**：6特征加权+连续帧确认，避免单帧误判
6. **分层架构设计**：引擎接口抽象，支持不同后端（RKNN/TensorRT）

### 可扩展方向
- 增加多人跟踪（DeepSORT）
- 增加行为识别（跌倒→坐下→站立的细粒度分类）
- 增加告警推送（WebSocket/短信/微信）
- 部署到边缘计算集群（多摄像头）
