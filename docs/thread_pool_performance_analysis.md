# 线程池模式性能问题分析报告

> 日期: 2026-06-09  
> 平台: OrangePi 5 Plus (RK3588)  
> 程序: `yolov8_fall_detect_tp` (线程池加速版)

---

## 一、问题描述

### 问题1：NPU未充分利用
```
NPU load: Core0: 0%, Core1: 0%, Core2: 0%
```
3个推理线程启动后，NPU三个核心负载均为0%，线程池未生效。

### 问题2：摄像头实时FPS偏低
| 场景 | FPS | 差距 |
|------|-----|------|
| 手机播放视频 | ~25 FPS | 基准 |
| USB摄像头实时 | ~17 FPS | 降低32% |

---

## 二、根因分析

### 问题1根因：NPU核心未显式绑定

**问题代码位置**：[`rknn_engine.cpp:26`](src/engine/rknn_engine.cpp:26)

```cpp
int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL);
//                                          ^^^
//                                    flag=0 即 RKNN_NPU_CORE_AUTO
```

**分析**：

RKNN API 提供了 [`rknn_set_core_mask()`](librknn_api/include/rknn_api.h:518) 接口用于绑定NPU核心：

| 枚举值 | 含义 |
|--------|------|
| `RKNN_NPU_CORE_AUTO (0)` | 随机分配（当前使用） |
| `RKNN_NPU_CORE_0 (1)` | 绑定核心0 |
| `RKNN_NPU_CORE_1 (2)` | 绑定核心1 |
| `RKNN_NPU_CORE_2 (4)` | 绑定核心2 |
| `RKNN_NPU_CORE_0_1_2 (7)` | 使用全部3个核心 |

当前代码使用 `flag=0`（`RKNN_NPU_CORE_AUTO`），RKNN驱动会**随机**将3个context分配到NPU核心。存在以下可能：

1. **3个context全部被分配到同一个NPU核心** → 只有1个核心工作，其余2个空闲
2. **驱动调度策略导致负载不均** → 部分核心空闲
3. **NPU监控工具读取的是单核利用率** → 如果3个context都分配到Core0，Core1/Core2显示0%是正常的

**关键证据**：日志中 `Queue:0` 表明任务队列始终为空，推理速度跟得上摄像头输入，但FPS只有17，说明**并非3个NPU核心并行推理**，而是串行或仅使用了1个核心。

**解决方案**：

在 [`rknn_engine.cpp`](src/engine/rknn_engine.cpp) 中，`rknn_init` 之后调用 `rknn_set_core_mask()` 显式绑定每个context到独立的NPU核心：

```
线程0 → RKEngine → rknn_ctx_ → rknn_set_core_mask(ctx, RKNN_NPU_CORE_0)
线程1 → RKEngine → rknn_ctx_ → rknn_set_core_mask(ctx, RKNN_NPU_CORE_1)
线程2 → RKEngine → rknn_ctx_ → rknn_set_core_mask(ctx, RKNN_NPU_CORE_2)
```

需要修改的文件：
- [`src/engine/engine.h`](src/engine/engine.h) — `NNEngine` 接口增加 `SetCoreMask()` 虚函数
- [`src/engine/rknn_engine.h`](src/engine/rknn_engine.h) — `RKEngine` 声明 `SetCoreMask()`
- [`src/engine/rknn_engine.cpp`](src/engine/rknn_engine.cpp) — 实现 `SetCoreMask()`，调用 `rknn_set_core_mask()`
- [`src/task/yolov8_custom.h`](src/task/yolov8_custom.h) — `Yolov8Custom` 暴露 `SetCoreMask()`
- [`src/task/yolov8_custom.cpp`](src/task/yolov8_custom.cpp) — 转发调用
- [`src/task/yolov8_thread_pool.cpp`](src/task/yolov8_thread_pool.cpp) — `setUp()` 中为每个线程设置不同核心掩码

---

### 问题2根因：RGA互斥锁串行化预处理 + 摄像头读取延迟

**FPS差距的3个贡献因素：**

#### 因素A：RGA互斥锁串行化（主要因素）

**问题代码位置**：[`preprocess.cpp:12`](src/process/preprocess.cpp:12) + [`preprocess.cpp:101`](src/process/preprocess.cpp:101) + [`preprocess.cpp:157`](src/process/preprocess.cpp:157)

上一轮修复中添加的 `g_rga_mutex` 将所有RGA操作串行化：

```
时间线（有互斥锁）:
  t=0ms   线程0: 锁定RGA → letterbox_rga + cvimg2tensor_rga (~8ms)
  t=8ms   线程0: 解锁RGA → NPU推理 (~25ms)
  t=8ms   线程1: 锁定RGA → letterbox_rga + cvimg2tensor_rga (~8ms)
  t=16ms  线程1: 解锁RGA → NPU推理 (~25ms)
  t=16ms  线程2: 锁定RGA → letterbox_rga + cvimg2tensor_rga (~8ms)
  t=24ms  线程2: 解锁RGA → NPU推理 (~25ms)
  
  3帧总耗时: 24ms(RGA串行) + 25ms(最后1个NPU) = ~49ms
  有效FPS: 1000/49 × 3 ≈ 61 FPS (理论值，实际受其他因素影响)

时间线（无互斥锁）:
  t=0ms   线程0/1/2: 并行RGA (~8ms)
  t=8ms   线程0/1/2: 并行NPU推理 (~25ms)
  
  3帧总耗时: 8ms + 25ms = ~33ms
  有效FPS: 1000/33 × 3 ≈ 90 FPS (理论值)
```

**互斥锁导致RGA预处理从并行变为串行，每3帧多消耗 ~16ms 的RGA等待时间。**

#### 因素B：USB摄像头读取开销

| 数据源 | 读取方式 | 典型延迟 |
|--------|----------|----------|
| 视频文件 | 内存缓冲区读取，几乎无延迟 | <1ms |
| USB摄像头 | USB传输 + V4L2驱动 + 格式转换 | 5-15ms |

USB摄像头的 `cap >> frame` 涉及：
1. USB批量传输（USB 2.0: 640×480×3=921KB，约3ms）
2. V4L2缓冲区管理
3. GStreamer管道处理（日志中可见GStreamer warning）
4. 可能的格式转换（MJPEG→BGR）

**视频文件**读取不经过USB，直接从内存缓冲区取帧，延迟极低。

#### 因素C：MJPG流编码开销

结果线程需要将处理后的帧编码为JPEG并通过HTTP推流。USB摄像头场景下，帧率更高时编码压力更大。但从日志 `Queue:0` 来看，这不是主要瓶颈。

---

## 三、问题关联性分析

两个问题存在**因果关系**：

```
NPU未绑定独立核心 → 3个线程可能竞争同一个NPU → 推理串行化
                                                    ↓
                                              有效FPS下降
                                                    ↓
RGA互斥锁串行化预处理 → 进一步降低FPS → 摄像头场景17FPS
```

如果NPU核心正确绑定（3核并行），即使有RGA互斥锁，FPS也应该能提升到25+。

---

## 四、解决方案

### 方案A：修复NPU核心绑定（优先级最高）

**修改范围**：4个文件

1. **[`src/engine/engine.h`](src/engine/engine.h)** — 接口层增加核心掩码设置
2. **[`src/engine/rknn_engine.cpp`](src/engine/rknn_engine.cpp)** — 实现 `rknn_set_core_mask()` 调用
3. **[`src/task/yolov8_custom.h`](src/task/yolov8_custom.h) / [`.cpp`](src/task/yolov8_custom.cpp)** — 透传核心掩码
4. **[`src/task/yolov8_thread_pool.cpp`](src/task/yolov8_thread_pool.cpp)** — `setUp()` 中为每个线程分配不同核心

**预期效果**：NPU三核并行推理，FPS从17提升到25+

### 方案B：优化RGA互斥锁策略（优先级中）

**当前问题**：全局互斥锁粒度太粗，3个线程的RGA操作完全串行。

**优化选项**：

| 选项 | 方案 | 优点 | 缺点 |
|------|------|------|------|
| B1 | 保留互斥锁，但减小锁粒度 | 安全 | 仍有串行化开销 |
| B2 | 移除互斥锁，添加重试+降级逻辑 | 性能最优 | 代码复杂度增加 |
| B3 | 移除互斥锁，RGA失败时fallback到OpenCV | 平衡 | 需要错误检测 |

**推荐方案B3**：移除互斥锁，在RGA调用失败时自动降级到OpenCV预处理。

### 方案C：摄像头读取优化（优先级低）

1. 设置 `CAP_PROP_FOURCC` 为MJPG减少USB带宽
2. 减小 `CAP_PROP_BUFFERSIZE` 减少缓冲延迟
3. 使用 `V4L2` 直接读取替代GStreamer

---

## 五、预期效果

| 指标 | 当前 | 修复NPU绑定后 | 修复NPU+RGA后 |
|------|------|---------------|---------------|
| NPU Core0 负载 | 0% | ~90% | ~80% |
| NPU Core1 负载 | 0% | ~90% | ~80% |
| NPU Core2 负载 | 0% | ~90% | ~80% |
| 摄像头实时FPS | ~17 | ~25 | ~28-30 |
| 视频播放FPS | ~25 | ~30+ | ~35+ |

---

## 六、建议执行顺序

1. **先修复NPU核心绑定**（方案A）→ 验证FPS是否提升到25+
2. **再优化RGA策略**（方案B）→ 验证FPS是否进一步提升
3. **最后优化摄像头**（方案C，可选）→ 边际收益较小

请审阅后决定下一步执行方案。
