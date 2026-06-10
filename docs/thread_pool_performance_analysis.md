# 线程池模式性能问题分析报告

> 日期: 2026-06-09  
> 平台: OrangePi 5 Plus (RK3588)  
> 程序: `yolov8_fall_detect_tp` (线程池加速版)

---

## 一、问题描述

### 问题1：NPU负载不均衡
```
NPU load: Core0: 70%, Core1: 5%, Core2: 0%
```
3个推理线程启动后，NPU负载严重不均：Core0承担了绝大部分工作（70%），Core1几乎空闲（5%），Core2完全空闲（0%）。线程池虽然启动了3个线程，但实际只有1个NPU核心在高效工作。

### 问题2：FPS差异（同一USB摄像头，不同拍摄内容）
| 场景 | FPS | 说明 |
|------|-----|------|
| 摄像头对准手机播放的视频 | ~26 FPS | 画面简洁，单人，亮度恒定 |
| 摄像头拍摄真实环境 | ~17 FPS | 画面复杂，可能多人，光线变化 |

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

当前代码使用 `flag=0`（`RKNN_NPU_CORE_AUTO`），RKNN驱动会**随机**将3个context分配到NPU核心。

**实际运行数据证实了负载不均**：
```
NPU load: Core0: 70%, Core1: 5%, Core2: 0%
```

这说明 RKNN 驱动的 `AUTO` 模式将大部分推理任务分配到了 Core0，Core1 偶尔参与，Core2 完全空闲。3个线程的推理实际上**主要由1个NPU核心串行处理**，线程池的并行效果大打折扣。

**关键证据**：
1. NPU负载数据：Core0(70%) >> Core1(5%) >> Core2(0%)，严重不均
2. 日志中 `Queue:0` 表明任务队列始终为空，说明推理速度勉强跟得上输入
3. 真实环境FPS仅17，远低于3线程并行应有的25+

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

### 问题2根因：USB摄像头拍摄手机视频(~26FPS) vs 实时场景(~17FPS)

> **场景澄清**：两个场景都使用USB摄像头，区别在于摄像头拍摄的内容：
> - **场景A**：USB摄像头对准手机屏幕，手机播放跌倒检测视频 → FPS ~26
> - **场景B**：USB摄像头拍摄真实环境 → FPS ~17

**FPS差距的贡献因素分析：**

#### 因素A：画面内容复杂度影响绘制和编码耗时（主要因素）

两个场景的**推理管线耗时相同**（预处理+NPU推理+后处理），差异在**结果线程**：

| 处理阶段 | 手机视频场景 | 实时场景 | 差异原因 |
|----------|-------------|---------|----------|
| 检测结果 | 通常1人，背景简洁 | 可能多人，背景复杂 | NMS+绘制耗时不同 |
| 关键点绘制 | 1人骨架，线条少 | 多人骨架，线条多 | `DrawCocoKps` 耗时不同 |
| 跌倒状态 | 状态稳定，绘制简单 | 状态频繁变化 | `DrawFallResult` 耗时略增 |
| MJPEG编码 | 简洁画面，JPEG小 | 复杂画面，JPEG大 | 编码+网络传输耗时不同 |

**关键证据**：日志中 `Queue:0` 表明任务队列始终为空，说明**推理管线不是瓶颈**，瓶颈在结果线程的绘制+编码环节。

#### 因素B：RGA互斥锁串行化（已由方案B解决）

上一轮修复中添加的 `g_rga_mutex` 将所有RGA操作串行化，降低了整体吞吐量。此因素对两个场景的影响相同，已通过方案B（移除互斥锁+降级策略）解决。

#### 因素C：摄像头自动曝光差异

| 场景 | 画面亮度 | 自动曝光行为 | 帧率影响 |
|------|----------|-------------|----------|
| 手机屏幕 | 恒定亮度，无闪烁 | 曝光稳定，帧间隔均匀 | 无额外延迟 |
| 真实环境 | 光线变化，运动模糊 | 曝光频繁调整，可能丢帧 | 偶发帧间隔增大 |

手机屏幕的亮度恒定，摄像头的3A（自动曝光/白平衡/对焦）算法运行稳定，帧间隔均匀。真实环境中光线变化会导致摄像头调整曝光时间，某些帧的采集时间变长。

---

## 三、问题关联性分析

两个问题**相互独立**，但共同影响FPS：

```
问题1: NPU未绑定独立核心 → 3个线程竞争同一NPU → 推理串行化 → FPS基础值低
问题2: RGA互斥锁串行化 → 预处理不能并行 → 进一步降低FPS

两个问题叠加:
  NPU串行 + RGA串行 → 整个管线串行化 → 真实环境仅17FPS

FPS差异（26 vs 17）的原因:
  手机视频: 画面简洁 → 绘制快 + JPEG编码快 → 结果线程不阻塞 → 26FPS
  真实环境: 画面复杂 → 绘制慢 + JPEG编码慢 → 结果线程成瓶颈 → 17FPS
```

方案A（NPU绑定）+ 方案B（RGA降级）共同解决管线串行化问题，预期真实环境FPS提升到22-25。

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

| 指标 | 修复前（AUTO模式） | 方案A+B实施后（预期） |
|------|-------------------|----------------------|
| NPU Core0 负载 | 70%（过载） | ~30-35%（均衡） |
| NPU Core1 负载 | 5%（几乎空闲） | ~30-35%（均衡） |
| NPU Core2 负载 | 0%（完全空闲） | ~30-35%（均衡） |
| 手机视频场景FPS | ~26 | ~28-30（RGA并行提升） |
| 真实环境场景FPS | ~17 | ~22-25（NPU三核并行+RGA并行） |

**FPS差距分析结论**：
- 手机视频 vs 真实环境的FPS差距（26 vs 17）主要由**画面内容复杂度**决定
- 推理管线（预处理+NPU推理+后处理）的耗时在两个场景中基本相同
- 差异来自结果线程的**绘制复杂度**和**MJPEG编码耗时**
- 这是正常现象，不是程序bug

---

## 六、已执行方案

| 方案 | 状态 | 说明 |
|------|------|------|
| 方案A：NPU核心绑定 | ✅ 已完成 | 3个线程分别绑定NPU Core 0/1/2 |
| 方案B：RGA降级策略 | ✅ 已完成 | 移除互斥锁，RGA失败自动降级OpenCV |
| 方案C：摄像头优化 | ⏸ 可选 | 边际收益较小 |

详细实现文档：
- 方案A：[`docs/npu_core_binding_impl.md`](docs/npu_core_binding_impl.md)
- 方案B：[`docs/rga_fallback_strategy_impl.md`](docs/rga_fallback_strategy_impl.md)
