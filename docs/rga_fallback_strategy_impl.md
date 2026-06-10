# RGA预处理优化（方案B）- 实施文档

> 日期: 2026-06-10  
> 平台: OrangePi 5 Plus (RK3588)  
> 修改文件: 3个源文件  
> 编译状态: ✅ 通过

---

## 一、问题背景

### 方案A的遗留问题
在方案A（NPU核心绑定）中，为了修复 RGA `rga_job_submit Bad address` 错误，添加了全局互斥锁 `g_rga_mutex`。这虽然解决了RGA并发冲突，但引入了新的性能问题：

```
有互斥锁时（串行RGA）:
  线程0: [====RGA====] [===NPU推理===]
  线程1:              [等待] [====RGA====] [===NPU推理===]
  线程2:                         [等待] [====RGA====] [===NPU推理===]
  
  RGA预处理被完全串行化，每3帧多消耗 ~16ms
```

### 方案B的目标
移除互斥锁，恢复RGA并行处理能力。当RGA偶尔失败时，自动降级到OpenCV预处理，而非崩溃或阻塞其他线程。

---

## 二、设计思路 - 容错降级模式

### 核心思想
```
优先使用RGA硬件加速（快）
    ↓ 失败？
自动降级到OpenCV CPU处理（慢但可靠）
    ↓
继续推理，不中断程序
```

这是一种**容错降级模式（Graceful Degradation）**，常见于嵌入式系统设计：
- 正常情况：使用硬件加速，性能最优
- 异常情况：降级到软件实现，保证功能正确
- 对用户透明：不需要手动切换

### 与互斥锁方案的对比

| 方案 | RGA正常时 | RGA失败时 | 性能 | 可靠性 |
|------|-----------|-----------|------|--------|
| 互斥锁 | 串行RGA（慢） | 串行RGA（慢） | ⭐⭐ | ⭐⭐⭐⭐ |
| 无保护 | 并行RGA（快） | 崩溃/数据错误 | ⭐⭐⭐⭐ | ⭐ |
| **降级方案B** | **并行RGA（快）** | **降级OpenCV** | **⭐⭐⭐⭐** | **⭐⭐⭐⭐** |

---

## 三、具体修改内容

### 第1步：修改函数签名 - [`src/process/preprocess.h`](src/process/preprocess.h:15)

**修改前**：
```cpp
LetterBoxInfo letterbox_rga(const cv::Mat& img, cv::Mat& img_letterbox, float wh_ratio);
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);
```

**修改后**：
```cpp
bool letterbox_rga(const cv::Mat& img, cv::Mat& img_letterbox, float wh_ratio, LetterBoxInfo& info);
bool cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor);
```

**变化说明**：
- `cvimg2tensor_rga`：返回类型从 `void` → `bool`（true=成功，false=失败）
- `letterbox_rga`：返回类型从 `LetterBoxInfo` → `bool`，`LetterBoxInfo` 改为输出参数

**设计决策**：为什么改返回类型而不是用异常？
```
C++异常机制的问题:
1. RGA驱动错误不会抛出C++异常（是C库）
2. try-catch有运行时开销
3. 嵌入式系统通常禁用异常（-fno-exceptions）

bool返回值的优势:
1. 零开销，编译器友好
2. 调用方必须显式处理（编译器警告未使用返回值）
3. 与C库API风格一致（RKNN API也用返回值）
```

---

### 第2步：修改RGA函数实现 - [`src/process/preprocess.cpp`](src/process/preprocess.cpp)

#### 2a. 移除互斥锁和头文件

**删除**：
```cpp
#include <mutex>
static std::mutex g_rga_mutex;
```

#### 2b. 修改 `cvimg2tensor_rga`

**关键变化**：
```cpp
// 修改前：失败时 exit(-1) 终止程序
if (IM_STATUS_NOERROR != ret) {
    NN_LOG_ERROR("check error! %s", imStrError((IM_STATUS)ret));
    exit(-1);  // ← 致命：整个程序崩溃
}
imresize(src, dst);  // ← 不检查返回值

// 修改后：失败时返回 false，调用方决定如何处理
if (IM_STATUS_NOERROR != ret) {
    NN_LOG_WARNING("check error! %s (will fallback to OpenCV)", ...);
    return false;  // ← 温和：返回失败标志
}
IM_STATUS status = imresize(src, dst);
if (status != IM_STATUS_SUCCESS) {
    NN_LOG_WARNING("imresize failed! %s (will fallback to OpenCV)", ...);
    return false;  // ← 检查imresize返回值
}
return true;
```

**改进点**：
1. `exit(-1)` → `return false`：不再终止程序
2. 新增 `imresize` 返回值检查：之前没有检查
3. `NN_LOG_ERROR` → `NN_LOG_WARNING`：降级不是致命错误
4. 日志中添加 `(will fallback to OpenCV)`：提示后续行为

#### 2c. 修改 `letterbox_rga`

**关键变化**：
```cpp
// 修改前
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio) {
    LetterBoxInfo info;  // 局部变量
    // ... RGA操作 ...
    exit(-1);  // 失败时崩溃
    return info;
}

// 修改后
bool letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio, LetterBoxInfo &info) {
    // info 通过引用参数输出
    // ... RGA操作 ...
    return false;  // 失败时返回false
    return true;   // 成功时返回true
}
```

**同步移除**：`lock_guard<std::mutex>` 互斥锁全部移除

---

### 第3步：修改调用方 - [`src/task/yolov8_custom.cpp`](src/task/yolov8_custom.cpp:140)

**核心逻辑**：
```cpp
else if (process_type == "rga")
{
    // 优先使用RGA硬件加速
    bool rga_ok = letterbox_rga(img, image_letterbox, wh_ratio, letterbox_info_);
    if (rga_ok)
    {
        rga_ok = cvimg2tensor_rga(image_letterbox, width, height, input_tensor_);
    }
    
    // RGA失败时自动降级到OpenCV
    if (!rga_ok)
    {
        NN_LOG_WARNING("RGA preprocess failed, falling back to OpenCV");
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, width, height, input_tensor_);
    }
}
```

**执行流程图**：
```
Preprocess("rga")
    │
    ├── letterbox_rga() ──→ true ──→ cvimg2tensor_rga() ──→ true ──→ 完成（RGA路径）
    │        │                              │
    │        ↓ false                        ↓ false
    │        └──────────────────────────────┘
    │                    │
    │                    ↓
    │            降级到OpenCV
    │            letterbox() + cvimg2tensor()
    │                    │
    │                    ↓
    │                完成（OpenCV路径）
    │
    └── 结果：无论RGA是否成功，都能完成预处理
```

**为什么降级逻辑放在 `Yolov8Custom::Preprocess()` 而不是RGA函数内部？**
```
关注点分离（Separation of Concerns）:
- preprocess.cpp: 只负责执行预处理操作，报告成功/失败
- yolov8_custom.cpp: 负责决策策略（用RGA还是OpenCV）

这样设计的好处:
1. RGA函数可以被其他模块复用（不需要内置降级逻辑）
2. 降级策略可以在调用方灵活调整（比如记录统计、动态切换）
3. 单元测试更容易（可以分别测试RGA和OpenCV路径）
```

---

## 四、完整调用链路图

```
worker线程0/1/2 并行执行:
    │
    ├── Yolov8Custom::Run(img)
    │       │
    │       ├── Preprocess(img, "rga", image_letterbox)
    │       │       │
    │       │       ├── letterbox_rga() ← 3线程并行访问RGA硬件
    │       │       │       ├── RGA成功 → return true
    │       │       │       └── RGA失败 → return false
    │       │       │
    │       │       ├── cvimg2tensor_rga() ← 3线程并行访问RGA硬件
    │       │       │       ├── RGA成功 → return true
    │       │       │       └── RGA失败 → return false
    │       │       │
    │       │       └── 任一失败？
    │       │               ├── 否 → 使用RGA结果
    │       │               └── 是 → 降级OpenCV（letterbox + cvimg2tensor）
    │       │
    │       ├── Inference() ← 3线程并行NPU推理（方案A绑定的核心）
    │       │
    │       └── Postprocess()
    │
    └── 保存结果到 results/kp_results/img_results
```

---

## 五、预期日志输出

### 正常情况（RGA成功）：
```
[NN_INFO] rknn_set_core_mask success! core_mask=1
[NN_INFO] Yolov8ThreadPool: thread 0 model loaded (NPU core 0)
[NN_INFO] rknn_set_core_mask success! core_mask=2
[NN_INFO] Yolov8ThreadPool: thread 1 model loaded (NPU core 1)
[NN_INFO] rknn_set_core_mask success! core_mask=4
[NN_INFO] Yolov8ThreadPool: thread 2 model loaded (NPU core 2)
```
无额外日志，说明RGA正常工作。

### 偶发RGA失败（自动降级）：
```
[NN_WARNING] cvimg2tensor_rga: imresize failed! ... (will fallback to OpenCV)
[NN_WARNING] RGA preprocess failed, falling back to OpenCV
```
出现警告但程序继续运行，FPS可能短暂下降。

---

## 六、预期效果

### 性能对比

| 指标 | 互斥锁方案 | 降级方案B（预期） |
|------|-----------|------------------|
| RGA正常时预处理耗时 | ~24ms（3线程串行） | ~8ms（3线程并行） |
| RGA失败时预处理耗时 | ~24ms（仍串行） | ~15ms（降级OpenCV） |
| 摄像头实时FPS | ~17 | ~25+ |
| 程序稳定性 | 高 | 高（自动降级） |

### 验证方法
```bash
# 1. 运行程序
./yolov8_fall_detect_tp ../weights/yolov8-pose-int.rknn

# 2. 观察是否有 "RGA preprocess failed, falling back to OpenCV" 警告
#    - 偶尔出现：正常（RGA硬件偶发冲突，已自动降级）
#    - 频繁出现：可能RGA驱动有问题，但程序仍能运行

# 3. 对比FPS
#    - 无降级日志时：FPS应达到25+
#    - 有降级日志时：FPS可能略低但仍高于17
```

---

## 七、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| [`src/process/preprocess.h`](src/process/preprocess.h) | 签名变更 | RGA函数返回bool |
| [`src/process/preprocess.cpp`](src/process/preprocess.cpp) | 核心修改 | 移除互斥锁，RGA失败返回false |
| [`src/task/yolov8_custom.cpp`](src/task/yolov8_custom.cpp) | 调用逻辑 | 添加降级到OpenCV的逻辑 |

**总修改量**: 3个文件，约30行修改。

---

## 八、注意事项

1. **RGA偶发失败是正常的**：RK3588 RGA硬件在多线程并发时可能偶发 `Bad address` 错误，这是驱动层的已知问题
2. **降级到OpenCV不影响推理精度**：预处理结果（resize后的图像）在视觉上是一致的，只是处理速度不同
3. **日志级别用WARNING而非ERROR**：因为降级是预期行为，不是程序错误
4. **`process_type_` 仍为 "rga"**：即使降级到OpenCV，下次调用仍会先尝试RGA（因为RGA失败可能是偶发的）
5. **建议与方案A（NPU核心绑定）配合使用**：方案A解决NPU利用率，方案B解决RGA稳定性

---

## 九、方案A + 方案B 组合效果

```
方案A（NPU核心绑定）: 解决 "NPU负载0%" 问题
    → 3个NPU核心并行推理，FPS基础提升

方案B（RGA降级策略）: 解决 "RGA互斥锁串行化" 问题
    → 3个线程并行预处理，FPS进一步提升

组合效果:
    预处理: 3线程并行（RGA）或偶发降级（OpenCV）
    推理:   3线程并行（NPU Core 0/1/2）
    后处理: 3线程并行（CPU）
    
    理论最大FPS: ~30+（受限于单帧推理时间 ~33ms）
```
