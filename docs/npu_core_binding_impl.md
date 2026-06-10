# NPU核心绑定优化 - 实施文档

> 日期: 2026-06-09  
> 平台: OrangePi 5 Plus (RK3588)  
> 修改文件: 5个源文件  
> 编译状态: ✅ 通过

---

## 一、问题背景

### 原始问题
程序启动3个推理线程后，NPU负载严重不均，线程池并行效果大打折扣：
```
NPU load: Core0: 70%, Core1: 5%, Core2: 0%
```
Core0承担了绝大部分推理工作（70%），Core1偶尔参与（5%），Core2完全空闲（0%）。3个线程的推理实际上主要由1个NPU核心串行处理。

### 根因
[`rknn_engine.cpp:26`](src/engine/rknn_engine.cpp:26) 中 `rknn_init()` 使用 `flag=0`（即 `RKNN_NPU_CORE_AUTO`），RKNN驱动将大部分推理任务分配到Core0，导致负载严重不均。

### 关键API
RKNN SDK 提供了 [`rknn_set_core_mask()`](librknn_api/include/rknn_api.h:518) 接口，用于在 `rknn_init()` 之后显式绑定context到指定NPU核心：

```c
// 定义在 librknn_api/include/rknn_api.h:212-221
typedef enum _rknn_core_mask {
    RKNN_NPU_CORE_AUTO = 0,        // 默认，随机分配
    RKNN_NPU_CORE_0 = 1,           // 核心0
    RKNN_NPU_CORE_1 = 2,           // 核心1
    RKNN_NPU_CORE_2 = 4,           // 核心2
    RKNN_NPU_CORE_0_1 = 3,         // 核心0+1
    RKNN_NPU_CORE_0_1_2 = 7,       // 全部3个核心
} rknn_core_mask;

// API签名
int rknn_set_core_mask(rknn_context context, rknn_core_mask core_mask);
```

---

## 二、修改方案 - 分层架构设计

采用**自底向上**的修改策略，遵循项目的分层架构：

```
┌─────────────────────────────────────────────────┐
│  yolov8_thread_pool.cpp  (使用层 - 分配核心)      │  ← 第4步
├─────────────────────────────────────────────────┤
│  yolov8_custom.h         (封装层 - 透传接口)      │  ← 第3步
├─────────────────────────────────────────────────┤
│  rknn_engine.h/cpp       (实现层 - 调用RKNN API) │  ← 第2步
├─────────────────────────────────────────────────┤
│  engine.h                (接口层 - 定义虚函数)    │  ← 第1步
└─────────────────────────────────────────────────┘
```

---

## 三、具体修改内容

### 第1步：接口层 - [`src/engine/engine.h`](src/engine/engine.h:22)

**修改内容**：在 `NNEngine` 抽象基类中添加 `SetCoreMask()` 虚函数。

```cpp
// 新增：第22行之后
virtual nn_error_e SetCoreMask(int core_mask) { return NN_SUCCESS; }
```

**设计要点**：
- 使用**非纯虚函数**（有默认实现），而不是纯虚函数 `= 0`
- 默认实现返回 `NN_SUCCESS`（空操作），这样非RKNN引擎（如未来的TensorRT引擎）不需要实现此方法
- 参数类型为 `int` 而不是 `rknn_core_mask`，保持接口层不依赖RKNN头文件

**C++知识点**：
```
纯虚函数 (= 0): 子类必须实现，否则子类也是抽象类
非纯虚函数: 子类可以选择性重写，基类提供默认行为
这里用非纯虚函数是正确的，因为SetCoreMask是RKNN特有的功能
```

---

### 第2步：实现层 - [`src/engine/rknn_engine.h`](src/engine/rknn_engine.h:23) + [`rknn_engine.cpp`](src/engine/rknn_engine.cpp:182)

**头文件修改**：声明 `override`
```cpp
// rknn_engine.h 第23行
nn_error_e SetCoreMask(int core_mask) override;
```

**实现文件修改**：新增 `SetCoreMask()` 方法
```cpp
// rknn_engine.cpp 第182-197行（新增）
nn_error_e RKEngine::SetCoreMask(int core_mask)
{
    if (!ctx_created_)
    {
        NN_LOG_ERROR("SetCoreMask: rknn context not created yet");
        return NN_RKNN_INIT_FAIL;
    }
    int ret = rknn_set_core_mask(rknn_ctx_, (rknn_core_mask)core_mask);
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_set_core_mask fail! ret=%d, core_mask=%d", ret, core_mask);
        return NN_RKNN_INIT_FAIL;
    }
    NN_LOG_INFO("rknn_set_core_mask success! core_mask=%d", core_mask);
    return NN_SUCCESS;
}
```

**实现要点**：
- **前置检查**：必须在 `rknn_init()` 之后调用（`ctx_created_ == true`），否则返回错误
- **类型转换**：`int` → `rknn_core_mask` 枚举，因为接口层用 `int` 避免依赖RKNN头文件
- **错误处理**：`rknn_set_core_mask()` 返回负值表示失败，需要检查并记录日志

---

### 第3步：封装层 - [`src/task/yolov8_custom.h`](src/task/yolov8_custom.h:28)

**修改内容**：在 `Yolov8Custom` 类中添加 `SetCoreMask()` 方法。

```cpp
// yolov8_custom.h 第28行（新增）
nn_error_e SetCoreMask(int core_mask) { return engine_->SetCoreMask(core_mask); }
```

**设计要点**：
- **内联实现**：直接在头文件中实现，因为只是一行转发调用
- **透传模式**：`Yolov8Custom` 不关心核心掩码的具体含义，只是将调用转发给底层引擎
- **依赖注入**：`engine_` 是 `shared_ptr<NNEngine>`，通过虚函数机制自动调用 `RKEngine::SetCoreMask()`

**架构图**：
```
Yolov8ThreadPool::setUp()
    └── Yolov8Custom::SetCoreMask(core_mask)
            └── NNEngine::SetCoreMask(core_mask)   // 虚函数调用
                    └── RKEngine::SetCoreMask()     // 实际实现
                            └── rknn_set_core_mask() // RKNN SDK API
```

---

### 第4步：使用层 - [`src/task/yolov8_thread_pool.cpp`](src/task/yolov8_thread_pool.cpp:29)

**修改内容**：在 `setUp()` 中为每个线程加载模型后绑定不同的NPU核心。

```cpp
// 新增头文件包含
#include "rknn_api.h"

// setUp() 中新增核心掩码表和绑定逻辑
static const int npu_core_masks[] = {
    RKNN_NPU_CORE_0,   // 线程0 → NPU核心0 (值=1)
    RKNN_NPU_CORE_1,   // 线程1 → NPU核心1 (值=2)
    RKNN_NPU_CORE_2,   // 线程2 → NPU核心2 (值=4)
};

// 在 LoadModel 之后，push_back 之前
if (i < sizeof(npu_core_masks) / sizeof(npu_core_masks[0]))
{
    ret = Yolov8->SetCoreMask(npu_core_masks[i]);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_WARNING("...failed to set core mask, using AUTO", i);
    }
}
```

**执行时序**：
```
setUp() 执行流程:
  1. 创建 Yolov8Custom 实例 (模型类型: NN_YOLOV8_POSE)
  2. LoadModel() → rknn_init() → context创建
  3. SetCoreMask(RKNN_NPU_CORE_0) → rknn_set_core_mask() → 绑定核心0
  4. 重复步骤1-3，线程1绑定核心1，线程2绑定核心2
  5. 创建3个工作线程
```

**安全设计**：
- 使用 `sizeof` 计算数组大小，防止越界访问
- 绑定失败时只打印警告（`NN_LOG_WARNING`），不中断程序，降级为 `RKNN_NPU_CORE_AUTO`
- 日志中打印每个线程绑定的核心编号，方便调试

---

## 四、完整调用链路图

```
main()
 └── Yolov8ThreadPool::setUp(model_path, 3)
      ├── 线程0:
      │    ├── Yolov8Custom(NN_YOLOV8_POSE)
      │    ├── LoadModel("yolov8-pose-int.rknn")
      │    │    └── RKEngine::LoadModelFile()
      │    │         └── rknn_init(&ctx, model, len, 0, NULL)  // 创建context
      │    └── SetCoreMask(RKNN_NPU_CORE_0)                    // 绑定核心0
      │         └── RKEngine::SetCoreMask(1)
      │              └── rknn_set_core_mask(ctx, RKNN_NPU_CORE_0)
      │
      ├── 线程1:
      │    ├── Yolov8Custom(NN_YOLOV8_POSE)
      │    ├── LoadModel("yolov8-pose-int.rknn")
      │    │    └── rknn_init(&ctx, ...)                       // 创建context
      │    └── SetCoreMask(RKNN_NPU_CORE_1)                    // 绑定核心1
      │         └── rknn_set_core_mask(ctx, RKNN_NPU_CORE_1)
      │
      ├── 线程2:
      │    ├── Yolov8Custom(NN_YOLOV8_POSE)
      │    ├── LoadModel("yolov8-pose-int.rknn")
      │    │    └── rknn_init(&ctx, ...)                       // 创建context
      │    └── SetCoreMask(RKNN_NPU_CORE_2)                    // 绑定核心2
      │         └── rknn_set_core_mask(ctx, RKNN_NPU_CORE_2)
      │
      └── 创建3个工作线程 (worker)
```

---

## 五、预期日志输出

修改后的启动日志应包含以下关键信息：

```
[NN_INFO] Initializing thread pool with 3 threads...
[NN_INFO] Yolov8ThreadPool: setting up with 3 threads
[NN_INFO] rknn_init success!
[NN_INFO] rknn_set_core_mask success! core_mask=1      ← 核心0绑定
[NN_INFO] Yolov8ThreadPool: thread 0 model loaded (NPU core 0)
[NN_INFO] rknn_init success!
[NN_INFO] rknn_set_core_mask success! core_mask=2      ← 核心1绑定
[NN_INFO] Yolov8ThreadPool: thread 1 model loaded (NPU core 1)
[NN_INFO] rknn_init success!
[NN_INFO] rknn_set_core_mask success! core_mask=4      ← 核心2绑定
[NN_INFO] Yolov8ThreadPool: thread 2 model loaded (NPU core 2)
[NN_INFO] Yolov8ThreadPool: 3 worker threads started
```

---

## 六、预期效果

### NPU利用率
| 指标 | 修改前（AUTO模式） | 修改后（预期） |
|------|-------------------|----------------|
| NPU Core0 负载 | 70%（过载） | ~30-35%（均衡） |
| NPU Core1 负载 | 5%（几乎空闲） | ~30-35%（均衡） |
| NPU Core2 负载 | 0%（完全空闲） | ~30-35%（均衡） |

### FPS提升
| 场景 | 修改前 | 修改后（预期） |
|------|--------|----------------|
| 手机视频场景 | ~26 FPS | ~28-30 FPS |
| 真实环境场景 | ~17 FPS | ~22-25 FPS |

### 验证方法
```bash
# 1. 运行程序
./yolov8_fall_detect_tp ../weights/yolov8-pose-int.rknn

# 2. 观察启动日志中的 core_mask 信息
# 3. 使用 rknpu 负载监控工具查看NPU利用率
cat /sys/kernel/debug/rknpu/load    # 或使用 rknpu_load 工具
# 预期: Core0/1/2 各有 ~30% 以上负载
```

---

## 七、修改文件清单

| 文件 | 修改类型 | 修改行数 | 说明 |
|------|----------|----------|------|
| [`src/engine/engine.h`](src/engine/engine.h) | 新增虚函数 | +1行 | 接口层定义 |
| [`src/engine/rknn_engine.h`](src/engine/rknn_engine.h) | 新增声明 | +1行 | override声明 |
| [`src/engine/rknn_engine.cpp`](src/engine/rknn_engine.cpp) | 新增方法 | +16行 | rknn_set_core_mask实现 |
| [`src/task/yolov8_custom.h`](src/task/yolov8_custom.h) | 新增内联方法 | +2行 | 透传接口 |
| [`src/task/yolov8_thread_pool.cpp`](src/task/yolov8_thread_pool.cpp) | 修改setUp | +15行 | 核心分配逻辑 |

**总修改量**: 5个文件，约35行新增/修改代码。

---

## 八、注意事项

1. **`rknn_set_core_mask()` 必须在 `rknn_init()` 之后调用**，否则会失败
2. **核心绑定是建议性的**，RKNN驱动在极端情况下可能仍会调度到其他核心
3. **如果线程数超过3**，多余的线程将使用 `RKNN_NPU_CORE_AUTO`（不绑定）
4. **此修改不影响单线程模式**，单线程时只绑定核心0
5. **RK3588的3个NPU核心是对等的**，不存在性能差异，绑定哪个核心到哪个线程无所谓

请在 OrangePi 5 Plus 上重新编译并运行验证。
