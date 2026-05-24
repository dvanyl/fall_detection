# YOLOv8-Pose 人体跌倒检测系统 - 项目规划

## 一、项目概述

基于现有YOLOv8-Pose RK3588部署代码，新增跌倒检测功能模块，实现：
- USB摄像头实时读流
- YOLOv8-Pose人体关键点检测
- 基于关键点的跌倒状态判断
- 跌倒报警（声音/视觉/日志）
- RKNN + RGA硬件加速，目标FPS 30+

## 二、现有代码架构分析

```
┌─────────────────────────────────────────────────────┐
│                    应用层 (main)                      │
│  yolov8_pose_img.cpp / yolov8_pose_video.cpp        │
├─────────────────────────────────────────────────────┤
│                  任务层 (Task)                        │
│  Yolov8Custom: LoadModel → Run → Preprocess          │
│                → Inference → Postprocess              │
│  Yolov8ThreadPool: 多线程并行推理                      │
├─────────────────────────────────────────────────────┤
│                  引擎层 (Engine)                      │
│  NNEngine (抽象) → RKEngine (RKNN实现)                │
├─────────────────────────────────────────────────────┤
│                  处理层 (Process)                     │
│  Preprocess: letterbox + resize (OpenCV/RGA)         │
│  Postprocess: 解码检测框 + 17个COCO关键点              │
├─────────────────────────────────────────────────────┤
│                  数据类型 (Types)                     │
│  Detection: class_id, className, confidence, box     │
│  KeyPoint: x, y, score, id (17个COCO关键点)           │
│  DetectRect: xmin, ymin, xmax, ymax, score, keyPoints│
└─────────────────────────────────────────────────────┘
```

### COCO 17个关键点索引
```
0: 鼻子      1: 左眼     2: 右眼     3: 左耳     4: 右耳
5: 左肩      6: 右肩     7: 左肘     8: 右肘     9: 左腕
10: 右腕     11: 左髋    12: 右髋    13: 左膝    14: 右膝
15: 左踝     16: 右踝
```

## 三、跌倒检测算法设计

### 3.1 核心特征提取

从17个关键点中提取以下特征用于跌倒判断：

| 特征 | 计算方式 | 跌倒时表现 |
|------|----------|------------|
| 躯干角度 | 肩膀中点→髋部中点连线与垂直方向夹角 | >60° |
| 身体宽高比 | bbox宽度/bbox高度 | >1.2 |
| 头部相对位置 | 头部y坐标 vs 脚部y坐标 | 头部接近脚部 |
| 重心高度 | 关键点y坐标均值归一化 | 显著降低 |
| 肩膀倾斜角 | 左右肩连线与水平方向夹角 | >45° |
| 身体展平度 | 关键点y坐标标准差 | 减小（身体水平） |

### 3.2 跌倒判断逻辑（多条件融合）

```
跌倒得分 = w1 * 角度得分 + w2 * 宽高比得分 + w3 * 头脚距离得分 + w4 * 重心得分

if 跌倒得分 > 阈值:
    判定为跌倒状态
```

### 3.3 状态机设计

```
                    检测到人体
                        │
                        ▼
              ┌─────────────────┐
              │    STANDING     │ ← 初始/正常状态
              │    (站立)        │
              └────────┬────────┘
                       │ 跌倒得分持续 > 阈值 (连续N帧)
                       ▼
              ┌─────────────────┐
              │    FALLING      │ ← 跌倒中（过渡态）
              │    (跌倒中)      │
              └────────┬────────┘
                       │ 跌倒得分持续 > 阈值 (连续M帧)
                       ▼
              ┌─────────────────┐
              │    FALLEN       │ ← 已跌倒（报警）
              │    (已跌倒)      │ ← 触发报警
              └────────┬────────┘
                       │ 跌倒得分 < 阈值 (连续K帧)
                       ▼
              ┌─────────────────┐
              │   RECOVERING    │ ← 恢复中
              │   (恢复中)       │
              └────────┬────────┘
                       │ 恢复得分持续 > 阈值
                       ▼
              ┌─────────────────┐
              │    STANDING     │ ← 回到正常
              └─────────────────┘
```

### 3.4 报警机制

1. **视觉报警**：画面中绘制红色警告框 + "FALL DETECTED!" 文字
2. **声音报警**：蜂鸣器/音频输出（可选）
3. **日志报警**：记录跌倒时间、持续时长、关键点数据
4. **GPIO报警**：可扩展接外部蜂鸣器/LED

## 四、新增文件规划

### 4.1 文件结构

```
src/
├── task/
│   ├── yolov8_custom.h/cpp          # 已有，不修改
│   ├── yolov8_thread_pool.h/cpp     # 已有，不修改
│   └── fall_detector.h/cpp          # 【新增】跌倒检测器
├── types/
│   ├── nn_datatype.h                # 已有，不修改
│   └── fall_datatype.h              # 【新增】跌倒检测数据类型
├── draw/
│   ├── cv_draw.h/cpp                # 已有，不修改
│   └── fall_draw.h/cpp              # 【新增】跌倒检测绘制
├── yolov8_fall_detect.cpp           # 【新增】跌倒检测主程序
└── yolov8_fall_detect_usb.cpp       # 【新增】USB摄像头跌倒检测主程序
```

### 4.2 核心类设计

#### FallDetector 类

```cpp
class FallDetector {
public:
    FallDetector();
    ~FallDetector();

    // 设置检测参数
    void SetThreshold(float fall_threshold, int fall_frames, int recover_frames);

    // 核心接口：输入关键点和检测框，输出跌倒状态
    FallState Update(const std::vector<Detection>& objects,
                     const std::vector<std::map<int, KeyPoint>>& keypoints);

    // 获取当前状态
    FallState GetState() const;

    // 获取跌倒得分（用于调试）
    float GetFallScore() const;

private:
    // 特征提取
    float CalcTrunkAngle(const std::map<int, KeyPoint>& kps);
    float CalcBboxRatio(const Detection& det);
    float CalcHeadFootDistance(const std::map<int, KeyPoint>& kps);
    float CalcCenterOfGravity(const std::map<int, KeyPoint>& kps);
    float CalcShoulderTilt(const std::map<int, KeyPoint>& kps);
    float CalcBodyFlatness(const std::map<int, KeyPoint>& kps);

    // 跌倒得分计算
    float CalcFallScore(const Detection& det, const std::map<int, KeyPoint>& kps);

    // 状态机更新
    FallState UpdateStateMachine(float fall_score);

    // 参数
    float fall_threshold_;      // 跌倒得分阈值
    int fall_confirm_frames_;   // 确认跌倒所需连续帧数
    int recover_confirm_frames_;// 确认恢复所需连续帧数

    // 状态机
    FallState current_state_;
    int fall_frame_count_;      // 连续跌倒帧计数
    int recover_frame_count_;   // 连续恢复帧计数
    float last_fall_score_;     // 上一帧跌倒得分
};
```

## 五、实现步骤

### Step 1: 创建数据类型定义
- 文件：`src/types/fall_datatype.h`
- 定义 `FallState` 枚举、`FallResult` 结构体

### Step 2: 实现 FallDetector 核心类
- 文件：`src/task/fall_detector.h` + `src/task/fall_detector.cpp`
- 实现特征提取、得分计算、状态机

### Step 3: 实现跌倒检测绘制
- 文件：`src/draw/fall_draw.h` + `src/draw/fall_draw.cpp`
- 绘制跌倒警告、状态信息、特征数据

### Step 4: 创建跌倒检测主程序
- 文件：`src/yolov8_fall_detect_usb.cpp`
- USB摄像头读取 → YOLOv8-Pose推理 → 跌倒检测 → 报警显示

### Step 5: 更新CMakeLists.txt
- 添加新的库和可执行文件

### Step 6: 测试和调优
- 调整跌倒阈值参数
- 优化FPS性能

## 六、性能优化策略

1. **RGA硬件加速**：预处理使用RGA替代OpenCV
2. **多线程流水线**：读流线程 + 推理线程 + 显示线程分离
3. **跳帧策略**：可选隔帧推理，降低计算量
4. **NPU核心分配**：RK3588有3个NPU核心，可并行使用

## 七、预期效果

- 跌倒检测准确率：90%+（通过多特征融合 + 状态机平滑）
- 实时FPS：30+（640x640输入，RGA加速）
- 误报率：<5%（状态机过滤短暂异常姿态）
- 漏报率：<5%（多特征互补检测）
