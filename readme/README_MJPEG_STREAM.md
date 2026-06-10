# MJPEG 网络推流功能 - 完整技术文档

> 本文档详细记录了从零实现 MJPEG HTTP 推流服务器、集成到 YOLOv8-Pose 跌倒检测系统、以及后续显示优化的完整过程。
> 适合嵌入式 AI 部署方向的学习、面试准备和工程参考。

---

## 一、需求背景

### 1.1 问题描述

RK3588 开发板运行跌倒检测程序时，由于没有外接显示器（Headless 模式），无法实时查看摄像头画面和检测结果。虽然日志中会输出 FPS、跌倒状态等信息，但缺少直观的视觉反馈。

### 1.2 解决方案选择

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **MJPEG HTTP 推流** ✅ | 浏览器直接查看，无需插件，延迟低 | 占用网络带宽 | 嵌入式设备远程监控 |
| RTSP 推流 | 标准协议，VLC 等播放器支持 | 需要额外库（live555），配置复杂 | 专业视频监控 |
| VNC/X11 转发 | 无需改代码 | 需要 X11 支持，带宽消耗大 | 临时调试 |
| 录像+下载 | 最简单 | 不是实时的 | 事后分析 |

**选择 MJPEG 的理由**：
- 实现简单，只需 POSIX socket + OpenCV JPEG 编码
- 浏览器原生支持（`<img src="/stream">`），无需任何插件
- 嵌入式友好，不依赖外部库
- 延迟可控（约 66ms 一帧）

---

## 二、MJPEG 协议原理

### 2.1 什么是 MJPEG

MJPEG（Motion JPEG）是将视频的每一帧独立编码为 JPEG 图片，然后通过 HTTP 协议连续发送的技术。

```
HTTP 响应:
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg
Content-Length: 12345

[JPEG 数据1]

--frame
Content-Type: image/jpeg
Content-Length: 12346

[JPEG 数据2]

--frame
...
```

### 2.2 与传统视频流的对比

| 特性 | MJPEG | H.264/H.265 |
|------|-------|-------------|
| 压缩方式 | 帧内压缩（每帧独立） | 帧间压缩（利用前后帧相似性） |
| 带宽占用 | 较高（约 200-500 KB/帧） | 较低（约 20-50 KB/帧） |
| 编码复杂度 | 低（JPEG 编码） | 高（需要硬件编码器） |
| 延迟 | 低（无 GOP 延迟） | 较高（需要 I 帧参考） |
| 浏览器支持 | 原生支持 | 需要 MSE/WASM |
| 实现难度 | 简单 | 复杂 |

### 2.3 HTTP 服务器架构

```
┌─────────────────────────────────────────────────┐
│                 MJPEG Server                      │
│                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │ HTTP     │    │ 帧缓冲   │    │ JPEG     │    │
│  │ 监听器   │    │ (最新帧) │    │ 编码器   │    │
│  │ :8080    │    │          │    │ OpenCV   │    │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    │
│       │               │               │          │
│  ┌────▼─────┐    ┌────▼─────┐    ┌────▼─────┐    │
│  │ 路由分发 │    │ 线程安全 │    │ 帧推送   │    │
│  │ / → HTML │    │ mutex    │    │ PushFrame│    │
│  │ /stream  │    │ 保护     │    │ ()       │    │
│  └──────────┘    └──────────┘    └──────────┘    │
└─────────────────────────────────────────────────┘
         │                              │
    浏览器请求                     推理线程推送帧
```

---

## 三、实现详解

### 3.1 文件结构

```
src/stream/
├── mjpeg_server.h      # 头文件：类声明、接口定义
└── mjpeg_server.cpp    # 实现：HTTP 服务器、MJPEG 流、HTML 页面
```

### 3.2 核心类设计

```cpp
class MjpegServer {
public:
    explicit MjpegServer(int port = 8080);  // 构造，指定端口
    ~MjpegServer();

    bool Start();                           // 启动服务器（非阻塞）
    void Stop();                            // 停止服务器
    void PushFrame(const cv::Mat& frame);   // 推送帧（线程安全）
    bool HasClients() const;                // 是否有客户端连接
    int GetClientCount() const;             // 获取客户端数

private:
    void ServerThread();                    // 服务器主循环
    void HandleClient(int client_fd);       // 处理单个客户端
    void SendHomePage(int client_fd);       // 发送 HTML 页面
    void SendMjpegStream(int client_fd);    // 发送 MJPEG 流
    bool SendAll(int fd, const void* buf, size_t len);  // 可靠发送

    int port_;
    int server_fd_;
    std::atomic<bool> running_;
    std::atomic<int> client_count_;
    std::mutex frame_mutex_;
    std::vector<uchar> frame_buffer_;       // JPEG 编码后的帧数据
    std::thread server_thread_;
};
```

**设计要点**：

1. **非阻塞启动**：`Start()` 内部创建线程，立即返回，不阻塞主线程
2. **线程安全帧缓冲**：`PushFrame()` 使用 mutex 保护，推理线程和 HTTP 线程安全共享
3. **只保留最新帧**：帧缓冲区只有一个槽位，新帧覆盖旧帧，避免内存堆积
4. **智能编码**：只在有客户端连接时才进行 JPEG 编码，无客户端时零开销

### 3.3 HTTP 服务器实现

#### 3.3.1 Socket 创建和监听

```cpp
bool MjpegServer::Start()
{
    // 1. 创建 TCP socket
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);

    // 2. 设置 SO_REUSEADDR（允许快速重启）
    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 3. 绑定到所有网卡的指定端口
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);
    bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr));

    // 4. 开始监听（ backlog=5 ）
    listen(server_fd_, 5);

    // 5. 启动服务器线程
    running_ = true;
    server_thread_ = std::thread(&MjpegServer::ServerThread, this);
}
```

#### 3.3.2 accept 超时机制

```cpp
void MjpegServer::ServerThread()
{
    while (running_.load())
    {
        // 使用 select() 实现 accept 超时
        // 这样 running_ 变为 false 时，线程能在 1 秒内退出
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_fd_, &read_fds);
        struct timeval tv = {1, 0};  // 1 秒超时

        int ret = select(server_fd_ + 1, &read_fds, nullptr, nullptr, &tv);
        if (ret <= 0) continue;  // 超时或错误，重试

        int client_fd = accept(server_fd_, ...);
        // 每个客户端一个独立线程处理
        std::thread([this, client_fd]() {
            HandleClient(client_fd);
            close(client_fd);
        }).detach();
    }
}
```

**为什么用 `select()` 而不是直接 `accept()`？**

直接 `accept()` 会阻塞，当调用 `Stop()` 时，服务器线程卡在 `accept()` 上无法退出。使用 `select()` 设置超时后，每秒检查一次 `running_` 标志，实现优雅退出。

#### 3.3.3 HTTP 路由

```cpp
void MjpegServer::HandleClient(int client_fd)
{
    // 读取 HTTP 请求
    char buf[4096];
    int n = recv(client_fd, buf, sizeof(buf) - 1, 0);

    // 解析请求路径
    std::string path = parse_path(buf);

    // 路由
    if (path == "/" || path == "/index.html")
        SendHomePage(client_fd);        // 返回 HTML 页面
    else if (path == "/stream")
        SendMjpegStream(client_fd);     // 返回 MJPEG 流
    else
        send_404(client_fd);            // 404
}
```

#### 3.3.4 MJPEG 流发送

```cpp
void MjpegServer::SendMjpegStream(int client_fd)
{
    client_count_++;  // 原子计数

    // 1. 发送 HTTP 头（声明是 MJPEG 流）
    std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache\r\n"
        "\r\n";
    SendAll(client_fd, header.c_str(), header.size());

    // 2. 循环发送每一帧
    while (running_.load())
    {
        // 从帧缓冲区取出最新帧
        std::vector<uchar> jpeg_data;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (!frame_buffer_.empty())
                jpeg_data = frame_buffer_;
        }

        if (!jpeg_data.empty())
        {
            // 发送帧头
            std::string hdr = "--frame\r\n"
                              "Content-Type: image/jpeg\r\n"
                              "Content-Length: " + std::to_string(jpeg_data.size()) + "\r\n"
                              "\r\n";
            SendAll(client_fd, hdr.c_str(), hdr.size());

            // 发送 JPEG 数据
            SendAll(client_fd, jpeg_data.data(), jpeg_data.size());

            // 帧间分隔
            SendAll(client_fd, "\r\n", 2);
        }

        // 控制帧率：约 15 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(66));
    }

    client_count_--;
}
```

### 3.4 帧推送接口

```cpp
void MjpegServer::PushFrame(const cv::Mat& frame)
{
    // 1. 无客户端时跳过编码（节省 CPU）
    if (client_count_.load() <= 0) return;

    // 2. JPEG 编码
    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 75};
    cv::imencode(".jpg", frame, buf, params);

    // 3. 更新帧缓冲（线程安全）
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        frame_buffer_ = std::move(buf);
    }
}
```

**关键设计**：
- `cv::imencode` 是 OpenCV 提供的 JPEG 编码函数，直接在内存中完成
- 质量参数 75 是画质和带宽的平衡点（0-100，越高质量越好但越大）
- `std::move(buf)` 避免拷贝，直接转移所有权

### 3.5 主程序集成

在 [`yolov8_fall_detect_usb.cpp`](src/yolov8_fall_detect_usb.cpp) 中的集成方式：

```cpp
// 1. 创建 MJPEG 服务器
MjpegServer* mjpeg_server = new MjpegServer(8080);
mjpeg_server->Start();

// 2. 在显示线程中推送帧
void displayThread(..., MjpegServer* mjpeg_server)
{
    while (!g_stop)
    {
        ResultData result = ...;  // 从队列取出推理结果

        // 绘制检测框、关键点、跌倒状态等
        DrawDetections(result.frame, result.objects);
        DrawCocoKps(result.frame, result.keypoints);
        DrawFallResult(result.frame, result.fall_result);

        // 推送到 MJPEG 服务器（绘制后的帧）
        if (mjpeg_server)
            mjpeg_server->PushFrame(result.frame);
    }
}

// 3. 程序退出时清理
mjpeg_server->Stop();
delete mjpeg_server;
```

### 3.6 CMakeLists.txt 集成

```cmake
# 新增 MJPEG 推流库
add_library(mjpeg_stream_lib SHARED src/stream/mjpeg_server.cpp)
target_link_libraries(mjpeg_stream_lib
    ${OpenCV_LIBS}
    pthread
)

# 链接到主程序
target_link_libraries(yolov8_fall_detect_usb
    ...
    mjpeg_stream_lib
    pthread
)
```

---

## 四、显示优化详解

### 4.1 问题诊断

初始版本存在以下显示问题：

| 问题 | 根因 | 影响 |
|------|------|------|
| 浏览器中视频窗口很小 | HTML 用 `max-width:100%`，640×480 在高清屏幕上确实很小 | 只能看到一小块画面 |
| 内容被遮挡 | 调试面板 280×208 像素，占帧面积 43% | 人体躯干被遮挡 |
| 只能看到脸 | 调试面板从左上角开始，覆盖了躯干区域 | 检测结果不可见 |
| 关键点太大 | 圆半径=5，在 640×480 上过大 | 视觉混乱 |
| 骨架线太粗 | 线宽=2，多人时混乱 | 难以区分不同人 |

### 4.2 HTML 页面优化

**优化前**：
```css
img { max-width: 100%; height: auto; }
/* 640×480 的图片在 1920×1080 屏幕上只占 1/3 宽度 */
```

**优化后**：
```css
.stream-container {
    width: 100%;           /* 容器占满屏幕宽度 */
    max-width: 1280px;     /* 最大不超过 1280px */
}
img {
    width: 100%;           /* 图片强制填满容器 */
    height: auto;          /* 高度等比缩放 */
}
```

**核心思想**：
- `max-width: 100%` 是"不超过容器宽度"，如果图片本身只有 640px，就不会放大
- `width: 100%` 是"强制占满容器宽度"，图片会被拉伸到容器大小
- 对于 MJPEG 视频流，`width: 100%` 更合适，因为视频内容会随摄像头分辨率变化

### 4.3 叠加层布局优化

**优化前的布局**（640×480 帧）：
```
┌──────────────────────────────┐
│ FPS: 17.5                    │  ← 左上角
│ ┌────────────────────┐       │
│ │=== Fall Debug ===   │       │  ← 280×208 的巨大面板
│ │Trunk Angle:   12.34│       │     覆盖了 43% 的画面
│ │BBox Ratio:     0.65│       │
│ │Head-Foot Dist: 0.82│       │
│ │Center Gravity: 0.45│       │
│ │Shoulder Tilt:   5.2│       │
│ │Body Flatness:  0.18│       │
│ │Fall Score:     0.40│       │
│ │State: STANDING ... │       │
│ └────────────────────┘       │
│              State: STANDING │  ← 右上角
│              Fall Score: 0.40│
│                              │
│ Latency: 274.7ms             │  ← 左下角
└──────────────────────────────┘
```

**优化后的布局**：
```
┌──────────────────────────────┐
│ FPS:17.5        State:STANDING Score:0.40│  ← 顶部一行
│                              │
│        ┌─────┐               │
│        │person│               │  ← 人体区域完全可见
│        │  ●───●──●           │
│        │  │   │  │           │  ← 小关键点(半径3)
│        │  ●───●  │           │     细骨架(线宽1)
│        │  │     │            │     按部位着色
│        │  ●     ●            │
│        └─────┘               │
│ ┌──────────────────┐         │
│ │Fall Debug         │         │  ← 左下角小面板
│ │Angle:  12.34      │         │     200×130 紧凑
│ │Ratio:  0.65       │         │     不遮挡人体
│ │...                │         │
│ └──────────────────┘         │
└──────────────────────────────┘
```

### 4.4 关键点和骨架优化

**关键点着色方案**（按 COCO 17 关键点分组）：

```cpp
// 头部（红色）：鼻子、左眼、右眼、左耳、右耳
// 上半身（蓝色）：左右肩、左右肘、左右腕
// 下半身（绿色）：左右髋、左右膝、左右踝

static const cv::Scalar kp_colors[17] = {
    cv::Scalar(0, 0, 255),     // 0: 鼻子 - 红
    cv::Scalar(0, 0, 255),     // 1: 左眼
    // ...
    cv::Scalar(255, 128, 0),   // 5: 左肩 - 蓝
    // ...
    cv::Scalar(0, 255, 0),     // 11: 左髋 - 绿
    // ...
};
```

**骨架着色方案**（按身体部位）：
```cpp
// 腿部骨架：绿色
// 躯干/手臂骨架：蓝色
// 头部骨架：红色
```

**优化参数对比**：

| 参数 | 优化前 | 优化后 | 效果 |
|------|--------|--------|------|
| 关键点半径 | 5px | 3px | 不遮挡人体细节 |
| 关键点颜色 | 统一绿色 | 按部位分色 | 快速识别身体部位 |
| 骨架线宽 | 2px | 1px | 更精细，不遮挡 |
| 骨架颜色 | 统一黄色 | 按部位分色 | 视觉层次清晰 |
| 标签背景 | 无 | 半透明黑色 | 文字更清晰 |
| FPS 字号 | 0.7 | 0.45 | 更紧凑 |
| 调试面板位置 | 左上角 | 左下角 | 不遮挡人体 |

### 4.5 JPEG 编码质量调优

| 质量值 | 文件大小 | 画质 | 适用场景 |
|--------|----------|------|----------|
| 50 | ~50KB/帧 | 较差 | 低带宽环境 |
| 70 | ~80KB/帧 | 良好 | 平衡选择 |
| **75** | ~100KB/帧 | **优秀** | **推荐值** |
| 90 | ~200KB/帧 | 极好 | 高带宽环境 |
| 100 | ~400KB/帧 | 无损 | 调试用 |

---

## 五、数据流架构

### 5.1 完整数据流

```
USB 摄像头
    │
    ▼
┌──────────┐     SafeQueue      ┌──────────┐     SafeQueue      ┌──────────┐
│ 读流线程  │ ──────────────→  │ 推理线程  │ ──────────────→  │ 显示线程  │
│          │    FrameData       │ YOLOv8   │    ResultData      │          │
│ cv::Video│                    │ +RGA     │                    │ Draw*    │
│ Capture  │                    │ +FallDet │                    │ +MJPEG   │
└──────────┘                    └──────────┘                    └────┬─────┘
                                                                     │
                                                          PushFrame()│
                                                                     ▼
                                                              ┌──────────┐
                                                              │ MJPEG    │
                                                              │ Server   │
                                                              │ :8080    │
                                                              └────┬─────┘
                                                                   │
                                                        HTTP MJPEG │
                                                                   ▼
                                                              ┌──────────┐
                                                              │ 浏览器   │
                                                              │          │
                                                              └──────────┘
```

### 5.2 线程模型

```
线程1: 读流线程 (readThread)
  └── cv::VideoCapture::read() → SafeQueue<FrameData>

线程2: 推理线程 (inferenceThread)
  └── SafeQueue<FrameData>::pop()
      → Yolov8Custom::Run() (预处理 + NPU推理 + 后处理)
      → FallDetector::Update() (跌倒检测)
      → SafeQueue<ResultData>::push()

线程3: 显示线程 (displayThread)
  └── SafeQueue<ResultData>::pop()
      → DrawDetections() + DrawCocoKps() + DrawFallResult()
      → MjpegServer::PushFrame()  ← 新增
      → cv::VideoWriter::write() (录像)
      → cv::imshow() (本地显示) 或 日志输出 (headless)

线程4: HTTP 服务器线程 (MjpegServer::ServerThread)
  └── accept() → 每个客户端一个线程
      → HandleClient() → SendMjpegStream()

线程5+: HTTP 客户端线程 (每个浏览器连接一个)
  └── 循环读取 frame_buffer_ → 发送 JPEG 数据
```

---

## 六、关键知识点（面试/工作参考）

### 6.1 网络编程

| 知识点 | 在本项目中的应用 |
|--------|------------------|
| TCP Socket 编程 | `socket()` → `bind()` → `listen()` → `accept()` → `recv()`/`send()` |
| HTTP 协议 | 请求解析、响应构造、Content-Type 头 |
| 多线程服务器 | 每个客户端一个线程（thread-per-connection 模型） |
| select() 超时 | 实现非阻塞 accept，支持优雅退出 |
| TCP_NODELAY | 禁用 Nagle 算法，减少小包延迟 |
| MSG_NOSIGNAL | 防止对已关闭的 socket 发送导致 SIGPIPE 信号 |

### 6.2 多线程编程

| 知识点 | 在本项目中的应用 |
|--------|------------------|
| std::mutex | 保护帧缓冲区的读写 |
| std::atomic | 无锁的 running_ 标志和 client_count_ 计数 |
| std::condition_variable | SafeQueue 中的等待/通知机制 |
| std::thread | 服务器线程和客户端线程 |
| thread::detach() | 客户端线程独立运行，不阻塞主线程 |
| 生产者-消费者模式 | 推理线程生产，显示线程消费 |

### 6.3 图像处理

| 知识点 | 在本项目中的应用 |
|--------|------------------|
| JPEG 编码 | `cv::imencode(".jpg", frame, buf, params)` |
| 图像叠加 | `cv::addWeighted()` 实现半透明背景 |
| 绘制基元 | `cv::rectangle()`, `cv::circle()`, `cv::line()`, `cv::putText()` |
| 字体缩放 | `cv::FONT_HERSHEY_SIMPLEX` + font_scale 参数 |
| ROI 操作 | `cv::Mat roi = img(rect)` 局部区域操作 |

### 6.4 嵌入式系统设计

| 知识点 | 在本项目中的应用 |
|--------|------------------|
| Headless 模式 | 无显示器时的运行策略 |
| 资源受限优化 | 无客户端时跳过 JPEG 编码 |
| 帧率控制 | sleep_for 控制推流帧率 |
| 信号处理 | SIGINT/SIGTERM 优雅退出 |
| 内存管理 | 只保留最新帧，避免内存堆积 |

---

## 七、扩展学习方向

### 7.1 性能优化

1. **零拷贝优化**：使用共享内存替代 mutex + 拷贝
2. **JPEG 硬件编码**：RK3588 支持硬件 JPEG 编码，比 OpenCV 软编码快 5-10 倍
3. **自适应质量**：根据网络带宽动态调整 JPEG 质量
4. **帧率自适应**：根据客户端数量调整推流帧率

### 7.2 功能扩展

1. **RTSP 推流**：使用 GStreamer 或 live555 实现标准 RTSP 协议
2. **WebSocket 推流**：更低延迟，支持双向通信
3. **多路流**：同时推原始画面和检测结果画面
4. **录制回放**：在服务器端缓存最近 N 分钟的视频

### 7.3 架构改进

1. **事件驱动模型**：用 epoll 替代 thread-per-connection，支持更多并发
2. **异步 I/O**：使用 io_uring（Linux 5.1+）提升 I/O 性能
3. **微服务化**：将推流服务独立为单独进程，通过 IPC 通信

---

## 八、编译和运行

### 8.1 编译

```bash
cd ~/course_projs_dyl/fall_detection_deploy/build
cmake ..
make -j$(nproc)
```

### 8.2 运行

```bash
# 基本运行（MJPEG 推流默认 8080 端口）
./yolov8_fall_detect_usb ../weights/yolov8-pose-int.rknn 0 640 480 0 1

# 指定推流端口
./yolov8_fall_detect_usb ../weights/yolov8-pose-int.rknn 0 640 480 0 1 9090

# 禁用推流
./yolov8_fall_detect_usb ../weights/yolov8-pose-int.rknn 0 640 480 0 1 0
```

### 8.3 浏览器查看

1. 确保电脑和 RK3588 板子在同一网络
2. 查看板子 IP：`hostname -I` 或 `ip addr`
3. 浏览器打开：`http://<板子IP>:8080/`
4. 也可以用 VLC 打开网络流：`http://<板子IP>:8080/stream`

---

## 九、修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| [`src/stream/mjpeg_server.h`](src/stream/mjpeg_server.h) | 新增 | MJPEG 服务器头文件 |
| [`src/stream/mjpeg_server.cpp`](src/stream/mjpeg_server.cpp) | 新增 | MJPEG 服务器实现 + HTML 页面 |
| [`src/yolov8_fall_detect_usb.cpp`](src/yolov8_fall_detect_usb.cpp) | 修改 | 集成 MJPEG 推流，新增 stream_port 参数 |
| [`src/draw/fall_draw.cpp`](src/draw/fall_draw.cpp) | 修改 | 优化叠加层布局（面板缩小、移至左下角） |
| [`src/draw/cv_draw.cpp`](src/draw/cv_draw.cpp) | 修改 | 关键点缩小、骨架着色、标签背景 |
| [`CMakeLists.txt`](CMakeLists.txt) | 修改 | 新增 mjpeg_stream_lib 构建目标 |
