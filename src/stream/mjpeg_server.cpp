// MJPEG HTTP 推流服务器实现
// 使用 POSIX socket 实现轻量级 HTTP 服务器，无需外部依赖

#include "mjpeg_server.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <sstream>
#include <iostream>

#include "utils/logging.h"

// ==================== HTML 页面模板 ====================
// 全屏响应式布局，视频自动填满浏览器窗口宽度
static const char* HTML_PAGE = R"HTML(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8-Pose Fall Detection - RK3588</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a14;
            color: #eee;
            font-family: 'Segoe UI', Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 8px;
        }
        .header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 6px;
            width: 100%;
            max-width: 1280px;
            justify-content: center;
            flex-wrap: wrap;
        }
        h1 {
            color: #e94560;
            font-size: 18px;
        }
        .info {
            color: #888;
            font-size: 12px;
        }
        .stream-container {
            width: 100%;
            max-width: 1280px;
            border: 2px solid #e94560;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 0 15px rgba(233, 69, 96, 0.2);
            background: #000;
        }
        img {
            display: block;
            width: 100%;
            height: auto;
        }
        .controls {
            margin-top: 6px;
            display: flex;
            gap: 12px;
            align-items: center;
            width: 100%;
            max-width: 1280px;
            justify-content: center;
        }
        .status {
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 11px;
        }
        .status.live {
            background: #27ae60;
            color: white;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .stat {
            color: #888;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>YOLOv8-Pose Fall Detection</h1>
        <span class="info">RK3588 NPU | INT8 | MJPEG Stream</span>
    </div>
    <div class="stream-container">
        <img src="/stream" alt="Live Stream" />
    </div>
    <div class="controls">
        <span class="status live">LIVE</span>
        <span class="stat" id="fps"></span>
    </div>
    <script>
        var fc=0,lt=Date.now();
        document.querySelector('img').onload=function(){
            fc++;
            var n=Date.now();
            if(n-lt>=1000){
                document.getElementById('fps').textContent='Display: '+fc+' FPS';
                fc=0;lt=n;
            }
        };
    </script>
</body>
</html>
)HTML";

// ==================== 构造/析构 ====================

MjpegServer::MjpegServer(int port)
    : port_(port)
    , server_fd_(-1)
    , running_(false)
    , client_count_(0)
{
}

MjpegServer::~MjpegServer()
{
    Stop();
}

// ==================== 公共接口 ====================

bool MjpegServer::Start()
{
    if (running_.load())
    {
        NN_LOG_WARNING("MjpegServer already running");
        return true;
    }

    // 创建 socket
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0)
    {
        NN_LOG_ERROR("MjpegServer: socket() failed: %s", strerror(errno));
        return false;
    }

    // 设置 SO_REUSEADDR
    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 绑定地址
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0)
    {
        NN_LOG_ERROR("MjpegServer: bind() failed on port %d: %s", port_, strerror(errno));
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    // 开始监听
    if (listen(server_fd_, 5) < 0)
    {
        NN_LOG_ERROR("MjpegServer: listen() failed: %s", strerror(errno));
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_ = true;
    server_thread_ = std::thread(&MjpegServer::ServerThread, this);

    NN_LOG_INFO("========================================");
    NN_LOG_INFO("  MJPEG Stream Server Started");
    NN_LOG_INFO("  URL: http://0.0.0.0:%d/", port_);
    NN_LOG_INFO("  Stream: http://0.0.0.0:%d/stream", port_);
    NN_LOG_INFO("========================================");
    NN_LOG_INFO("Open browser and navigate to: http://<board_ip>:%d/", port_);

    return true;
}

void MjpegServer::Stop()
{
    if (!running_.load())
        return;

    running_ = false;

    // 关闭 server socket，使 accept() 返回
    if (server_fd_ >= 0)
    {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }

    if (server_thread_.joinable())
    {
        server_thread_.join();
    }

    NN_LOG_INFO("MjpegServer stopped");
}

void MjpegServer::PushFrame(const cv::Mat& frame)
{
    if (client_count_.load() <= 0)
        return;  // 没有客户端连接，跳过编码

    if (frame.empty())
        return;

    // 将帧编码为 JPEG
    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 75};  // 质量75，平衡带宽和画质
    cv::imencode(".jpg", frame, buf, params);

    // 更新帧缓冲（只保留最新帧）
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        frame_buffer_ = std::move(buf);
    }
}

bool MjpegServer::HasClients() const
{
    return client_count_.load() > 0;
}

int MjpegServer::GetClientCount() const
{
    return client_count_.load();
}

// ==================== 内部实现 ====================

void MjpegServer::ServerThread()
{
    NN_LOG_INFO("MjpegServer: server thread started (port %d)", port_);

    while (running_.load())
    {
        // 设置 accept 超时，避免阻塞在 accept 上无法退出
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_fd_, &read_fds);
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int ret = select(server_fd_ + 1, &read_fds, nullptr, nullptr, &tv);
        if (ret <= 0)
            continue;

        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0)
        {
            if (running_.load())
                NN_LOG_WARNING("MjpegServer: accept() failed: %s", strerror(errno));
            continue;
        }

        NN_LOG_INFO("MjpegServer: client connected from %s:%d",
                     inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

        // 在新线程中处理客户端（每个客户端一个线程）
        std::thread client_thread([this, client_fd, client_addr]() {
            HandleClient(client_fd);
            close(client_fd);
            NN_LOG_INFO("MjpegServer: client disconnected from %s:%d",
                         inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
        });
        client_thread.detach();
    }

    NN_LOG_INFO("MjpegServer: server thread stopped");
}

void MjpegServer::HandleClient(int client_fd)
{
    // 读取 HTTP 请求
    char buf[4096];
    memset(buf, 0, sizeof(buf));
    int n = recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0)
        return;

    std::string request(buf, n);

    // 解析请求行
    std::string path = "/";
    {
        size_t method_end = request.find(' ');
        if (method_end != std::string::npos)
        {
            size_t path_start = method_end + 1;
            size_t path_end = request.find(' ', path_start);
            if (path_end != std::string::npos)
            {
                path = request.substr(path_start, path_end - path_start);
            }
        }
    }

    // 路由
    if (path == "/" || path == "/index.html")
    {
        SendHomePage(client_fd);
    }
    else if (path == "/stream")
    {
        SendMjpegStream(client_fd);
    }
    else
    {
        // 404
        std::string resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        send(client_fd, resp.c_str(), resp.size(), 0);
    }
}

void MjpegServer::SendHomePage(int client_fd)
{
    std::string body(HTML_PAGE);
    std::ostringstream resp;
    resp << "HTTP/1.1 200 OK\r\n"
         << "Content-Type: text/html; charset=UTF-8\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Connection: close\r\n"
         << "\r\n"
         << body;

    std::string data = resp.str();
    SendAll(client_fd, data.c_str(), data.size());
}

void MjpegServer::SendMjpegStream(int client_fd)
{
    client_count_++;

    // 设置 TCP_NODELAY 减少延迟
    int flag = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // 发送 MJPEG 流头
    std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n";

    if (!SendAll(client_fd, header.c_str(), header.size()))
    {
        client_count_--;
        return;
    }

    // 持续发送帧
    const std::string boundary = "--frame\r\n";
    const std::string content_type = "Content-Type: image/jpeg\r\n";

    while (running_.load())
    {
        std::vector<uchar> jpeg_data;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (!frame_buffer_.empty())
            {
                jpeg_data = frame_buffer_;
            }
        }

        if (!jpeg_data.empty())
        {
            // 构造 MJPEG 帧
            std::ostringstream part_header;
            part_header << boundary
                        << content_type
                        << "Content-Length: " << jpeg_data.size() << "\r\n"
                        << "\r\n";

            std::string hdr = part_header.str();
            if (!SendAll(client_fd, hdr.c_str(), hdr.size()))
                break;
            if (!SendAll(client_fd, jpeg_data.data(), jpeg_data.size()))
                break;
            // 帧间分隔
            const char* end = "\r\n";
            if (!SendAll(client_fd, end, 2))
                break;
        }

        // 控制帧率：约 15 FPS（66ms 间隔）
        std::this_thread::sleep_for(std::chrono::milliseconds(66));
    }

    client_count_--;
}

bool MjpegServer::SendAll(int fd, const void* buf, size_t len)
{
    const char* ptr = (const char*)buf;
    size_t remaining = len;
    while (remaining > 0)
    {
        ssize_t sent = send(fd, ptr, remaining, MSG_NOSIGNAL);
        if (sent <= 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        ptr += sent;
        remaining -= sent;
    }
    return true;
}
