// MJPEG HTTP 推流服务器
// 通过 HTTP 协议将视频帧以 MJPEG 格式推送到浏览器
// 使用方法：在浏览器中访问 http://<板子IP>:<端口>/

#ifndef MJPEG_SERVER_H
#define MJPEG_SERVER_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>

class MjpegServer {
public:
    /**
     * @brief 构造函数
     * @param port HTTP 监听端口（默认 8080）
     */
    explicit MjpegServer(int port = 8080);
    ~MjpegServer();

    // 启动 HTTP 服务器（非阻塞，内部创建线程）
    bool Start();

    // 停止 HTTP 服务器
    void Stop();

    // 推送一帧图像（线程安全，由推理线程调用）
    void PushFrame(const cv::Mat& frame);

    // 是否有客户端连接
    bool HasClients() const;

    // 获取当前连接的客户端数
    int GetClientCount() const;

    // 获取服务器端口
    int GetPort() const { return port_; }

private:
    // HTTP 服务器主循环
    void ServerThread();

    // 处理单个客户端连接
    void HandleClient(int client_fd);

    // 发送主页面 HTML
    void SendHomePage(int client_fd);

    // 发送 MJPEG 流
    void SendMjpegStream(int client_fd);

    // 辅助函数：发送 HTTP 响应
    bool SendAll(int fd, const void* buf, size_t len);

    int port_;
    int server_fd_;
    std::atomic<bool> running_;
    std::atomic<int> client_count_;

    // 最新帧缓冲（只保留最新一帧）
    std::mutex frame_mutex_;
    std::vector<uchar> frame_buffer_;

    std::thread server_thread_;
};

#endif // MJPEG_SERVER_H
