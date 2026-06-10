// YOLOv8-Pose 人体跌倒检测系统 - 线程池加速版
// 利用RK3588三核NPU并行推理，目标FPS 30+
//
// 架构：
//   线程1: 读流线程 - 从USB摄像头读取帧
//   线程池(3): 推理线程 - 每个线程独立YOLOv8实例，对应一个NPU核心
//   线程4: 显示线程 - 跌倒检测 + 绘制 + MJPEG推流 + 录像
//
// 用法: ./yolov8_fall_detect_tp <model_path> [camera_id] [width] [height] [num_threads] [stream_port]
//   model_path:   RKNN模型文件路径
//   camera_id:    USB摄像头设备号 (默认0)
//   width:        摄像头分辨率宽度 (默认640)
//   height:       摄像头分辨率高度 (默认480)
//   num_threads:  推理线程数 (默认3，对应3个NPU核心)
//   stream_port:  MJPEG推流端口 (0=禁用, 默认8080)

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <signal.h>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "task/yolov8_thread_pool.h"
#include "task/fall_detector.h"
#include "draw/cv_draw.h"
#include "draw/fall_draw.h"
#include "types/fall_datatype.h"
#include "stream/mjpeg_server.h"
#include "utils/logging.h"

// ==================== 全局变量 ====================
static std::atomic<bool> g_stop(false);
static std::atomic<int> g_frame_start_id(0);
static std::atomic<bool> g_read_finished(false);

void signalHandler(int signum)
{
    NN_LOG_INFO("Received signal %d, stopping...", signum);
    g_stop = true;
}

// ==================== 读流线程 ====================
void readThread(cv::VideoCapture& cap, Yolov8ThreadPool& pool)
{
    NN_LOG_INFO("Read thread started");

    while (!g_stop)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            NN_LOG_INFO("Video end or camera disconnected");
            g_read_finished = true;
            break;
        }

        int frame_id = g_frame_start_id++;
        pool.submitTask(frame, frame_id);
    }

    NN_LOG_INFO("Read thread stopped, total frames submitted: %d", g_frame_start_id.load());
}

// ==================== 结果处理线程 ====================
void resultThread(Yolov8ThreadPool& pool, FallDetector& fall_detector,
                  bool record, MjpegServer* mjpeg_server)
{
    NN_LOG_INFO("Result thread started");

    cv::VideoWriter writer;
    int next_frame_id = 0;

    // FPS 统计
    auto fps_start = std::chrono::high_resolution_clock::now();
    int fps_frame_count = 0;
    float smooth_fps = 0.0f;
    const float alpha = 0.3f;

    while (!g_stop)
    {
        // 帧跳过优化：如果结果线程落后，跳到最新的已完成帧
        int latest_id = pool.getLatestResultId();
        if (latest_id > next_frame_id + 1)
        {
            // 跳过中间帧，直接到最新帧，清理被跳过帧的结果防止内存泄漏
            pool.cleanResultsUpTo(latest_id);
            next_frame_id = latest_id;
        }

        // 获取推理结果
        std::vector<Detection> objects;
        std::vector<std::map<int, KeyPoint>> keypoints;
        auto ret = pool.getTargetResultFull(objects, keypoints, next_frame_id);

        if (ret == NN_TIMEOUT)
        {
            // 超时可能是视频结束
            if (g_read_finished.load())
            {
                NN_LOG_INFO("All frames processed, exiting result thread");
                break;
            }
            continue;
        }
        if (ret != NN_SUCCESS)
        {
            next_frame_id++;
            continue;
        }

        // 获取原始帧用于绘制
        cv::Mat frame;
        auto img_ret = pool.getTargetImgResult(frame, next_frame_id);
        if (img_ret != NN_SUCCESS)
        {
            NN_LOG_WARNING("Failed to get image for frame %d, skipping draw", next_frame_id);
            next_frame_id++;
            continue;
        }

        auto infer_time = std::chrono::high_resolution_clock::now();
        next_frame_id++;

        // 跌倒检测
        FallResult fall_result = fall_detector.Update(objects, keypoints);

        // 绘制检测框和关键点
        DrawDetections(frame, objects);
        DrawCocoKps(frame, keypoints);

        // 绘制跌倒检测结果
        DrawFallResult(frame, fall_result);
        DrawFallDebugInfo(frame, fall_result);

        // 绘制FPS
        fps_frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - fps_start).count() / 1000.0f;
        if (elapsed > 1000.0f)
        {
            float instant_fps = fps_frame_count / (elapsed / 1000.0f);
            smooth_fps = alpha * instant_fps + (1.0f - alpha) * smooth_fps;
            fps_frame_count = 0;
            fps_start = now;
        }
        DrawFPS(frame, smooth_fps);

        // MJPEG 推流
        if (mjpeg_server != nullptr)
        {
            mjpeg_server->PushFrame(frame);
        }

        // 录像
        if (record)
        {
            if (!writer.isOpened())
            {
                int w = frame.cols;
                int h = frame.rows;
                writer = cv::VideoWriter("fall_detect_result.mp4",
                                         cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                         25, cv::Size(w, h));
            }
            if (writer.isOpened())
            {
                writer << frame;
            }
        }

        // 日志输出（每30帧一次）
        if (next_frame_id % 30 == 0)
        {
            NN_LOG_INFO("[Frame %d] FPS:%.1f, State:%s, Score:%.2f, Queue:%d, Clients:%d",
                        next_frame_id, smooth_fps,
                        fall_result.state_str.c_str(),
                        fall_result.fall_score,
                        pool.getTaskQueueSize(),
                        mjpeg_server ? mjpeg_server->GetClientCount() : 0);
        }

        // 跌倒报警
        if (fall_result.is_alarm)
        {
            NN_LOG_WARNING("!!! FALL DETECTED !!! Frame:%d, Score:%.2f, State:%s",
                           next_frame_id, fall_result.fall_score,
                           fall_result.state_str.c_str());
        }
    }

    if (writer.isOpened())
    {
        writer.release();
    }

    NN_LOG_INFO("Result thread stopped");
}

// ==================== 主函数 ====================
int main(int argc, char** argv)
{
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    if (argc < 2)
    {
        std::cout << "========================================" << std::endl;
        std::cout << "  YOLOv8-Pose Fall Detection System" << std::endl;
        std::cout << "  Thread Pool Mode (Multi-NPU)" << std::endl;
        std::cout << "  Platform: RK3588 + RKNN + RGA" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Usage: " << argv[0]
                  << " <model_path> [camera_id] [width] [height] [num_threads] [stream_port]" << std::endl;
        std::cout << "  model_path:   RKNN model file path (.rknn)" << std::endl;
        std::cout << "  camera_id:    USB camera device id (default: 0)" << std::endl;
        std::cout << "  width:        Camera resolution width (default: 640)" << std::endl;
        std::cout << "  height:       Camera resolution height (default: 480)" << std::endl;
        std::cout << "  num_threads:  Inference threads (default: 3, for 3 NPU cores)" << std::endl;
        std::cout << "  stream_port:  MJPEG stream port (0=disabled, default: 8080)" << std::endl;
        std::cout << std::endl;
        std::cout << "Example:" << std::endl;
        std::cout << "  " << argv[0] << " ../weights/yolov8-pose-int.rknn 0 640 480 3 8080" << std::endl;
        return -1;
    }

    const char* model_path = argv[1];
    int camera_id = argc > 2 ? atoi(argv[2]) : 0;
    int cam_width = argc > 3 ? atoi(argv[3]) : 640;
    int cam_height = argc > 4 ? atoi(argv[4]) : 480;
    int num_threads = argc > 5 ? atoi(argv[5]) : 3;
    int stream_port = argc > 6 ? atoi(argv[6]) : 8080;
    bool record = false;  // 可通过参数扩展

    NN_LOG_INFO("========================================");
    NN_LOG_INFO("  YOLOv8-Pose Fall Detection System");
    NN_LOG_INFO("  Thread Pool Mode (Multi-NPU)");
    NN_LOG_INFO("  Platform: RK3588 + RKNN + RGA");
    NN_LOG_INFO("========================================");
    NN_LOG_INFO("Model: %s", model_path);
    NN_LOG_INFO("Camera: %d (%dx%d)", camera_id, cam_width, cam_height);
    NN_LOG_INFO("Threads: %d (NPU cores)", num_threads);
    NN_LOG_INFO("Stream: %s", stream_port > 0 ? ("port " + std::to_string(stream_port)).c_str() : "Disabled");

    try
    {
        // ==================== 1. 初始化线程池 ====================
        NN_LOG_INFO("Initializing thread pool with %d threads...", num_threads);
        std::string model_str(model_path);
        Yolov8ThreadPool pool;
        auto ret = pool.setUp(model_str, num_threads);
        if (ret != NN_SUCCESS)
        {
            NN_LOG_ERROR("Failed to initialize thread pool, error: %d", ret);
            return -1;
        }
        NN_LOG_INFO("Thread pool initialized: %d YOLOv8 instances for %d NPU cores",
                     num_threads, num_threads);

        // ==================== 2. 初始化跌倒检测器 ====================
        NN_LOG_INFO("Initializing fall detector...");
        FallDetector fall_detector;
        FallDetectConfig config = getDefaultFallConfig();
        config.fall_threshold = 0.55f;
        config.fall_confirm_frames = 3;
        config.recover_confirm_frames = 8;
        fall_detector.SetConfig(config);
        NN_LOG_INFO("Fall detector initialized");

        // ==================== 3. 打开摄像头 ====================
        NN_LOG_INFO("Opening camera %d...", camera_id);
        cv::VideoCapture cap;
        // 尝试多种后端打开摄像头：V4L2 → GStreamer → 默认
        int backends[] = {cv::CAP_V4L2, cv::CAP_GSTREAMER, cv::CAP_ANY};
        const char* backend_names[] = {"V4L2", "GStreamer", "ANY"};
        bool camera_opened = false;
        for (int b = 0; b < 3; b++)
        {
            NN_LOG_INFO("Trying camera %d with %s backend...", camera_id, backend_names[b]);
            cap.open(camera_id, backends[b]);
            if (cap.isOpened())
            {
                NN_LOG_INFO("Camera opened with %s backend", backend_names[b]);
                camera_opened = true;
                break;
            }
            NN_LOG_WARNING("Failed to open camera with %s backend", backend_names[b]);
        }
        if (!camera_opened)
        {
            NN_LOG_ERROR("Failed to open camera %d with all backends", camera_id);
            return -1;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_height);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        // 读取首帧需要重试（USB摄像头通常需要预热时间）
        cv::Mat test_frame;
        const int max_retries = 30;
        for (int retry = 0; retry < max_retries; retry++)
        {
            cap >> test_frame;
            if (!test_frame.empty())
                break;
            NN_LOG_INFO("Waiting for camera frame... (%d/%d)", retry + 1, max_retries);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (test_frame.empty())
        {
            NN_LOG_ERROR("Failed to read from camera %d after %d retries", camera_id, max_retries);
            return -1;
        }
        NN_LOG_INFO("Camera opened: %dx%d", test_frame.cols, test_frame.rows);

        // ==================== 4. 启动 MJPEG 推流 ====================
        MjpegServer* mjpeg_server = nullptr;
        if (stream_port > 0)
        {
            mjpeg_server = new MjpegServer(stream_port);
            if (!mjpeg_server->Start())
            {
                NN_LOG_WARNING("Failed to start MJPEG server on port %d", stream_port);
                delete mjpeg_server;
                mjpeg_server = nullptr;
            }
        }

        // ==================== 5. 启动线程 ====================
        NN_LOG_INFO("Starting threads...");

        // 读流线程
        std::thread read_thread(readThread, std::ref(cap), std::ref(pool));

        // 结果处理线程（包含跌倒检测 + 绘制 + MJPEG + 录像）
        std::thread result_thread(resultThread, std::ref(pool), std::ref(fall_detector),
                                  record, mjpeg_server);

        NN_LOG_INFO("All threads started. Press Ctrl+C to exit.");
        if (mjpeg_server)
        {
            NN_LOG_INFO("MJPEG stream: http://<board_ip>:%d/", stream_port);
        }

        // 等待读流线程结束
        read_thread.join();

        // 等待结果处理完成
        g_stop = true;
        result_thread.join();

        // 停止线程池
        pool.stopAll();

        // 停止 MJPEG 服务器
        if (mjpeg_server)
        {
            mjpeg_server->Stop();
            delete mjpeg_server;
        }

        cap.release();
        NN_LOG_INFO("Fall detection system stopped normally.");
        NN_LOG_INFO("Total frames processed: %d", g_frame_start_id.load());
    }
    catch (const std::exception& e)
    {
        NN_LOG_ERROR("Fatal exception: %s", e.what());
        return -1;
    }

    return 0;
}
