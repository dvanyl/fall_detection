// YOLOv8-Pose 人体跌倒检测系统 - USB摄像头实时检测
// 基于RK3588 + RKNN + RGA硬件加速
//
// 用法: ./yolov8_fall_detect <model_path> [camera_id] [width] [height] [record]
//   model_path: RKNN模型文件路径
//   camera_id:  USB摄像头设备号 (默认0)
//   width:      摄像头分辨率宽度 (默认640)
//   height:     摄像头分辨率高度 (默认480)
//   record:     是否录像保存 (0=否, 1=是, 默认0)

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

#include "task/yolov8_custom.h"
#include "task/fall_detector.h"
#include "draw/cv_draw.h"
#include "draw/fall_draw.h"
#include "types/fall_datatype.h"
#include "utils/logging.h"

// 全局停止标志
static std::atomic<bool> g_stop(false);

// 信号处理函数
void signalHandler(int signum)
{
    NN_LOG_INFO("Received signal %d, stopping...", signum);
    g_stop = true;
}

// ==================== 多线程流水线架构 ====================
// 线程1: 读流线程 - 从USB摄像头读取帧
// 线程2: 推理线程 - YOLOv8-Pose推理 + 跌倒检测
// 线程3: 显示线程 - 绘制结果并显示

// 共享数据结构
struct FrameData
{
    cv::Mat frame;
    int frame_id;
    std::chrono::steady_clock::time_point read_time;
};

struct ResultData
{
    cv::Mat frame;
    int frame_id;
    std::vector<Detection> objects;
    std::vector<std::map<int, KeyPoint>> keypoints;
    FallResult fall_result;
    float inference_fps;
    float total_fps;
    std::chrono::steady_clock::time_point read_time;
};

// 线程安全队列
template <typename T>
class SafeQueue
{
public:
    void push(T item)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        // 保持队列大小，丢弃旧帧
        while (queue_.size() >= max_size_)
        {
            queue_.pop();
        }
        queue_.push(std::move(item));
        cond_.notify_one();
    }

    bool pop(T& item, int timeout_ms = 100)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cond_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [this] { return !queue_.empty() || g_stop; }))
        {
            return false;
        }
        if (queue_.empty())
            return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }

    void setMaxSize(size_t max_size) { max_size_ = max_size; }

private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cond_;  // 避免与 cv 命名空间冲突
    size_t max_size_ = 2;  // 最多缓存2帧，降低延迟
};

// ==================== 读流线程 ====================
void readThread(cv::VideoCapture& cap, SafeQueue<FrameData>& frame_queue)
{
    NN_LOG_INFO("Read thread started");
    int frame_id = 0;

    while (!g_stop)
    {
        FrameData data;
        data.read_time = std::chrono::steady_clock::now();
        cap >> data.frame;
        if (data.frame.empty())
        {
            NN_LOG_WARNING("Empty frame from camera, retrying...");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        data.frame_id = frame_id++;
        frame_queue.push(std::move(data));
    }

    NN_LOG_INFO("Read thread stopped");
}

// ==================== 推理线程 ====================
void inferenceThread(Yolov8Custom& yolo, FallDetector& fall_detector,
                     SafeQueue<FrameData>& frame_queue,
                     SafeQueue<ResultData>& result_queue,
                     std::atomic<float>& inference_fps,
                     std::atomic<float>& total_fps)
{
    NN_LOG_INFO("Inference thread started");

    while (!g_stop)
    {
        FrameData input;
        if (!frame_queue.pop(input, 100))
        {
            continue;
        }

        try
        {
            auto start_time = std::chrono::steady_clock::now();

            // YOLOv8-Pose推理（内部使用RGA硬件加速预处理）
            std::vector<Detection> objects;
            std::vector<std::map<int, KeyPoint>> keypoints;
            auto ret = yolo.Run(input.frame, objects, keypoints);
            if (ret != NN_SUCCESS)
            {
                NN_LOG_ERROR("YOLOv8 inference failed for frame %d, error: %d", input.frame_id, ret);
                continue;
            }

            auto inference_end = std::chrono::steady_clock::now();

            // 跌倒检测
            FallResult fall_result = fall_detector.Update(objects, keypoints);

            auto end_time = std::chrono::steady_clock::now();

            // 计算FPS
            float infer_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                inference_end - start_time).count() / 1000.0f;
            float total_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count() / 1000.0f;

            // 避免除零
            inference_fps = (infer_ms > 0.1f) ? (1000.0f / infer_ms) : 0.0f;
            total_fps = (total_ms > 0.1f) ? (1000.0f / total_ms) : 0.0f;

            // 组装结果
            ResultData result;
            result.frame = std::move(input.frame);
            result.frame_id = input.frame_id;
            result.objects = std::move(objects);
            result.keypoints = std::move(keypoints);
            result.fall_result = std::move(fall_result);
            result.inference_fps = inference_fps.load();
            result.total_fps = total_fps.load();
            result.read_time = input.read_time;

            result_queue.push(std::move(result));
        }
        catch (const std::exception& e)
        {
            NN_LOG_ERROR("Inference thread exception: %s", e.what());
        }
    }

    NN_LOG_INFO("Inference thread stopped");
}

// ==================== 显示/处理线程 ====================
void displayThread(SafeQueue<ResultData>& result_queue, bool record,
                   std::atomic<float>& inference_fps, std::atomic<float>& total_fps,
                   bool headless = false)
{
    NN_LOG_INFO("Display thread started (headless=%s)", headless ? "true" : "false");

    cv::VideoWriter writer;
    if (record)
    {
        // 获取第一帧的尺寸
        ResultData first_result;
        if (result_queue.pop(first_result, 5000))
        {
            int w = first_result.frame.cols;
            int h = first_result.frame.rows;
            writer = cv::VideoWriter("fall_detect_result.mp4",
                                     cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                     25, cv::Size(w, h));
            // 处理第一帧
            DrawDetections(first_result.frame, first_result.objects);
            DrawCocoKps(first_result.frame, first_result.keypoints);
            DrawFallResult(first_result.frame, first_result.fall_result);
            DrawFallDebugInfo(first_result.frame, first_result.fall_result);
            DrawFPS(first_result.frame, first_result.total_fps);
            writer << first_result.frame;
        }
    }

    // FPS平滑计算
    float smooth_fps = 0.0f;
    const float alpha = 0.3f;  // 指数移动平均系数
    // 帧计数（用于headless模式下定期输出日志）
    int log_interval = 0;

    while (!g_stop)
    {
        ResultData result;
        if (!result_queue.pop(result, 100))
        {
            continue;
        }

        // 绘制检测框和关键点
        DrawDetections(result.frame, result.objects);
        DrawCocoKps(result.frame, result.keypoints);

        // 绘制跌倒检测结果
        DrawFallResult(result.frame, result.fall_result);

        // 绘制调试信息
        DrawFallDebugInfo(result.frame, result.fall_result);

        // 绘制FPS
        smooth_fps = alpha * result.total_fps + (1.0f - alpha) * smooth_fps;
        DrawFPS(result.frame, smooth_fps);

        // 计算端到端延迟
        auto now = std::chrono::steady_clock::now();
        float latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            now - result.read_time).count() / 1000.0f;
        std::ostringstream latency_ss;
        latency_ss << "Latency: " << std::fixed << std::setprecision(1) << latency_ms << "ms";
        cv::putText(result.frame, latency_ss.str(), cv::Point(10, result.frame.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        // 录像
        if (record && writer.isOpened())
        {
            writer << result.frame;
        }

        // headless模式：无显示器，仅输出日志
        if (headless)
        {
            log_interval++;
            // 每30帧输出一次状态日志
            if (log_interval % 30 == 0)
            {
                NN_LOG_INFO("[Frame %d] FPS:%.1f, State:%s, FallScore:%.2f, Latency:%.1fms",
                            result.frame_id, smooth_fps,
                            result.fall_result.state_str.c_str(),
                            result.fall_result.fall_score,
                            latency_ms);
            }
            // 跌倒报警时立即输出
            if (result.fall_result.is_alarm)
            {
                NN_LOG_WARNING("!!! FALL DETECTED !!! Frame:%d, Score:%.2f, State:%s",
                               result.frame_id, result.fall_result.fall_score,
                               result.fall_result.state_str.c_str());
            }
        }
        else
        {
            // 有显示器模式：显示画面
            cv::imshow("YOLOv8-Pose Fall Detection - RK3588", result.frame);

            // 按ESC退出
            int key = cv::waitKey(1);
            if (key == 27)  // ESC
            {
                g_stop = true;
                break;
            }
            else if (key == 'r' || key == 'R')
            {
                // 按R重置跌倒检测器
                NN_LOG_INFO("Reset fall detector");
            }
        }
    }

    if (writer.isOpened())
    {
        writer.release();
    }

    cv::destroyAllWindows();
    NN_LOG_INFO("Display thread stopped");
}

// ==================== 主函数 ====================
int main(int argc, char** argv)
{
    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // 参数解析
    if (argc < 2)
    {
        std::cout << "========================================" << std::endl;
        std::cout << "  YOLOv8-Pose Fall Detection System" << std::endl;
        std::cout << "  Platform: RK3588 + RKNN + RGA" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Usage: " << argv[0]
                  << " <model_path> [camera_id] [width] [height] [record] [headless]" << std::endl;
        std::cout << "  model_path: RKNN model file path (.rknn)" << std::endl;
        std::cout << "  camera_id:  USB camera device id (default: 0)" << std::endl;
        std::cout << "  width:      Camera resolution width (default: 640)" << std::endl;
        std::cout << "  height:     Camera resolution height (default: 480)" << std::endl;
        std::cout << "  record:     Save video (0=no, 1=yes, default: 0)" << std::endl;
        std::cout << "  headless:   Headless mode without display (0=no, 1=yes, default: 0)" << std::endl;
        return -1;
    }

    const char* model_path = argv[1];
    int camera_id = argc > 2 ? atoi(argv[2]) : 0;
    int cam_width = argc > 3 ? atoi(argv[3]) : 640;
    int cam_height = argc > 4 ? atoi(argv[4]) : 480;
    bool record = argc > 5 ? (atoi(argv[5]) != 0) : false;
    bool headless = argc > 6 ? (atoi(argv[6]) != 0) : false;

    NN_LOG_INFO("========================================");
    NN_LOG_INFO("  YOLOv8-Pose Fall Detection System");
    NN_LOG_INFO("  Platform: RK3588 + RKNN + RGA");
    NN_LOG_INFO("========================================");
    NN_LOG_INFO("Model: %s", model_path);
    NN_LOG_INFO("Camera: %d (%dx%d)", camera_id, cam_width, cam_height);
    NN_LOG_INFO("Record: %s", record ? "Yes" : "No");
    NN_LOG_INFO("Headless: %s", headless ? "Yes" : "No");

    try
    {
        // ==================== 初始化 ====================

        // 1. 初始化YOLOv8-Pose模型
        NN_LOG_INFO("Loading YOLOv8-Pose model...");
        nn_model_type_e model_type = NN_YOLOV8_POSE;
        Yolov8Custom yolo(model_type);
        auto ret = yolo.LoadModel(model_path);
        if (ret != NN_SUCCESS)
        {
            NN_LOG_ERROR("Failed to load model: %s, error: %d", model_path, ret);
            return -1;
        }
        NN_LOG_INFO("Model loaded successfully");

        // 2. 初始化跌倒检测器
        NN_LOG_INFO("Initializing fall detector...");
        FallDetector fall_detector;
        FallDetectConfig config = getDefaultFallConfig();
        // 可根据实际场景调整参数
        config.fall_threshold = 0.55f;       // 跌倒得分阈值
        config.fall_confirm_frames = 3;      // 连续3帧确认跌倒（降低延迟）
        config.recover_confirm_frames = 8;   // 连续8帧确认恢复（避免误恢复）
        fall_detector.SetConfig(config);
        NN_LOG_INFO("Fall detector initialized");

        // 3. 打开USB摄像头
        NN_LOG_INFO("Opening camera %d...", camera_id);
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened())
        {
            NN_LOG_ERROR("Failed to open camera %d", camera_id);
            return -1;
        }
        // 设置摄像头分辨率
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_height);
        // 设置缓冲区大小为1，降低延迟
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        // 读取一帧验证
        cv::Mat test_frame;
        cap >> test_frame;
        if (test_frame.empty())
        {
            NN_LOG_ERROR("Failed to read from camera %d", camera_id);
            cap.release();
            return -1;
        }
        int actual_width = test_frame.cols;
        int actual_height = test_frame.rows;
        NN_LOG_INFO("Camera opened: %dx%d", actual_width, actual_height);

        // ==================== 启动多线程流水线 ====================
        NN_LOG_INFO("Starting multi-thread pipeline...");

        // 创建线程安全队列
        SafeQueue<FrameData> frame_queue;
        SafeQueue<ResultData> result_queue;
        frame_queue.setMaxSize(2);
        result_queue.setMaxSize(2);

        // FPS统计
        std::atomic<float> inference_fps(0.0f);
        std::atomic<float> total_fps(0.0f);

        // 启动线程
        std::thread read_thread(readThread, std::ref(cap), std::ref(frame_queue));
        std::thread infer_thread(inferenceThread, std::ref(yolo), std::ref(fall_detector),
                                 std::ref(frame_queue), std::ref(result_queue),
                                 std::ref(inference_fps), std::ref(total_fps));
        std::thread disp_thread(displayThread, std::ref(result_queue), record,
                                std::ref(inference_fps), std::ref(total_fps), headless);

        if (headless)
            NN_LOG_INFO("All threads started. Running in headless mode (no display).");
        else
            NN_LOG_INFO("All threads started. Press ESC to exit.");

        // 等待显示线程结束（用户按ESC或收到信号）
        disp_thread.join();

        // 停止其他线程
        g_stop = true;
        if (read_thread.joinable()) read_thread.join();
        if (infer_thread.joinable()) infer_thread.join();

        // 释放摄像头
        cap.release();

        NN_LOG_INFO("Fall detection system stopped normally.");
    }
    catch (const std::exception& e)
    {
        NN_LOG_ERROR("Fatal exception: %s", e.what());
        g_stop = true;
        return -1;
    }
    catch (...)
    {
        NN_LOG_ERROR("Unknown fatal exception occurred");
        g_stop = true;
        return -1;
    }

    return 0;
}
