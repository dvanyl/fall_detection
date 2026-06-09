
#include <opencv2/opencv.hpp>

#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

#include "task/yolov8_thread_pool.h"

static int g_frame_start_id = 0;
static int g_frame_end_id = 0;

static Yolov8ThreadPool *g_pool = nullptr;
bool end = false;

// 获取结果线程
void get_results(int width = 1280, int height = 720, int fps = 30)
{
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true)
    {
        // 获取完整结果（检测框 + 关键点）
        std::vector<Detection> objects;
        std::vector<std::map<int, KeyPoint>> keypoints;
        auto ret = g_pool->getTargetResultFull(objects, keypoints, g_frame_end_id);

        if (end && ret != NN_SUCCESS)
        {
            g_pool->stopAll();
            break;
        }

        if (ret == NN_SUCCESS)
        {
            // 获取原始帧并绘制
            cv::Mat img;
            g_pool->getTargetImgResult(img, g_frame_end_id);

            if (!img.empty())
            {
                DrawDetections(img, objects);
                DrawCocoKps(img, keypoints);
            }
        }

        g_frame_end_id++;

        // 计算FPS
        frame_count++;
        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        if (elapsed_all_2 > 1000)
        {
            NN_LOG_INFO("ThreadPool FPS:%.1f, Frames:%d, Queue:%d",
                        frame_count / (elapsed_all_2 / 1000.0f), frame_count, g_pool->getTaskQueueSize());
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->stopAll();
    NN_LOG_INFO("Get results end.");
}

// 读取视频帧，提交任务
void read_stream(const char *video_file)
{
    cv::VideoCapture cap(video_file);
    if (!cap.isOpened())
    {
        NN_LOG_ERROR("Failed to open video file: %s", video_file);
        return;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    NN_LOG_INFO("Video size: %d x %d, fps: %d", width, height, fps);

    cv::Mat img;
    while (true)
    {
        cap >> img;
        if (img.empty())
        {
            NN_LOG_INFO("Video end.");
            end = true;
            break;
        }
        g_pool->submitTask(img, g_frame_start_id++);
    }
    cap.release();
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        NN_LOG_ERROR("Usage: %s <model_file> <video_file> [num_threads]", argv[0]);
        return -1;
    }

    std::string model_file = argv[1];
    const char *video_file = argv[2];
    const int num_threads = (argc > 3) ? atoi(argv[3]) : 3;

    // 创建线程池
    g_pool = new Yolov8ThreadPool();
    g_pool->setUp(model_file, num_threads);

    // 读取视频线程
    std::thread read_stream_thread(read_stream, video_file);
    // 结果处理线程
    std::thread result_thread(get_results, 1280, 720, 25);

    read_stream_thread.join();
    result_thread.join();

    delete g_pool;
    return 0;
}
