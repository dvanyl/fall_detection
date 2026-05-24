
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <errno.h>

#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

// 创建目录（递归创建，兼容嵌入式系统）
static int create_directory(const char *path)
{
    char tmp[512];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++)
    {
        if (*p == '/')
        {
            *p = 0;
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
                return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
        return -1;
    return 0;
}

int main(int argc, char **argv)
{
    // 参数检查
    if (argc < 3)
    {
        NN_LOG_ERROR("Usage: %s <model_file> <video_file> [output_file]", argv[0]);
        NN_LOG_ERROR("  model_file: RKNN model path");
        NN_LOG_ERROR("  video_file: input video path");
        NN_LOG_ERROR("  output_file: output video path (default: output/result_pose.mp4)");
        return -1;
    }

    // model file path
    const char *model_file = argv[1];
    // input video
    const char *video_file = argv[2];
    // 输出视频路径（可选参数，默认 output/result_pose.mp4）
    const char *output_file = argc > 3 ? argv[3] : "output/result_pose.mp4";

    // 读取视频
    cv::VideoCapture cap(video_file);
    if (!cap.isOpened())
    {
        NN_LOG_ERROR("Failed to open video file: %s", video_file);
        return -1;
    }
    // 获取视频尺寸、帧率
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int video_fps = (int)cap.get(cv::CAP_PROP_FPS);
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    NN_LOG_INFO("Video size: %d x %d, fps: %d, total frames: %d", width, height, video_fps, total_frames);

    // 验证视频参数有效性
    if (video_fps <= 0)
    {
        video_fps = 25; // 默认帧率
        NN_LOG_WARNING("Invalid video fps, using default: %d", video_fps);
    }

    // 初始化
    nn_model_type_e model_type = NN_YOLOV8_POSE;
    Yolov8Custom yolo(model_type);
    // 加载模型
    auto load_ret = yolo.LoadModel(model_file);
    if (load_ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("Failed to load model: %s", model_file);
        return -1;
    }

    // 创建输出视频目录
    std::string output_path(output_file);
    std::string output_dir = output_path.substr(0, output_path.find_last_of('/'));
    if (!output_dir.empty())
    {
        if (create_directory(output_dir.c_str()) != 0)
        {
            NN_LOG_ERROR("Failed to create output directory: %s", output_dir.c_str());
            return -1;
        }
    }

    // 创建 VideoWriter，保存带检测结果的视频
    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer = cv::VideoWriter(output_file, fourcc, video_fps, cv::Size(width, height));
    if (!writer.isOpened())
    {
        NN_LOG_ERROR("Failed to create output video writer: %s", output_file);
        NN_LOG_ERROR("Trying alternative codec (XVID)...");
        fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        writer = cv::VideoWriter(output_file, fourcc, video_fps, cv::Size(width, height));
        if (!writer.isOpened())
        {
            NN_LOG_ERROR("Failed to create output video writer with XVID codec");
            return -1;
        }
    }
    NN_LOG_INFO("Output video will be saved to: %s", output_file);

    // 视频帧
    cv::Mat img;
    // 预分配半透明覆盖层（避免每帧 clone）
    cv::Mat overlay(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    // all start
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    int total_frame_count = 0;

    while (true)
    {
        // 开始计时
        auto start_1 = std::chrono::high_resolution_clock::now();

        // 读取视频帧
        cap >> img;
        if (img.empty())
        {
            NN_LOG_INFO("Video end.");
            break;
        }
        total_frame_count++;

        // 记录读取视频帧的时间
        auto end_1 = std::chrono::high_resolution_clock::now();
        auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count() / 1000.0;

        // 开始计时
        auto start_2 = std::chrono::high_resolution_clock::now();
        // 检测结果
        std::vector<Detection> objects;
        // 关键点结果
        std::vector<std::map<int, KeyPoint>> kps;
        // 推理
        yolo.Run(img, objects, kps);

        // 绘制检测框和关键点
        DrawDetections(img, objects);
        DrawCocoKps(img, kps);

        // 结束计时
        auto end_2 = std::chrono::high_resolution_clock::now();
        auto elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count() / 1000.0;

        // 计算总耗时和帧率
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_1).count() / 1000.0;
        auto current_fps = (duration > 0) ? (1000.0f / duration) : 0.0f;

        // 控制台输出
        NN_LOG_INFO("Frame %d | Time: %.1fms (read: %.1fms, infer: %.1fms) | FPS: %.1f | Objects: %zu",
                     total_frame_count, duration, elapsed_1, elapsed_2, current_fps, objects.size());

        // ====== 在帧上绘制信息 ======
        // 绘制半透明背景条（使用 ROI 避免全图 clone）
        cv::Rect info_bar(0, 0, width, 70);
        cv::Mat roi = img(info_bar);
        cv::Mat roi_result;
        cv::addWeighted(overlay(info_bar), 0.5, roi, 0.5, 0, roi_result);
        roi_result.copyTo(roi);

        // 绘制 FPS 信息（绿色）
        char info_buf[256];
        snprintf(info_buf, sizeof(info_buf), "FPS: %.1f  |  Infer: %.1fms  |  Objects: %zu",
                 current_fps, elapsed_2, objects.size());
        cv::putText(img, info_buf, cv::Point(15, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 绘制帧号信息（黄色）
        if (total_frames > 0)
        {
            snprintf(info_buf, sizeof(info_buf), "Frame: %d/%d  |  Size: %dx%d",
                     total_frame_count, total_frames, width, height);
        }
        else
        {
            snprintf(info_buf, sizeof(info_buf), "Frame: %d  |  Size: %dx%d",
                     total_frame_count, width, height);
        }
        cv::putText(img, info_buf, cv::Point(15, 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        // 如果有检测到的目标，在左下角显示每个目标的置信度
        if (!objects.empty())
        {
            int y_offset = height - 20;
            int max_display = std::min((int)objects.size(), 5);
            for (int i = 0; i < max_display; i++)
            {
                snprintf(info_buf, sizeof(info_buf), "Person %d: conf=%.2f",
                         i + 1, objects[i].confidence);
                cv::putText(img, info_buf, cv::Point(15, y_offset),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                y_offset -= 20;
            }
        }

        // 写入视频帧
        writer << img;

        // 每隔1秒打印一次平均帧率
        frame_count++;
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_all).count() / 1000.f;
        if (elapsed_all_2 > 1000)
        {
            NN_LOG_INFO("=== Avg FPS: %.1f, Frames in 1s: %d ===",
                         frame_count / (elapsed_all_2 / 1000.0f), frame_count);
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }

    // 释放资源
    writer.release();
    cap.release();

    NN_LOG_INFO("Total frames processed: %d", total_frame_count);
    NN_LOG_INFO("Output video saved to: %s", output_file);

    return 0;
}
