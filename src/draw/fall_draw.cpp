// 跌倒检测绘制模块实现

#include "fall_draw.h"

#include <string>
#include <sstream>
#include <iomanip>

// 绘制跌倒检测结果
void DrawFallResult(cv::Mat& img, const FallResult& result)
{
    // 根据状态选择颜色
    cv::Scalar state_color;
    std::string state_text;

    switch (result.state)
    {
    case FALL_STATE_STANDING:
        state_color = cv::Scalar(0, 255, 0);  // 绿色
        state_text = "STANDING";
        break;
    case FALL_STATE_FALLING:
        state_color = cv::Scalar(0, 255, 255);  // 黄色
        state_text = "FALLING...";
        break;
    case FALL_STATE_FALLEN:
        state_color = cv::Scalar(0, 0, 255);  // 红色
        state_text = "FALLEN!";
        break;
    case FALL_STATE_RECOVERING:
        state_color = cv::Scalar(0, 165, 255);  // 橙色
        state_text = "RECOVERING";
        break;
    default:
        state_color = cv::Scalar(255, 255, 255);
        state_text = "UNKNOWN";
        break;
    }

    // 绘制状态文字（右上角）
    int baseline = 0;
    std::string display_text = "State: " + state_text;
    cv::Size text_size = cv::getTextSize(display_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    cv::Point text_org(img.cols - text_size.width - 20, 35);
    cv::putText(img, display_text, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2);

    // 绘制跌倒得分
    std::ostringstream score_ss;
    score_ss << "Fall Score: " << std::fixed << std::setprecision(2) << result.fall_score;
    cv::putText(img, score_ss.str(), cv::Point(img.cols - text_size.width - 20, 65),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2);

    // 如果触发报警，绘制红色警告框
    if (result.is_alarm)
    {
        DrawFallAlarm(img);
    }
}

// 绘制跌倒报警（红色全屏警告）
void DrawFallAlarm(cv::Mat& img)
{
    // 绘制红色半透明边框
    int border_thickness = 15;
    cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols - 1, img.rows - 1),
                  cv::Scalar(0, 0, 255), border_thickness);

    // 绘制 "FALL DETECTED!" 警告文字（居中）
    std::string alarm_text = "FALL DETECTED!";
    int baseline = 0;
    double font_scale = 2.0;
    int thickness = 4;
    cv::Size text_size = cv::getTextSize(alarm_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    cv::Point text_org((img.cols - text_size.width) / 2, (img.rows + text_size.height) / 2);

    // 绘制文字背景（黑色半透明矩形）
    cv::Rect bg_rect(text_org.x - 10, text_org.y - text_size.height - 10,
                     text_size.width + 20, text_size.height + 20);
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, bg_rect, cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.6, img, 0.4, 0, img);

    // 绘制红色警告文字
    cv::putText(img, alarm_text, text_org, cv::FONT_HERSHEY_SIMPLEX,
                font_scale, cv::Scalar(0, 0, 255), thickness);

    // 绘制闪烁效果（基于帧计数，这里用简单的交替显示）
    static int blink_count = 0;
    blink_count++;
    if ((blink_count / 10) % 2 == 0)
    {
        // 绘制额外的警告图标 "!!!"
        std::string icon = "!!!";
        cv::Size icon_size = cv::getTextSize(icon, cv::FONT_HERSHEY_SIMPLEX, 1.5, 3, &baseline);
        cv::Point icon_org((img.cols - icon_size.width) / 2, text_org.y + text_size.height + 30);
        cv::putText(img, icon, icon_org, cv::FONT_HERSHEY_SIMPLEX,
                    1.5, cv::Scalar(0, 0, 255), 3);
    }
}

// 绘制特征调试信息
void DrawFallDebugInfo(cv::Mat& img, const FallResult& result, int x, int y)
{
    cv::Scalar text_color(255, 255, 255);  // 白色
    int line_height = 22;
    double font_scale = 0.45;
    int thickness = 1;

    // 背景矩形
    int bg_width = 280;
    int bg_height = line_height * 9 + 10;
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, cv::Point(x - 5, y - 18), cv::Point(x + bg_width, y + bg_height),
                  cv::Scalar(50, 50, 50), -1);
    cv::addWeighted(overlay, 0.7, img, 0.3, 0, img);

    // 标题
    cv::putText(img, "=== Fall Detection Debug ===", cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    // 各特征值
    auto draw_line = [&](const std::string& label, float value, int line_idx) {
        std::ostringstream ss;
        ss << label << std::fixed << std::setprecision(2) << value;
        cv::putText(img, ss.str(), cv::Point(x, y + line_height * (line_idx + 1)),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
    };

    draw_line("Trunk Angle:    ", result.trunk_angle, 0);
    draw_line("BBox Ratio:     ", result.bbox_ratio, 1);
    draw_line("Head-Foot Dist: ", result.head_foot_dist, 2);
    draw_line("Center Gravity: ", result.center_of_gravity, 3);
    draw_line("Shoulder Tilt:  ", result.shoulder_tilt, 4);
    draw_line("Body Flatness:  ", result.body_flatness, 5);
    draw_line("Fall Score:     ", result.fall_score, 6);

    // 状态机信息
    std::ostringstream state_ss;
    state_ss << "State: " << result.state_str
             << " | FallCnt:" << result.fall_frame_count
             << " RecCnt:" << result.recover_frame_count;
    cv::putText(img, state_ss.str(), cv::Point(x, y + line_height * 8),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), thickness);
}

// 绘制FPS信息
void DrawFPS(cv::Mat& img, float fps, int x, int y)
{
    std::ostringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(1) << fps;

    cv::Scalar color = (fps >= 25.0) ? cv::Scalar(0, 255, 0) :   // 绿色：流畅
                       (fps >= 15.0) ? cv::Scalar(0, 255, 255) : // 黄色：一般
                                       cv::Scalar(0, 0, 255);    // 红色：卡顿

    cv::putText(img, ss.str(), cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}
