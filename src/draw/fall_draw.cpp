// 跌倒检测绘制模块实现

#include "fall_draw.h"

#include <string>
#include <sstream>
#include <iomanip>

// 绘制跌倒检测结果（紧凑版）
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

    // 右上角：状态 + 得分（紧凑一行）
    char buf[128];
    snprintf(buf, sizeof(buf), "State: %s  Score: %.2f", state_text.c_str(), result.fall_score);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int tx = img.cols - text_size.width - 10;

    // 半透明背景
    cv::Mat roi = img(cv::Rect(tx - 5, 5, text_size.width + 10, text_size.height + 10));
    cv::Mat overlay_roi;
    roi.clone().copyTo(overlay_roi);
    cv::rectangle(overlay_roi, cv::Point(0, 0), cv::Point(overlay_roi.cols - 1, overlay_roi.rows - 1),
                  cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay_roi, 0.5, roi, 0.5, 0, roi);

    cv::putText(img, buf, cv::Point(tx, text_size.height + 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1);

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
    int border_thickness = 10;
    cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols - 1, img.rows - 1),
                  cv::Scalar(0, 0, 255), border_thickness);

    // 绘制 "FALL DETECTED!" 警告文字（居中，较小字号）
    std::string alarm_text = "FALL DETECTED!";
    int baseline = 0;
    double font_scale = 1.2;
    int thickness = 2;
    cv::Size text_size = cv::getTextSize(alarm_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    cv::Point text_org((img.cols - text_size.width) / 2, (img.rows + text_size.height) / 2);

    // 绘制文字背景（黑色半透明矩形）
    cv::Rect bg_rect(text_org.x - 8, text_org.y - text_size.height - 8,
                     text_size.width + 16, text_size.height + 16);
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, bg_rect, cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.6, img, 0.4, 0, img);

    // 绘制红色警告文字
    cv::putText(img, alarm_text, text_org, cv::FONT_HERSHEY_SIMPLEX,
                font_scale, cv::Scalar(0, 0, 255), thickness);

    // 闪烁效果
    static int blink_count = 0;
    blink_count++;
    if ((blink_count / 10) % 2 == 0)
    {
        std::string icon = "!!!";
        cv::Size icon_size = cv::getTextSize(icon, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        cv::Point icon_org((img.cols - icon_size.width) / 2, text_org.y + text_size.height + 20);
        cv::putText(img, icon, icon_org, cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 0, 255), 2);
    }
}

// 绘制特征调试信息（紧凑版 - 左下角）
void DrawFallDebugInfo(cv::Mat& img, const FallResult& result, int x, int y)
{
    // 移到左下角，避免遮挡人体检测区域
    x = 5;
    y = img.rows - 5;

    cv::Scalar text_color(255, 255, 255);  // 白色
    int line_height = 14;
    double font_scale = 0.35;
    int thickness = 1;

    // 背景矩形（紧凑）
    int bg_width = 200;
    int bg_height = line_height * 9 + 6;
    int bg_y = y - bg_height;

    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, cv::Point(x - 3, bg_y), cv::Point(x + bg_width, y),
                  cv::Scalar(30, 30, 30), -1);
    cv::addWeighted(overlay, 0.6, img, 0.4, 0, img);

    // 标题
    int cur_y = bg_y + line_height;
    cv::putText(img, "Fall Detection Debug", cv::Point(x, cur_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 255), 1);

    // 各特征值（紧凑格式）
    auto draw_line = [&](const std::string& label, float value, int line_idx) {
        std::ostringstream ss;
        ss << label << std::fixed << std::setprecision(2) << value;
        cv::putText(img, ss.str(), cv::Point(x, cur_y + line_height * (line_idx + 1)),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
    };

    draw_line("Angle:    ", result.trunk_angle, 0);
    draw_line("Ratio:    ", result.bbox_ratio, 1);
    draw_line("HeadFoot: ", result.head_foot_dist, 2);
    draw_line("Gravity:  ", result.center_of_gravity, 3);
    draw_line("Tilt:     ", result.shoulder_tilt, 4);
    draw_line("Flat:     ", result.body_flatness, 5);
    draw_line("Score:    ", result.fall_score, 6);

    // 状态机信息
    std::ostringstream state_ss;
    state_ss << result.state_str
             << " F:" << result.fall_frame_count
             << " R:" << result.recover_frame_count;
    cv::putText(img, state_ss.str(), cv::Point(x, cur_y + line_height * 8),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), thickness);
}

// 绘制FPS信息（紧凑版）
void DrawFPS(cv::Mat& img, float fps, int x, int y)
{
    // 左上角
    x = 5;
    y = 15;

    char buf[32];
    snprintf(buf, sizeof(buf), "FPS: %.1f", fps);

    cv::Scalar color = (fps >= 25.0) ? cv::Scalar(0, 255, 0) :   // 绿色：流畅
                       (fps >= 15.0) ? cv::Scalar(0, 255, 255) : // 黄色：一般
                                       cv::Scalar(0, 0, 255);    // 红色：卡顿

    // 半透明背景
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
    cv::Mat roi = img(cv::Rect(x - 3, y - text_size.height - 3, text_size.width + 6, text_size.height + 6));
    cv::Mat overlay_roi;
    roi.clone().copyTo(overlay_roi);
    cv::rectangle(overlay_roi, cv::Point(0, 0), cv::Point(overlay_roi.cols - 1, overlay_roi.rows - 1),
                  cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay_roi, 0.5, roi, 0.5, 0, roi);

    cv::putText(img, buf, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
}
