// 跌倒检测绘制模块

#ifndef RK3588_DEMO_FALL_DRAW_H
#define RK3588_DEMO_FALL_DRAW_H

#include <opencv2/opencv.hpp>
#include "types/fall_datatype.h"
#include "types/nn_datatype.h"

// 绘制跌倒检测结果（状态信息、报警框、特征数据）
void DrawFallResult(cv::Mat& img, const FallResult& result);

// 绘制跌倒报警（红色全屏警告）
void DrawFallAlarm(cv::Mat& img);

// 绘制特征调试信息
void DrawFallDebugInfo(cv::Mat& img, const FallResult& result, int x = 10, int y = 30);

// 绘制FPS信息
void DrawFPS(cv::Mat& img, float fps, int x = 10, int y = 10);

#endif // RK3588_DEMO_FALL_DRAW_H
