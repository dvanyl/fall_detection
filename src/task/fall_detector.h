// 跌倒检测器 - 基于YOLOv8-Pose关键点的跌倒检测

#ifndef RK3588_DEMO_FALL_DETECTOR_H
#define RK3588_DEMO_FALL_DETECTOR_H

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "types/nn_datatype.h"
#include "types/fall_datatype.h"

class FallDetector {
public:
    FallDetector();
    ~FallDetector();

    // 设置检测参数
    void SetConfig(const FallDetectConfig& config);

    // 核心接口：输入关键点和检测框，输出跌倒结果
    FallResult Update(const std::vector<Detection>& objects,
                      const std::vector<std::map<int, KeyPoint>>& keypoints);

    // 获取当前状态
    FallState GetState() const;

    // 获取跌倒得分（用于调试）
    float GetFallScore() const;

    // 重置状态机
    void Reset();

private:
    // 特征提取函数
    float CalcTrunkAngle(const std::map<int, KeyPoint>& kps);
    float CalcBboxRatio(const Detection& det);
    float CalcHeadFootDistance(const std::map<int, KeyPoint>& kps);
    float CalcCenterOfGravity(const std::map<int, KeyPoint>& kps);
    float CalcShoulderTilt(const std::map<int, KeyPoint>& kps);
    float CalcBodyFlatness(const std::map<int, KeyPoint>& kps);

    // 跌倒得分计算
    float CalcFallScore(const Detection& det, const std::map<int, KeyPoint>& kps);

    // 状态机更新
    FallState UpdateStateMachine(float fall_score);

    // 辅助函数：计算两点距离
    float Distance(const KeyPoint& p1, const KeyPoint& p2);

    // 辅助函数：计算角度（度）
    float CalcAngle(float dx, float dy);

    // 配置参数
    FallDetectConfig config_;

    // 状态机变量
    FallState current_state_;
    int fall_frame_count_;      // 连续跌倒帧计数
    int recover_frame_count_;   // 连续恢复帧计数
    float last_fall_score_;     // 上一帧跌倒得分
};

#endif // RK3588_DEMO_FALL_DETECTOR_H
