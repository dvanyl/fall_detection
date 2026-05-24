// 跌倒检测数据类型定义

#ifndef RK3588_DEMO_FALL_DATATYPE_H
#define RK3588_DEMO_FALL_DATATYPE_H

#include <string>
#include <chrono>

// 跌倒状态枚举
typedef enum {
    FALL_STATE_STANDING = 0,    // 站立/正常状态
    FALL_STATE_FALLING = 1,     // 跌倒中（过渡态）
    FALL_STATE_FALLEN = 2,      // 已跌倒（触发报警）
    FALL_STATE_RECOVERING = 3   // 恢复中
} FallState;

// 跌倒检测结果结构体
typedef struct {
    FallState state;            // 当前状态
    float fall_score;           // 跌倒得分 (0.0 ~ 1.0)
    float trunk_angle;          // 躯干角度（度）
    float bbox_ratio;           // 身体宽高比
    float head_foot_dist;       // 头脚距离比
    float center_of_gravity;    // 重心高度（归一化）
    float shoulder_tilt;        // 肩膀倾斜角（度）
    float body_flatness;        // 身体展平度
    int fall_frame_count;       // 连续跌倒帧计数
    int recover_frame_count;    // 连续恢复帧计数
    bool is_alarm;              // 是否触发报警
    std::string state_str;      // 状态字符串描述
    std::chrono::steady_clock::time_point timestamp;  // 时间戳
} FallResult;

// 跌倒检测配置参数
typedef struct {
    float fall_threshold;           // 跌倒得分阈值 (默认0.6)
    int fall_confirm_frames;        // 确认跌倒所需连续帧数 (默认5)
    int recover_confirm_frames;     // 确认恢复所需连续帧数 (默认10)
    float trunk_angle_weight;       // 躯干角度权重 (默认0.3)
    float bbox_ratio_weight;        // 宽高比权重 (默认0.2)
    float head_foot_dist_weight;    // 头脚距离权重 (默认0.2)
    float center_of_gravity_weight; // 重心高度权重 (默认0.15)
    float shoulder_tilt_weight;     // 肩膀倾斜权重 (默认0.1)
    float body_flatness_weight;     // 身体展平度权重 (默认0.05)
    float trunk_angle_threshold;    // 躯干角度阈值 (默认60度)
    float bbox_ratio_threshold;     // 宽高比阈值 (默认1.2)
    float head_foot_dist_threshold; // 头脚距离阈值 (默认0.4)
    float center_of_gravity_threshold; // 重心高度阈值 (默认0.6)
    float shoulder_tilt_threshold;  // 肩膀倾斜阈值 (默认45度)
    float body_flatness_threshold;  // 身体展平度阈值 (默认0.15)
} FallDetectConfig;

// 默认配置
static FallDetectConfig getDefaultFallConfig() {
    FallDetectConfig config;
    config.fall_threshold = 0.6f;
    config.fall_confirm_frames = 5;
    config.recover_confirm_frames = 10;
    config.trunk_angle_weight = 0.3f;
    config.bbox_ratio_weight = 0.2f;
    config.head_foot_dist_weight = 0.2f;
    config.center_of_gravity_weight = 0.15f;
    config.shoulder_tilt_weight = 0.1f;
    config.body_flatness_weight = 0.05f;
    config.trunk_angle_threshold = 60.0f;
    config.bbox_ratio_threshold = 1.2f;
    config.head_foot_dist_threshold = 0.4f;
    config.center_of_gravity_threshold = 0.6f;
    config.shoulder_tilt_threshold = 45.0f;
    config.body_flatness_threshold = 0.15f;
    return config;
}

// 状态转字符串
static std::string fallStateToString(FallState state) {
    switch (state) {
        case FALL_STATE_STANDING:   return "STANDING";
        case FALL_STATE_FALLING:    return "FALLING";
        case FALL_STATE_FALLEN:     return "FALLEN";
        case FALL_STATE_RECOVERING: return "RECOVERING";
        default:                    return "UNKNOWN";
    }
}

#endif // RK3588_DEMO_FALL_DATATYPE_H
