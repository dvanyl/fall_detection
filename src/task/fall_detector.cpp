// 跌倒检测器实现 - 基于YOLOv8-Pose关键点的跌倒检测

#include "fall_detector.h"

#include <cmath>
#include <algorithm>
#include <numeric>

#include "utils/logging.h"

// COCO 17个关键点索引定义
// 0:鼻子 1:左眼 2:右眼 3:左耳 4:右耳
// 5:左肩 6:右肩 7:左肘 8:右肘 9:左腕 10:右腕
// 11:左髋 12:右髋 13:左膝 14:右膝 15:左踝 16:右踝

#define KP_NOSE     0
#define KP_L_EYE    1
#define KP_R_EYE    2
#define KP_L_EAR    3
#define KP_R_EAR    4
#define KP_L_SHOULDER 5
#define KP_R_SHOULDER 6
#define KP_L_ELBOW  7
#define KP_R_ELBOW  8
#define KP_L_WRIST  9
#define KP_R_WRIST  10
#define KP_L_HIP    11
#define KP_R_HIP    12
#define KP_L_KNEE   13
#define KP_R_KNEE   14
#define KP_L_ANKLE  15
#define KP_R_ANKLE  16

// 最小关键点置信度阈值
static const float MIN_KP_SCORE = 0.3f;

FallDetector::FallDetector()
    : current_state_(FALL_STATE_STANDING)
    , fall_frame_count_(0)
    , recover_frame_count_(0)
    , last_fall_score_(0.0f)
{
    config_ = getDefaultFallConfig();
}

FallDetector::~FallDetector()
{
}

void FallDetector::SetConfig(const FallDetectConfig& config)
{
    config_ = config;
}

void FallDetector::Reset()
{
    current_state_ = FALL_STATE_STANDING;
    fall_frame_count_ = 0;
    recover_frame_count_ = 0;
    last_fall_score_ = 0.0f;
}

FallState FallDetector::GetState() const
{
    return current_state_;
}

float FallDetector::GetFallScore() const
{
    return last_fall_score_;
}

// 辅助函数：计算两点距离
float FallDetector::Distance(const KeyPoint& p1, const KeyPoint& p2)
{
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// 辅助函数：计算角度（度）
float FallDetector::CalcAngle(float dx, float dy)
{
    // 计算与垂直方向的夹角
    // 垂直方向为 (0, 1)，所以角度 = atan2(|dx|, |dy|)
    return std::atan2(std::abs(dx), std::abs(dy)) * 180.0f / M_PI;
}

// 特征1：躯干角度 - 肩膀中点→髋部中点连线与垂直方向夹角
float FallDetector::CalcTrunkAngle(const std::map<int, KeyPoint>& kps)
{
    // 检查关键点是否存在且置信度足够
    auto has_kp = [&](int idx) -> bool {
        auto it = kps.find(idx);
        return it != kps.end() && it->second.score > MIN_KP_SCORE;
    };

    // 优先使用双肩+双髋
    if (has_kp(KP_L_SHOULDER) && has_kp(KP_R_SHOULDER) && has_kp(KP_L_HIP) && has_kp(KP_R_HIP))
    {
        const auto& ls = kps.at(KP_L_SHOULDER);
        const auto& rs = kps.at(KP_R_SHOULDER);
        const auto& lh = kps.at(KP_L_HIP);
        const auto& rh = kps.at(KP_R_HIP);

        // 肩膀中点
        float mid_shoulder_x = (ls.x + rs.x) / 2.0f;
        float mid_shoulder_y = (ls.y + rs.y) / 2.0f;
        // 髋部中点
        float mid_hip_x = (lh.x + rh.x) / 2.0f;
        float mid_hip_y = (lh.y + rh.y) / 2.0f;

        float dx = mid_shoulder_x - mid_hip_x;
        float dy = mid_shoulder_y - mid_hip_y;

        return CalcAngle(dx, dy);
    }
    // 降级：使用单侧肩+髋
    else if (has_kp(KP_L_SHOULDER) && has_kp(KP_L_HIP))
    {
        const auto& s = kps.at(KP_L_SHOULDER);
        const auto& h = kps.at(KP_L_HIP);
        return CalcAngle(s.x - h.x, s.y - h.y);
    }
    else if (has_kp(KP_R_SHOULDER) && has_kp(KP_R_HIP))
    {
        const auto& s = kps.at(KP_R_SHOULDER);
        const auto& h = kps.at(KP_R_HIP);
        return CalcAngle(s.x - h.x, s.y - h.y);
    }

    return 0.0f; // 无法计算
}

// 特征2：身体宽高比 - bbox宽度/bbox高度
float FallDetector::CalcBboxRatio(const Detection& det)
{
    float w = static_cast<float>(det.box.width);
    float h = static_cast<float>(det.box.height);
    if (h < 1.0f) h = 1.0f; // 避免除零
    return w / h;
}

// 特征3：头脚距离比 - 头部y坐标与脚部y坐标的差值相对于bbox高度的比例
float FallDetector::CalcHeadFootDistance(const std::map<int, KeyPoint>& kps)
{
    auto has_kp = [&](int idx) -> bool {
        auto it = kps.find(idx);
        return it != kps.end() && it->second.score > MIN_KP_SCORE;
    };

    float head_y = 0.0f;
    float foot_y = 0.0f;
    bool has_head = false;
    bool has_foot = false;

    // 头部：取鼻子或眼睛的y坐标
    if (has_kp(KP_NOSE))
    {
        head_y = kps.at(KP_NOSE).y;
        has_head = true;
    }
    else if (has_kp(KP_L_EYE))
    {
        head_y = kps.at(KP_L_EYE).y;
        has_head = true;
    }
    else if (has_kp(KP_R_EYE))
    {
        head_y = kps.at(KP_R_EYE).y;
        has_head = true;
    }

    // 脚部：取脚踝的y坐标
    float l_ankle_y = 0.0f, r_ankle_y = 0.0f;
    bool has_l_ankle = has_kp(KP_L_ANKLE);
    bool has_r_ankle = has_kp(KP_R_ANKLE);

    if (has_l_ankle && has_r_ankle)
    {
        foot_y = std::max(kps.at(KP_L_ANKLE).y, kps.at(KP_R_ANKLE).y);
        has_foot = true;
    }
    else if (has_l_ankle)
    {
        foot_y = kps.at(KP_L_ANKLE).y;
        has_foot = true;
    }
    else if (has_r_ankle)
    {
        foot_y = kps.at(KP_R_ANKLE).y;
        has_foot = true;
    }
    // 降级：使用膝盖
    else if (has_kp(KP_L_KNEE))
    {
        foot_y = kps.at(KP_L_KNEE).y;
        has_foot = true;
    }
    else if (has_kp(KP_R_KNEE))
    {
        foot_y = kps.at(KP_R_KNEE).y;
        has_foot = true;
    }

    if (has_head && has_foot)
    {
        // 头脚距离越小，说明身体越水平（跌倒）
        // 返回值越小表示越可能跌倒
        float dist = std::abs(foot_y - head_y);
        // 归一化：假设图像高度为1.0（关键点已经是归一化坐标）
        return dist;
    }

    return 1.0f; // 默认返回较大值（站立）
}

// 特征4：重心高度 - 关键点y坐标均值（归一化）
float FallDetector::CalcCenterOfGravity(const std::map<int, KeyPoint>& kps)
{
    float sum_y = 0.0f;
    int count = 0;

    for (const auto& kp : kps)
    {
        if (kp.second.score > MIN_KP_SCORE)
        {
            sum_y += kp.second.y;
            count++;
        }
    }

    if (count > 0)
    {
        return sum_y / count;
    }

    return 0.5f; // 默认中间位置
}

// 特征5：肩膀倾斜角 - 左右肩连线与水平方向夹角
float FallDetector::CalcShoulderTilt(const std::map<int, KeyPoint>& kps)
{
    auto has_kp = [&](int idx) -> bool {
        auto it = kps.find(idx);
        return it != kps.end() && it->second.score > MIN_KP_SCORE;
    };

    if (has_kp(KP_L_SHOULDER) && has_kp(KP_R_SHOULDER))
    {
        const auto& ls = kps.at(KP_L_SHOULDER);
        const auto& rs = kps.at(KP_R_SHOULDER);

        float dx = rs.x - ls.x;
        float dy = rs.y - ls.y;

        // 与水平方向的夹角
        return std::atan2(std::abs(dy), std::abs(dx)) * 180.0f / M_PI;
    }

    return 0.0f;
}

// 特征6：身体展平度 - 关键点y坐标的标准差（归一化）
float FallDetector::CalcBodyFlatness(const std::map<int, KeyPoint>& kps)
{
    std::vector<float> y_values;
    for (const auto& kp : kps)
    {
        if (kp.second.score > MIN_KP_SCORE)
        {
            y_values.push_back(kp.second.y);
        }
    }

    if (y_values.size() < 3)
    {
        return 0.0f;
    }

    // 计算均值
    float mean = std::accumulate(y_values.begin(), y_values.end(), 0.0f) / y_values.size();

    // 计算标准差
    float variance = 0.0f;
    for (float y : y_values)
    {
        float diff = y - mean;
        variance += diff * diff;
    }
    variance /= y_values.size();
    float stddev = std::sqrt(variance);

    return stddev;
}

// 综合跌倒得分计算
float FallDetector::CalcFallScore(const Detection& det, const std::map<int, KeyPoint>& kps)
{
    // 提取各特征
    float trunk_angle = CalcTrunkAngle(kps);
    float bbox_ratio = CalcBboxRatio(det);
    float head_foot_dist = CalcHeadFootDistance(kps);
    float center_of_gravity = CalcCenterOfGravity(kps);
    float shoulder_tilt = CalcShoulderTilt(kps);
    float body_flatness = CalcBodyFlatness(kps);

    // 将各特征映射到 [0, 1] 得分
    // 特征值越大越可能跌倒

    // 1. 躯干角度得分：角度越大越可能跌倒
    float angle_score = std::min(trunk_angle / config_.trunk_angle_threshold, 1.0f);

    // 2. 宽高比得分：比值越大越可能跌倒（身体横向展开）
    float ratio_score = std::min(bbox_ratio / config_.bbox_ratio_threshold, 1.0f);

    // 3. 头脚距离得分：距离越小越可能跌倒（头和脚接近同一水平线）
    float dist_score = 1.0f - std::min(head_foot_dist / config_.head_foot_dist_threshold, 1.0f);
    dist_score = std::max(dist_score, 0.0f);

    // 4. 重心得分：重心越高（y值越大）越可能跌倒
    float gravity_score = std::min(center_of_gravity / config_.center_of_gravity_threshold, 1.0f);

    // 5. 肩膀倾斜得分
    float tilt_score = std::min(shoulder_tilt / config_.shoulder_tilt_threshold, 1.0f);

    // 6. 身体展平度得分：展平度越小越可能跌倒（身体水平时y值差异小）
    float flatness_score = 1.0f - std::min(body_flatness / config_.body_flatness_threshold, 1.0f);
    flatness_score = std::max(flatness_score, 0.0f);

    // 加权融合
    float fall_score = config_.trunk_angle_weight * angle_score +
                       config_.bbox_ratio_weight * ratio_score +
                       config_.head_foot_dist_weight * dist_score +
                       config_.center_of_gravity_weight * gravity_score +
                       config_.shoulder_tilt_weight * tilt_score +
                       config_.body_flatness_weight * flatness_score;

    // 限制在 [0, 1] 范围内
    fall_score = std::max(0.0f, std::min(fall_score, 1.0f));

    return fall_score;
}

// 状态机更新
FallState FallDetector::UpdateStateMachine(float fall_score)
{
    bool is_fall = fall_score > config_.fall_threshold;

    switch (current_state_)
    {
    case FALL_STATE_STANDING:
        if (is_fall)
        {
            fall_frame_count_++;
            if (fall_frame_count_ >= config_.fall_confirm_frames)
            {
                current_state_ = FALL_STATE_FALLEN;
                fall_frame_count_ = 0;
                NN_LOG_INFO("State: STANDING -> FALLEN (fall_score=%.2f)", fall_score);
            }
            else
            {
                current_state_ = FALL_STATE_FALLING;
            }
        }
        else
        {
            fall_frame_count_ = 0;
        }
        break;

    case FALL_STATE_FALLING:
        if (is_fall)
        {
            fall_frame_count_++;
            if (fall_frame_count_ >= config_.fall_confirm_frames)
            {
                current_state_ = FALL_STATE_FALLEN;
                fall_frame_count_ = 0;
                NN_LOG_INFO("State: FALLING -> FALLEN (fall_score=%.2f)", fall_score);
            }
        }
        else
        {
            // 跌倒得分降低，回到站立
            fall_frame_count_ = 0;
            current_state_ = FALL_STATE_STANDING;
            NN_LOG_INFO("State: FALLING -> STANDING (fall_score=%.2f)", fall_score);
        }
        break;

    case FALL_STATE_FALLEN:
        if (!is_fall)
        {
            recover_frame_count_++;
            if (recover_frame_count_ >= config_.recover_confirm_frames)
            {
                current_state_ = FALL_STATE_STANDING;
                recover_frame_count_ = 0;
                NN_LOG_INFO("State: FALLEN -> STANDING (fall_score=%.2f)", fall_score);
            }
            else
            {
                current_state_ = FALL_STATE_RECOVERING;
            }
        }
        else
        {
            recover_frame_count_ = 0;
        }
        break;

    case FALL_STATE_RECOVERING:
        if (!is_fall)
        {
            recover_frame_count_++;
            if (recover_frame_count_ >= config_.recover_confirm_frames)
            {
                current_state_ = FALL_STATE_STANDING;
                recover_frame_count_ = 0;
                NN_LOG_INFO("State: RECOVERING -> STANDING (fall_score=%.2f)", fall_score);
            }
        }
        else
        {
            // 恢复过程中又检测到跌倒
            recover_frame_count_ = 0;
            fall_frame_count_ = 1;
            current_state_ = FALL_STATE_FALLEN;
            NN_LOG_INFO("State: RECOVERING -> FALLEN (fall_score=%.2f)", fall_score);
        }
        break;

    default:
        current_state_ = FALL_STATE_STANDING;
        break;
    }

    return current_state_;
}

// 核心接口：输入关键点和检测框，输出跌倒结果
FallResult FallDetector::Update(const std::vector<Detection>& objects,
                                 const std::vector<std::map<int, KeyPoint>>& keypoints)
{
    FallResult result;
    result.timestamp = std::chrono::steady_clock::now();

    // 如果没有检测到人，保持当前状态
    if (objects.empty() || keypoints.empty())
    {
        result.state = current_state_;
        result.fall_score = last_fall_score_;
        result.state_str = fallStateToString(current_state_);
        result.is_alarm = (current_state_ == FALL_STATE_FALLEN);
        result.trunk_angle = 0.0f;
        result.bbox_ratio = 0.0f;
        result.head_foot_dist = 0.0f;
        result.center_of_gravity = 0.0f;
        result.shoulder_tilt = 0.0f;
        result.body_flatness = 0.0f;
        result.fall_frame_count = fall_frame_count_;
        result.recover_frame_count = recover_frame_count_;
        return result;
    }

    // 对每个检测到的人计算跌倒得分，取最高分
    float max_fall_score = 0.0f;
    float best_trunk_angle = 0.0f;
    float best_bbox_ratio = 0.0f;
    float best_head_foot_dist = 0.0f;
    float best_center_of_gravity = 0.0f;
    float best_shoulder_tilt = 0.0f;
    float best_body_flatness = 0.0f;

    for (size_t i = 0; i < objects.size() && i < keypoints.size(); i++)
    {
        // 提取各特征（用于结果展示）
        float trunk_angle = CalcTrunkAngle(keypoints[i]);
        float bbox_ratio = CalcBboxRatio(objects[i]);
        float head_foot_dist = CalcHeadFootDistance(keypoints[i]);
        float center_of_gravity = CalcCenterOfGravity(keypoints[i]);
        float shoulder_tilt = CalcShoulderTilt(keypoints[i]);
        float body_flatness = CalcBodyFlatness(keypoints[i]);

        // 计算跌倒得分
        float fall_score = CalcFallScore(objects[i], keypoints[i]);

        if (fall_score > max_fall_score)
        {
            max_fall_score = fall_score;
            best_trunk_angle = trunk_angle;
            best_bbox_ratio = bbox_ratio;
            best_head_foot_dist = head_foot_dist;
            best_center_of_gravity = center_of_gravity;
            best_shoulder_tilt = shoulder_tilt;
            best_body_flatness = body_flatness;
        }
    }

    // 更新状态机
    FallState new_state = UpdateStateMachine(max_fall_score);
    last_fall_score_ = max_fall_score;

    // 填充结果
    result.state = new_state;
    result.fall_score = max_fall_score;
    result.trunk_angle = best_trunk_angle;
    result.bbox_ratio = best_bbox_ratio;
    result.head_foot_dist = best_head_foot_dist;
    result.center_of_gravity = best_center_of_gravity;
    result.shoulder_tilt = best_shoulder_tilt;
    result.body_flatness = best_body_flatness;
    result.fall_frame_count = fall_frame_count_;
    result.recover_frame_count = recover_frame_count_;
    result.is_alarm = (new_state == FALL_STATE_FALLEN);
    result.state_str = fallStateToString(new_state);

    return result;
}
