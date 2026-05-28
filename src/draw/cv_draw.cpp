
#include "cv_draw.h"

#include "utils/logging.h"

void DrawDetections(cv::Mat& img, const std::vector<Detection>& objects) {
    NN_LOG_DEBUG("draw %ld objects", objects.size());
    for (const auto& object : objects) {
        cv::rectangle(img, object.box, object.color, 2);
        // class name with confidence
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", object.className.c_str(), object.confidence);
        // 绘制标签背景
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        cv::rectangle(img,
                      cv::Point(object.box.x, object.box.y - textSize.height - 4),
                      cv::Point(object.box.x + textSize.width, object.box.y),
                      object.color, -1);
        cv::putText(img, label, cv::Point(object.box.x, object.box.y - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    }
}

void DrawMask(cv::Mat& img, cv::Mat& seg_mask) {
    cv::addWeighted(img, 0.55, seg_mask, 0.45, 0, img);
}

void DrawCocoKps(cv::Mat& img, const std::vector<std::map<int, KeyPoint>>& keypoints) {
    // 关键点颜色（按身体部位分组）
    // 头部: 红色, 上半身: 蓝色, 下半身: 绿色
    static const cv::Scalar kp_colors[17] = {
        cv::Scalar(0, 0, 255),     // 0: 鼻子 - 红
        cv::Scalar(0, 0, 255),     // 1: 左眼
        cv::Scalar(0, 0, 255),     // 2: 右眼
        cv::Scalar(0, 0, 255),     // 3: 左耳
        cv::Scalar(0, 0, 255),     // 4: 右耳
        cv::Scalar(255, 128, 0),   // 5: 左肩 - 蓝
        cv::Scalar(255, 128, 0),   // 6: 右肩
        cv::Scalar(255, 128, 0),   // 7: 左肘
        cv::Scalar(255, 128, 0),   // 8: 右肘
        cv::Scalar(255, 128, 0),   // 9: 左腕
        cv::Scalar(255, 128, 0),   // 10: 右腕
        cv::Scalar(0, 255, 0),     // 11: 左髋 - 绿
        cv::Scalar(0, 255, 0),     // 12: 右髋
        cv::Scalar(0, 255, 0),     // 13: 左膝
        cv::Scalar(0, 255, 0),     // 14: 右膝
        cv::Scalar(0, 255, 0),     // 15: 左踝
        cv::Scalar(0, 255, 0),     // 16: 右踝
    };

    // 绘制关键点（小圆点，半径3）
    for (const auto& keypoint : keypoints) {
        for (const auto& keypoint_item : keypoint) {
            int idx = keypoint_item.first;
            cv::Scalar color = (idx < 17) ? kp_colors[idx] : cv::Scalar(0, 255, 0);
            cv::circle(img, cv::Point(keypoint_item.second.x, keypoint_item.second.y),
                       3, color, -1);
        }
    }

    // 绘制骨架（线条更细）
    // skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    //            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
    //            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    static const std::vector<std::vector<int>> joint_pairs =
                {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
                {6, 7},  {6, 8},  {7, 9},  {8, 10}, {9, 11},
                {2, 3},  {1, 2}, {1, 3},  {2, 4},  {3, 5},  {4, 6},  {5, 7}};

    // 骨架颜色按部位
    static const cv::Scalar sk_colors[] = {
        cv::Scalar(0, 255, 0),     // 腿部 (绿色)
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 128, 0),   // 躯干 (蓝色)
        cv::Scalar(255, 128, 0),
        cv::Scalar(255, 128, 0),   // 手臂
        cv::Scalar(255, 128, 0),
        cv::Scalar(255, 128, 0),
        cv::Scalar(255, 128, 0),
        cv::Scalar(255, 128, 0),
        cv::Scalar(0, 0, 255),     // 头部 (红色)
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 0, 255),
    };

    for (const auto& keypoint : keypoints) {
        for (size_t i = 0; i < joint_pairs.size(); i++) {
            const auto& joint_pair = joint_pairs[i];
            const auto& joint1 = keypoint.find(joint_pair[0] - 1);
            const auto& joint2 = keypoint.find(joint_pair[1] - 1);
            if (joint1 != keypoint.end() && joint2 != keypoint.end()) {
                cv::Scalar color = (i < 19) ? sk_colors[i] : cv::Scalar(0, 255, 255);
                cv::line(img, cv::Point(joint1->second.x, joint1->second.y),
                         cv::Point(joint2->second.x, joint2->second.y),
                         color, 1);
            }
        }
    }
}
