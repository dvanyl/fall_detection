
#include "yolov8_thread_pool.h"
#include "draw/cv_draw.h"
#include "rknn_api.h"

// 构造函数
Yolov8ThreadPool::Yolov8ThreadPool() { stop = false; }

// 析构函数
Yolov8ThreadPool::~Yolov8ThreadPool()
{
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

// 初始化：加载模型，创建线程
// 默认3线程，对应RK3588的3个NPU核心
nn_error_e Yolov8ThreadPool::setUp(std::string &model_path, int num_threads)
{
    NN_LOG_INFO("Yolov8ThreadPool: setting up with %d threads", num_threads);

    // NPU核心掩码表：RK3588有3个NPU核心，每个线程绑定一个独立核心
    static const int npu_core_masks[] = {
        RKNN_NPU_CORE_0,   // 线程0 → NPU核心0 (值=1)
        RKNN_NPU_CORE_1,   // 线程1 → NPU核心1 (值=2)
        RKNN_NPU_CORE_2,   // 线程2 → NPU核心2 (值=4)
    };

    // 为每个线程创建独立的模型实例（每个实例绑定独立的RKNN context）
    for (size_t i = 0; i < num_threads; ++i)
    {
        nn_model_type_e model_type = NN_YOLOV8_POSE;
        std::shared_ptr<Yolov8Custom> Yolov8 = std::make_shared<Yolov8Custom>(model_type);
        auto ret = Yolov8->LoadModel(model_path.c_str());
        if (ret != NN_SUCCESS)
        {
            NN_LOG_ERROR("Yolov8ThreadPool: failed to load model for thread %zu", i);
            return ret;
        }

        // 绑定NPU核心：将每个模型实例绑定到独立的NPU核心，实现真正的三核并行推理
        if (i < sizeof(npu_core_masks) / sizeof(npu_core_masks[0]))
        {
            ret = Yolov8->SetCoreMask(npu_core_masks[i]);
            if (ret != NN_SUCCESS)
            {
                NN_LOG_WARNING("Yolov8ThreadPool: failed to set core mask for thread %zu, using AUTO", i);
            }
        }

        Yolov8_instances.push_back(Yolov8);
        NN_LOG_INFO("Yolov8ThreadPool: thread %zu model loaded (NPU core %zu)", i, i);
    }

    // 创建工作线程
    for (size_t i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(&Yolov8ThreadPool::worker, this, i);
    }

    NN_LOG_INFO("Yolov8ThreadPool: %d worker threads started", num_threads);
    return NN_SUCCESS;
}

// 工作线程函数
void Yolov8ThreadPool::worker(int id)
{
    while (!stop)
    {
        std::pair<int, cv::Mat> task;
        std::shared_ptr<Yolov8Custom> instance = Yolov8_instances[id];
        {
            // 等待任务
            std::unique_lock<std::mutex> lock(mtx1);
            cv_task.wait(lock, [&] { return !tasks.empty() || stop; });

            if (stop) return;

            task = tasks.front();
            tasks.pop();
        }

        // 运行模型推理（包含预处理 + NPU推理 + 后处理）
        std::vector<Detection> detections;
        std::vector<std::map<int, KeyPoint>> kps;
        instance->Run(task.second, detections, kps);

        {
            // 保存结果（检测框 + 关键点）
            std::lock_guard<std::mutex> lock(mtx2);
            results.insert({task.first, detections});
            kp_results.insert({task.first, kps});
            img_results.insert({task.first, task.second.clone()});
            cv_result.notify_one();
        }
    }
}

// 提交任务
nn_error_e Yolov8ThreadPool::submitTask(const cv::Mat &img, int id)
{
    // 限制任务队列大小，避免内存溢出
    while (tasks.size() > 5)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img.clone()});
    }
    cv_task.notify_one();
    return NN_SUCCESS;
}

// 获取检测框结果
nn_error_e Yolov8ThreadPool::getTargetResult(std::vector<Detection> &objects, int id)
{
    while (results.find(id) == results.end())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::lock_guard<std::mutex> lock(mtx2);
    objects = results[id];
    results.erase(id);
    kp_results.erase(id);
    img_results.erase(id);
    return NN_SUCCESS;
}

// 获取完整结果（检测框 + 关键点）
nn_error_e Yolov8ThreadPool::getTargetResultFull(
    std::vector<Detection> &objects,
    std::vector<std::map<int, KeyPoint>> &keypoints,
    int id)
{
    int loop_cnt = 0;
    while (results.find(id) == results.end())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        loop_cnt++;
        if (loop_cnt > 2500)  // 5秒超时
        {
            NN_LOG_ERROR("getTargetResultFull timeout for frame %d", id);
            return NN_TIMEOUT;
        }
    }
    std::lock_guard<std::mutex> lock(mtx2);
    objects = results[id];
    keypoints = kp_results[id];
    results.erase(id);
    kp_results.erase(id);
    // 注意：不在这里删除 img_results，图片由 getTargetImgResult 独立获取和删除
    return NN_SUCCESS;
}

// 获取图片结果
nn_error_e Yolov8ThreadPool::getTargetImgResult(cv::Mat &img, int id)
{
    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        loop_cnt++;
        if (loop_cnt > 1000)
        {
            NN_LOG_ERROR("getTargetImgResult timeout");
            return NN_TIMEOUT;
        }
    }
    std::lock_guard<std::mutex> lock(mtx2);
    img = img_results[id];
    img_results.erase(id);
    results.erase(id);
    kp_results.erase(id);
    return NN_SUCCESS;
}

// 获取最新已完成帧ID（用于帧跳过）
int Yolov8ThreadPool::getLatestResultId() const
{
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mtx2));
    if (results.empty())
        return -1;
    return results.rbegin()->first;  // map按key排序，最后一个就是最大的
}

// 清理指定ID之前的所有结果（防止帧跳过时内存泄漏）
void Yolov8ThreadPool::cleanResultsUpTo(int id)
{
    std::lock_guard<std::mutex> lock(mtx2);
    // 清理 results 中 id 之前的条目
    for (auto it = results.begin(); it != results.end(); )
    {
        if (it->first < id)
            it = results.erase(it);
        else
            ++it;
    }
    // 清理 kp_results 中 id 之前的条目
    for (auto it = kp_results.begin(); it != kp_results.end(); )
    {
        if (it->first < id)
            it = kp_results.erase(it);
        else
            ++it;
    }
    // 清理 img_results 中 id 之前的条目
    for (auto it = img_results.begin(); it != img_results.end(); )
    {
        if (it->first < id)
            it = img_results.erase(it);
        else
            ++it;
    }
}

// 获取待处理任务数
int Yolov8ThreadPool::getTaskQueueSize() const
{
    // 注意：这里不加锁是因为只是用于统计监控
    return tasks.size();
}

// 停止所有线程
void Yolov8ThreadPool::stopAll()
{
    stop = true;
    cv_task.notify_all();
}
