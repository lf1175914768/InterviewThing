package com.study.my;

/**
 * @Description: TODO
 * @Author: Liufeng
 * @Date: 2020/3/17 11:13
 */
public interface ThreadPool<Job extends Runnable> {
    void execute(Job job);
    void shutdown();
    void addWorkers(int num);
    void removeWorker(int num);
    int getJobSize();
}
