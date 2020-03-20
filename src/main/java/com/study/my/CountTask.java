package com.study.my;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.RecursiveTask;

/**
 * @Description: TODO
 * @Author: Liufeng
 * @Date: 2020/3/20 9:46
 */
public class CountTask extends RecursiveTask<Integer> {

    private static final int THRESHOLD = 2; //阈值
    private int start, end;

    public CountTask(int start, int end) {
        this.start = start;
        this.end = end;
    }

    public static void main(String[] args) {
        ForkJoinPool forkJoinPool = new ForkJoinPool();
        CountTask task = new CountTask(1, 10);
        Future<Integer> result = forkJoinPool.submit(task);
        try {
            System.out.println(result.get());
        } catch (InterruptedException e) {
        } catch (ExecutionException e) {
        }
    }

    @Override
    protected Integer compute() {
        int sum = 0;
        // if task is small enough, then compute it.
        boolean canCompute = (end - start) <= THRESHOLD;
        if(canCompute) {
            for(int i = start; i <= end; i++) {
                sum += i;
            }
        } else {
            // 如果任务大于阈值，就分裂成两个子任务进行计算
            int middle = (end + start) / 2;
            CountTask leftTask = new CountTask(start, middle);
            CountTask rightTask = new CountTask(middle + 1, end);
            leftTask.fork();
            rightTask.fork();
            // 等待字任务执行完，并得到结果
            int leftResult = leftTask.join();
            int rightResult = rightTask.join();
            // 合并子任务
            sum = leftResult + rightResult;
        }
        return sum;
    }
}
