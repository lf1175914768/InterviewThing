package com.study.my;

import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class OrderedMultiThread {

    // -----------按顺序执行多次--------------

    /*
     * 注意点：
     * 多个线程共用一把锁，才能完成同步
     * 需要额外的共享变量，因为你执行完成后，发出的通知是通过所有等待的线程，所以没有办法准确
     * 指出通知的谁，所有等待的线程都醒来后，在进行判断
     * 需要使用 while (!judge(name, num)) 进行判断
     */

    static volatile int num = 1;

    // 并不建议使用这种方式，只是做一个参考示例
    static class Task implements Runnable {
        String name;
        Object lock;

        Task(String name, Object lock) {
            this.name = name;
            this.lock = lock;
        }

        @Override
        public void run() {
            for (int i = 0; i < 10; i++) {
                printNum();
            }
        }

        private void printNum() {
            synchronized (lock) {
                while (!judge(name, num)) {
                    try {
                        lock.wait();
                    } catch (InterruptedException e) {

                    }
                }
                num++;
                System.out.println(name);
                lock.notifyAll();
            }
        }

        private boolean judge(String name, int num) {
            if ("A".equals(name) && num % 3 == 1) {
                return true;
            } else if ("B".equals(name) && num % 3 == 2) {
                return true;
            } else {
                return "C".equals(name) && num % 3 == 0;
            }
        }
    }

    // -------另一种方式---------

    static class WorkerSemaphore implements Runnable {
        private final String key;
        private final Semaphore current;
        private final Semaphore next;
        private final int count;

        public WorkerSemaphore(String key, Semaphore current, Semaphore next, int count) {
            this.key = key;
            this.current = current;
            this.next = next;
            this.count = count;
        }

        @Override
        public void run() {
            for (int i = 0; i < count; i++) {
                try {
                    // 获取当前的锁
                    current.acquire();
                    System.out.println(i + ", " + key);
                    next.release();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private static volatile int state = 1;

    static class WorkerLock implements Runnable {
        private final String key;
        private final int count;
        private final Lock lock;
        private final Condition current;
        private final Condition next;
        private final int targetState;

        public WorkerLock(String key, int count, Lock lock, Condition current, Condition next, int targetState) {
            this.key = key;
            this.count = count;
            this.lock = lock;
            this.current = current;
            this.next = next;
            this.targetState = targetState;
        }

        @Override
        public void run() {
            lock.lock();
            try {
                for (int i = 0; i < count; i++) {
                    while (state != targetState) {
                        current.await();
                    }
                    System.out.println(i + ", " + key);
                    state++;
                    if (state > 3) {
                        state = 1;
                    }
                    next.signal();
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                lock.unlock();
            }
        }
    }
}
