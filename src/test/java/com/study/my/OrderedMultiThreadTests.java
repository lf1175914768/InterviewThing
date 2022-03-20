package com.study.my;

import org.junit.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class OrderedMultiThreadTests {

    @Test
    public void testOrderedPrint() throws InterruptedException {
        Object lock = new Object();
        new Thread(new OrderedMultiThread.Task("A", lock)).start();
        Thread.sleep(10L);
        new Thread(new OrderedMultiThread.Task("B", lock)).start();
        Thread.sleep(10L);
        new Thread(new OrderedMultiThread.Task("C", lock)).start();
    }

    @Test
    public void testOrderPrint2() throws InterruptedException {
        Semaphore a = new Semaphore(1);
        Semaphore b = new Semaphore(0);
        Semaphore c = new Semaphore(0);

        ExecutorService pool = Executors.newFixedThreadPool(3);
        int count = 10;
        pool.execute(new OrderedMultiThread.WorkerSemaphore("A", a, b, count));
        pool.execute(new OrderedMultiThread.WorkerSemaphore("B", b, c, count));
        pool.execute(new OrderedMultiThread.WorkerSemaphore("C", c, a, count));
        Thread.sleep(1000);
        pool.shutdown();
    }

    @Test
    public void testOrderedPrint3() throws InterruptedException {
        ReentrantLock lock = new ReentrantLock();
        Condition conditionA = lock.newCondition();
        Condition conditionB = lock.newCondition();
        Condition conditionC = lock.newCondition();
        ExecutorService pool = Executors.newFixedThreadPool(3);
        int count = 10;
        pool.execute(new OrderedMultiThread.WorkerLock("A", count, lock, conditionA, conditionB, 1));
        pool.execute(new OrderedMultiThread.WorkerLock("B", count, lock, conditionB, conditionC, 2));
        pool.execute(new OrderedMultiThread.WorkerLock("C", count, lock, conditionC, conditionA, 3));
        Thread.sleep(1000L);
        Thread.currentThread().join();
        pool.shutdown();
    }
}
