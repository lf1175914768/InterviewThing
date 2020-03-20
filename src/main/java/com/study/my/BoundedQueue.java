package com.study.my;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @Description: TODO
 * @Author: Liufeng
 * @Date: 2020/3/18 16:29
 */
public class BoundedQueue<T> {
    private int count, addIndex, removeIndex;
    private Object[] items;
    private Lock lock = new ReentrantLock();
    private Condition notEmpty = lock.newCondition();
    private Condition notFull = lock.newCondition();

    public BoundedQueue(int size) {
        items = new Object[size];
    }

    public void add(T t) throws InterruptedException {
        lock.lock();
        try {
            while(count == items.length) {
                notFull.await();
            }
            items[addIndex] = t;
            if(++addIndex == items.length) {
                addIndex = 0;
            }
            ++count;
            notEmpty.signal();
        } finally {
            lock.unlock();
        }
    }

    public T remove() throws InterruptedException {
        lock.lock();
        try {
            while(count == 0) {
                notEmpty.await();
            }
            Object t = items[removeIndex];
            if(++removeIndex == items.length) {
                removeIndex = 0;
            }
            --count;
            notFull.signal();
            return (T) t;
        } finally {
            lock.unlock();
        }
    }
}
