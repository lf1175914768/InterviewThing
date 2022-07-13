package com.study.bugs;

import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * <p>description: LinkedBlockingQueue's bug  </p>
 * <p>className:  LinkedBlockingQueueBug </p>
 * <p>create time:  2022/6/13 15:51 </p>
 *
 * @author feng.liu
 * @since 1.0
 * @see java.util.concurrent.LinkedBlockingQueue
 * @version JDK version 1.8, repaired at JDK 1.9
 **/
public class LinkedBlockingQueueBug {

    public static void main(String[] args) throws InterruptedException {
        LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>(1000);
        for (int i = 0; i < 10; i++) {
            new Thread(() -> {
                while (true) {
                    queue.offer(new Object());
                    queue.remove();
                }
            }).start();
        }
        while (true) {
            System.out.println("begin scan, i still alive");
            queue.stream().anyMatch(Objects::isNull);
            Thread.sleep(100);
            System.out.println("finish scan, i still alive");
        }
    }
}
