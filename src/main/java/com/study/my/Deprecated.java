package com.study.my;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.TimeUnit;

/**
 * @Description: TODO
 * @Author: Liufeng
 * @Date: 2020/3/16 21:37
 */
public class Deprecated {

    public static void main(String[] args) {
        DateFormat format = new SimpleDateFormat("HH:mm:ss");
        Thread printThread = new Thread(new Runner(), "PrintThread");
        printThread.setDaemon(true);
        printThread.start();
        try {
            TimeUnit.SECONDS.sleep(3);
            printThread.suspend();
            System.out.println("main thread suspend PrintThread at " + format.format(new Date()));
            TimeUnit.SECONDS.sleep(3);
            printThread.resume();
            System.out.println("main thread resume PrintThread at " + format.format(new Date()));
            TimeUnit.SECONDS.sleep(3);
            printThread.stop();
            System.out.println("main thread stop PrintThread at " + format.format(new Date()));
            TimeUnit.SECONDS.sleep(2);
        } catch (InterruptedException e) {

        }

    }

    static class Runner implements Runnable {
        private volatile boolean on = true;

        @Override
        public void run() {
           while(on && !Thread.currentThread().isInterrupted()) {
               // to do something
           }
        }

        public void cancel() {
            on = false;
        }
    }
}
