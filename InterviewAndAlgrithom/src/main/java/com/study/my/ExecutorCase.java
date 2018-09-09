package com.study.my;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * @author Liufeng
 * Created on 2018年9月9日 下午6:51:54
 */
public class ExecutorCase {
	
	private static Executor executor = Executors.newFixedThreadPool(10);
	
	public static void main(String[] args) {
		for(int i = 0; i < 20; i++) {
			executor.execute(new Task());
		}
	}
	
	static class Task implements Runnable {
		@Override
		public void run() {
			System.out.println(Thread.currentThread().getName());
		}
	}

}
