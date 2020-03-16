package com.study.interview.other;

/**
 * 用静态内部类来实现单例模式
 * @author LiuFeng
 *
 */
public class SingletonOfInnerClass {
	
	//由于静态内部类只会被加载一次，所以这种写法也是线程安全的
	
	private SingletonOfInnerClass() {}
	
	private static class Holder {
		private static SingletonOfInnerClass instance = new SingletonOfInnerClass();
	}
	
	public static SingletonOfInnerClass getInstance() {
		return Holder.instance;
	}

}
