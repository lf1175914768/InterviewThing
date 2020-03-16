package com.study.interview.other;

/**
 * 兼顾线程安全和效率的写法
 * @author LiuFeng
 *
 */
public class SingletonDoubleCheck {
	
	/**
	 * 这种写法被称为“双重检查法”， 就是在getInstance方法中， 进行两次检查， 看似多此一举，
	 * 实际上提升了并发度， 提升了性能。 因为在单例中new的情况非常少， 大多数时间都是可并行的读操作，
	 * 因此在进行加锁前，多检查null值，就可以避免不必要的加锁操作，执行效率也就提高了。
	 */
	
	/**
	 * volatile关键字有两层含义。第一层：可见性；第二层：禁止指令重排序优化。
	 * 可见性指的是在一个线程中对该变量的修改会马上由工作内存写回主内存，所以会马上反应在其他线程的读取操作中，
	 * 工作内存和朱内存可以理解为实际电脑中的高速缓存和主存，工作内存是线程独享的，而主存是线程共享的。
	 * volatile的第二层语义是禁止指令重排序优化。犹豫编译器优化，在实际执行的时候可能与我们编写的顺序不同， 
	 * 编译器只保证程序执行结果与源代码相同， 却不保证执行顺序与源代码相同。 这在单线程程序中没有什么影响。
	 * 然而一旦引入多线程， 这种程序就可能导致严重问题。volatile就能解决这种问题
	 */
	private static volatile SingletonDoubleCheck singleton = null;
	
	private SingletonDoubleCheck() {}
	
	public static SingletonDoubleCheck getInstance() {
		if(singleton == null) {
			synchronized(SingletonDoubleCheck.class) {
				if(singleton == null) {
					singleton = new SingletonDoubleCheck();
				}
			}
		} 
		return singleton;
	}

}
