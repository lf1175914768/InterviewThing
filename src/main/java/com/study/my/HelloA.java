package com.study.my;

/**
 * @author Liufeng
 * @createData Created on: 2018年9月13日 下午3:57:05
 */
public class HelloA {
	
	public HelloA() {
		System.out.println("HelloA");
	}
	
	{System.out.println("I'm A Class");}

	static {
		System.out.println("Static A.");
	}
}
