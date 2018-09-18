package com.study.my;

/**
 * @author Liufeng
 * @createData Created on: 2018年9月13日 下午3:58:49
 */
public class HelloB extends HelloA {

	public HelloB() {
		System.out.println("HelloB");
	}
	
	{System.out.println("I'm A HelloB.");}
	
	static {
		System.out.println("static HelloB.");
	}
	
	public static void main(String[] args) {
		new HelloB();
	}
}
