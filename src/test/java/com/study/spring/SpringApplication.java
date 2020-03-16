package com.study.spring;

import org.junit.Test;
import org.springframework.aop.framework.ProxyFactory;

public class SpringApplication {

	@Test
	public void test_1() {
		ProxyFactory factory = new ProxyFactory();   //创建代理工厂
		factory.setTarget(new GreetingImpl());    // 摄入目标类对象
		factory.addAdvice(new GreetingBeforeAndAfterAdvice());    //添加前置增强， 后置增强
		
		Greeting greeting = (Greeting) factory.getProxy(); // 从袋里工厂中获取代理，调用代理的方法
		greeting.sayHello("liufeng");
	}
	
	@Test
	public void testAroundAdvice() {
		ProxyFactory factory = new ProxyFactory();
		factory.setTarget(new GreetingImpl());
		factory.addAdvice(new GreetingAroundAdvice());
		
		Greeting result = (Greeting) factory.getProxy();
		result.sayHello("liufeng");
	}

}
