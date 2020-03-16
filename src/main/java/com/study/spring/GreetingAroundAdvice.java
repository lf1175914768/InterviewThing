package com.study.spring;

import org.aopalliance.intercept.MethodInterceptor;
import org.aopalliance.intercept.MethodInvocation;

public class GreetingAroundAdvice implements MethodInterceptor {

	@Override
	public Object invoke(MethodInvocation invocation) throws Throwable {
		before();
		Object result = invocation.proceed();
		after();
		return result;
	}

	private void after() {
		System.out.println("after");
	}

	private void before() {
		System.out.println("before");
	}
	
	

}
