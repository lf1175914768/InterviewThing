package com.study.spring;

import java.lang.reflect.Method;

import org.springframework.aop.AfterReturningAdvice;
import org.springframework.aop.MethodBeforeAdvice;

public class GreetingBeforeAndAfterAdvice implements MethodBeforeAdvice, AfterReturningAdvice {

	@Override
	public void afterReturning(Object returnValue, Method method, Object[] args, Object target) throws Throwable {
		System.out.println("after");
	}

	@Override
	public void before(Method method, Object[] args, Object target) throws Throwable {
		System.out.println("before");
	}

}
