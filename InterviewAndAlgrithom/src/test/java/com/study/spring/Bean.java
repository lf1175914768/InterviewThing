package com.study.spring;

public class Bean {
	
	private String name;
	private int id;
	
	public Bean(int id, String name) {
		this.id = id;
		this.name = name;
	}
	
	@Override
	public String toString() {
		return "this.id : " + id + " this.name :" + name;
	}

}
