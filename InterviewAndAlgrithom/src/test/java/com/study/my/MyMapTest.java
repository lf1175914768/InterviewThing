package com.study.my;

import static org.junit.Assert.*;

import org.junit.Test;

public class MyMapTest {

	@Test
	public void testForFirst() {
		
		MyMap<String, String> test = new MyHashMap<String, String>();
		for(int i = 0; i < 10000; i++) {
			test.put("key" + i, "value" + i);
		}
		long mill = System.currentTimeMillis();
		for(int i = 0; i < 10000; i++) {
			System.out.println(test.get("key" + i));
		}
		System.out.println(System.currentTimeMillis() - mill);
	}
	
	@Test
	public void testForNullValue() {
		MyMap<String, String> test = new MyHashMap<String, String>();
		test.put(null, "abc");
		test.put("key1", "value1");
		assertEquals("value1", test.get("key1"));
		assertEquals("abc", test.get(null));
		test.put(null, "test");
		assertEquals("test", test.get(null));
	}

}
