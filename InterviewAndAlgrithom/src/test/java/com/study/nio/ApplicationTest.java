package com.study.nio;

import org.junit.Test;

public class ApplicationTest {

	@Test
	public void test() {
		FirstChannel channel = new FirstChannel();
		channel.readAndWrite();
	}

}
