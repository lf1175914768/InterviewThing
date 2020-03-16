package com.study.jvm.heap;

import java.util.ArrayList;
import java.util.List;

/**
 * VM Args: -Xms20M -Xmx20M -XX:+HeapDumpOnOutofMemeryError
 * @author LiuFeng
 *
 */
public class HeapOOM {
	
	static class OOMObject {
		
	}
	
	public static void main(String[] args) {
		List<OOMObject> list = new ArrayList<OOMObject>();
		while(true) {
			list.add(new OOMObject());
		}
	}

}
