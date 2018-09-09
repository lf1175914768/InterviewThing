package com.stud.algorithm;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import com.study.algorithm.HeapSort;
import com.study.algorithm.QuickSort;
import com.study.interview.dynamicprogramming.MaxCommonSequence;
import com.study.interview.dynamicprogramming.MaxIncrementalSubArray;
import com.study.interview.string.KMPAlgorithm;
import com.study.interview.string.ReplaceString;
import com.study.interview.tree.BinaryTreeSerialize;
import com.study.interview.tree.ExchangeTwoErrorNodes;
import com.study.interview.tree.Node;
import com.study.interview.tree.TreePrinter;

public class ApplicationTest {
	
	Node n;
	
	@Before
	public void before() {
		Node n1 = new Node(15);
		Node n2 = new Node(26, new Node(20), new Node(24));
		Node n3 = new Node(17, n1, n2);
		n = new Node(25, n3, new Node(27, new Node(23), new Node(40)));
	}
	
	@Test
	public void testBinaryTreeSerial() {
		BinaryTreeSerialize bts = new BinaryTreeSerialize();
		String serial = null;
		System.out.println(bts.preOrderSerialize_1(n));
		System.out.println(bts.preOrderSerialize_2(n));
		System.out.println(serial = bts.preOrderSerialize_3(n));
		Node root = bts.preOrderDeserialize_1(serial);
		assertEquals(bts.preOrderSerialize_1(root), serial);
	}

	@Test
	public void test() {
		QuickSort sort = new QuickSort();
		int[] test = {1,2,6,9,8,3,6,5,7,8,9,4,5,6,5,6,8,9,7,4,5,6,12,456,78,9};
		sort.sort(test, 0, test.length - 1);
		for(int i = 0; i < test.length; i++) {
			System.out.print(test[i] + " ");
		}
	}
	
	@Test
	public void testHeapSort() {
		HeapSort sort = new HeapSort();
		char[] test = {'a','d', 'z','y','w','v','q','o','r','p','x','h'};
		sort.heapSort_2(test);
		for(int i = 0; i < test.length; i++) {
			System.out.print(test[i] + " ");
		}
	}
	
	@Test
	public void testKmp() {
		KMPAlgorithm kmp = new KMPAlgorithm();
		int test = kmp.getIndexOf("LiufengCando", "ngC");
		System.out.println(test);
	}
	
	@Test
	public void testIncrementSubArray() {
		MaxIncrementalSubArray arr = new MaxIncrementalSubArray();
		int[] test = {0,2,3,6,5,9,6,8,3,2,5,4,7,8,4};
//		int[] result = arr.list_1(test);
		int[] result = arr.list_2(test);
		for(int i = 0; i < result.length; i++) {	
			System.out.print(result[i] + " ");
		}
	}
	
	@Test
	public void testReplaceString() {
		ReplaceString r1 = new ReplaceString();
		System.out.println(r1.replace_1("123abc", "abc", "4567"));
		System.out.println(r1.replace_1("123", "abc", "4567"));
		System.out.println(r1.replace_1("123abcabcabcabc213", "abc", "4567"));
		System.out.println(r1.replace_2("123abc", "abc", "4567"));
		System.out.println(r1.replace_2("123", "abc", "4567"));
		System.out.println(r1.replace_2("123abcabcabcabc213", "abc", "4567"));
	}
	
	@Test
	public void testPrinter() {
		TreePrinter print = new TreePrinter();
		print.printByLevel_2(n);
		System.out.println();
		print.printByZigZag(n);
		System.out.println();
		print.printByLevel_1(n);
	}
	
	@Test
	public void testTwoErrors() {
		ExchangeTwoErrorNodes nodes = new ExchangeTwoErrorNodes();
		Node[] result = nodes.getTwoErrNodes(n);
		System.out.println(result[0]);
		System.out.println(result[1]);
	}
	
	@Test
	public void testMaxCommonSequence() {
		MaxCommonSequence test = new MaxCommonSequence();
		String result = test.lcse("1a2c3d4b56", "b1d23ca45b6a");
		System.out.println(result);
	}
	
}
