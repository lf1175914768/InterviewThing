package com.study.interview;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import com.study.interview.tree.AhoCorasickAutomation;
import org.junit.Test;

import com.study.interview.array.CombinationSum;
import com.study.interview.array.MinSubArray;
import com.study.interview.array.MinimunPathSum;
import com.study.interview.array.SpiralMatrixSecond;
import com.study.interview.array.SumRange;
import com.study.interview.array.ThreeSumCloset;
import com.study.interview.chain.JudgeHasCircle;
import com.study.interview.chain.Node;
import com.study.interview.dynamicprogramming.MaxSumOfSubArray;
import com.study.interview.dynamicprogramming.NPersonSeatOtherPlace;

public class InterviewTest {
	
	@Test
	public void testJudgeHasCircle() {
		JudgeHasCircle judger = new JudgeHasCircle();
		Node root = generateChainCircle();
		int result = judger.getCircleLength(root);
		assertEquals(result, 6);
		root = generateChain();
		assertEquals(judger.getCircleLength(root), 0);
	}
	
	@Test
	public void testMaxSumOfSubArray() {
		int [] arr = { -2,11,-4,13,-5,-2 };
		int[] arr2 = {};
		MaxSumOfSubArray msos = new MaxSumOfSubArray();
		assertEquals(msos.maxOneDimensionSum(arr), 20);
		assertEquals(msos.maxOneDimensionSum(null), 0);
		assertEquals(msos.maxOneDimensionSum(arr2), 0);
		assertEquals(msos.maxOneDimensionSum_2(arr), 20);
		assertEquals(msos.maxOneDimensionSum_2(null), 0);
		assertEquals(msos.maxOneDimensionSum_2(arr2), 0);
	}

	@Test
	public void testACAutomation() {
		List<String> target = new ArrayList<>();
		target.add("abcdef");
		target.add("abhab");
		target.add("bcd");
		target.add("cde");
		target.add("cdfkcdf");

		String text = "bcabcdebcedfabcdefababkabhabk";

		AhoCorasickAutomation automation = new AhoCorasickAutomation(target);
		Map<String, List<Integer>> result = automation.find(text);

		System.out.println("'" + text + "', its length is " + text.length());
		for(Map.Entry entry : result.entrySet()) {
			System.out.println(entry.getKey() + ": " + entry.getValue());
		}
	}
	
	@Test
	public void testNPersonNSeat() {
		NPersonSeatOtherPlace np = new NPersonSeatOtherPlace();
		assertEquals(np.getNumbers(3), 2);
		assertEquals(np.getNumbers(4), 9);
		assertEquals(np.getNumbers(-1), 0);
		assertEquals(np.getNumbers(1), 0);
	}
	
	@Test
	public void test3SumCloset() {
		int[] arr = {-1, 2, 1, -4};
		ThreeSumCloset closet = new ThreeSumCloset();
		assertEquals(closet.threeSumCloset(arr, 1), 2);
	}
	
	@Test
	public void testSpiralMatrixSecond() {
		SpiralMatrixSecond sms = new SpiralMatrixSecond();
		int[][] result = sms.generateMatrix(5);
		for(int i = 0; i < result.length; i++) {
			System.out.println(Arrays.toString(result[i]));
		}
	}
	
	@Test
	public void testMinimunPathsum() {
		MinimunPathSum sum = new MinimunPathSum();
		int [][] arr = {{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
		assertEquals(sum.minPathSum(arr), 7);
	}
	
	@Test
	public void testPrintRange() {
		SumRange range = new SumRange();
		int[] param = {0,1,2,4,5,7};
		int[] pa2 = {};
		int[] pa3 = {1};
		int[] pa4 = {0,2,3,4,6,8,9};
		List<String> result = range.summaryRanges(param);
		for(String can : result) {
			System.out.println(can);
		}
		assertEquals(range.summaryRanges(pa2).size(), 0);
		assertEquals(range.summaryRanges(pa3).size(), 1);
		assertEquals(range.summaryRanges(pa3).get(0), "1");
		assertEquals(range.summaryRanges(pa4).size(), 4);
		for(String can : range.summaryRanges(pa4)) {
			System.out.println(can);
		}
		List<String> result2 = range.summaryRanges_eayUnderstand(param);
		for(String can : result2) {
			System.out.println(can);
		}
		assertEquals(range.summaryRanges_eayUnderstand(pa2).size(), 0);
		assertEquals(range.summaryRanges_eayUnderstand(pa3).size(), 1);
		assertEquals(range.summaryRanges_eayUnderstand(pa3).get(0), "1");
		assertEquals(range.summaryRanges_eayUnderstand(pa4).size(), 4);
		for(String can : range.summaryRanges_eayUnderstand(pa4)) {
			System.out.println(can);
		}
	}
	
	@Test
	public void testMinSubArr() {
		MinSubArray arr = new MinSubArray();
		int[] nums = {2,3,1,2,4,3};
		assertEquals(2, arr.minSubArrayLen(7, nums));
		assertEquals(2, arr.minSubArrayLen_2(7, nums));
	}
	
	@Test
	public void testCombinationSum() {
		CombinationSum sum = new CombinationSum();
		List<List<Integer>> result = sum.combinationSum3(3, 9);
		for(List<Integer> candidate : result) {
			System.out.println(Arrays.toString(candidate.toArray()));
		}
	}
	
	private Node generateChain() {
		Node root = new Node(1);Node n2 = new Node(2);Node n3 = new Node(3);
		Node n4 = new Node(4);Node n5 = new Node(5);Node n6 = new Node(6);
		Node n7 = new Node(7);Node n8 = new Node(8);Node n9 = new Node(9);
		root.setNext(n2); n2.setNext(n3); n3.setNext(n4); n4.setNext(n5);
		n5.setNext(n6); n6.setNext(n7); n7.setNext(n8); 
		n8.setNext(n9); 
		return root;
	}

	private Node generateChainCircle() {
		Node root = new Node(1);Node n2 = new Node(2);Node n3 = new Node(3);
		Node n4 = new Node(4);Node n5 = new Node(5);Node n6 = new Node(6);
		Node n7 = new Node(7);Node n8 = new Node(8);Node n9 = new Node(9);
		root.setNext(n2); n2.setNext(n3); n3.setNext(n4); n4.setNext(n5);
		n5.setNext(n6); n6.setNext(n7); n7.setNext(n8); 
		n8.setNext(n9); n9.setNext(n4);
		return root;
	}

}
