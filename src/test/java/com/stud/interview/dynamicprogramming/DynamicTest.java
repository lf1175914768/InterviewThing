package com.stud.interview.dynamicprogramming;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import com.study.interview.dynamicprogramming.BestTimeToBuyAndSell;
import com.study.interview.dynamicprogramming.Triangle;

/**
 * @author Liufeng
 * Created on 2018年9月8日 下午11:09:53
 */
public class DynamicTest {
	
	@Test
	public void testTriangle() {
		Triangle tri = new Triangle();
		List<List<Integer>> list = new ArrayList<List<Integer>>();
		List<Integer> l1 = new ArrayList<Integer>(); l1.add(2);
		List<Integer> l2 = new ArrayList<Integer>(); l2.add(3);  l2.add(4);
		List<Integer> l3 = new ArrayList<Integer>(); l3.add(6);  l3.add(5); l3.add(7);
		List<Integer> l4 = new ArrayList<Integer>(); l4.add(4);  l4.add(1); l4.add(8); l4.add(3);
		list.add(l1); list.add(l2); list.add(l3);  list.add(l4);
		System.out.println(tri.minimumTotal(list));
	}
	
	@Test
	public void testBTTBAS() {
		BestTimeToBuyAndSell sell = new BestTimeToBuyAndSell();
		int[] param = {3,3,5,0,0,3,1,4};
		System.out.println(sell.maxProfit_1(param));
		System.out.println(sell.maxProfit_2(param));
	}

}
