package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * <p>description:  top 面试题 test </p>
 * <p>className:  TopInterviewProblemTests </p>
 * <p>create time:  2022/3/9 14:28 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class TopInterviewProblemTests {

    private TopInterviewProblems problems;

    @Before
    public void init() {
        problems = new TopInterviewProblems();
    }

    @Test
    public void testMyAtoi() {
        assertEquals(problems.myAtoi("42"), 42);
        assertEquals(problems.myAtoi("   -42"), -42);
        assertEquals(problems.myAtoi("4193 with words"), 4193);
        assertEquals(problems.myAtoi("-0000039"), -39);
    }

    @Test
    public void testCoinChange() {
        int[] pa = new int[] {1, 2, 5, 2};
        assertEquals(problems.coinChange(pa, 11), 3);
        assertEquals(problems.coinChange_v2(pa, 11), 3);
        pa = new int[] {2};
        assertEquals(problems.coinChange(pa, 3), -1);
        assertEquals(problems.coinChange_v2(pa, 3), -1);
        assertEquals(problems.coinChange(pa, 0), 0);
        assertEquals(problems.coinChange_v2(pa, 0), 0);
    }

    @Test
    public void testLargestNumber() {
        int[] param = new int[] {10,2};
        assertEquals(problems.largestNumber(param), "210");
        param = new int[] {3,30,34,5,9};
        assertEquals(problems.largestNumber(param), "9534330");
        param = new int[] {0, 0, 0};
        assertEquals(problems.largestNumber(param), "0");
    }

    @Test
    public void testSortList() {
        ListNode node3 = new ListNode(3);
        ListNode node1 = new ListNode(1, node3);
        ListNode node2 = new ListNode(2, node1);
        ListNode node4 = new ListNode(4, node2);
        ListNode head = problems.sortList(node4);
        int[] arr = new int[4], res = new int[] {1,2,3,4};
        int i = 0;
        while (head != null) {
            arr[i++] = head.val;
            head = head.next;
        }
        assertArrayEquals(arr, res);
    }

    @Test
    public void testFindPeakElement() {
        int[] param = new int[] {1,2,3,1};
        assertEquals(problems.findPeakElement(param), 2);
        param = new int[] {1,2,1,3,5,6,4};
        assertEquals(problems.findPeakElement(param), 5);
        param = new int[] {1,2,3,4,5,6};
        assertEquals(problems.findPeakElement(param), 5);
    }
}
