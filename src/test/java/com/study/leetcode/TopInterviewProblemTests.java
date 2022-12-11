package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.*;

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

    @Test
    public void testPartition() {
        List<List<String>> rs = problems.partition("aab");
        List<List<String>> rs2 = problems.partition_v2("aab");
        assertEquals(rs.size(), 2);
        assertEquals(rs2.size(), 2);
        String[] tmp1 = new String[] {"a","a","b"};
        String[] tmp2 = new String[] {"aa","b"};
        assertArrayEquals(rs.get(0).toArray(new String[0]), tmp1);
        assertArrayEquals(rs2.get(0).toArray(new String[0]), tmp1);
        assertArrayEquals(rs.get(1).toArray(new String[0]), tmp2);
        assertArrayEquals(rs2.get(1).toArray(new String[0]), tmp2);
        assertEquals(problems.partition("a").size(), 1);
        assertEquals(problems.partition_v2("a").size(), 1);
    }

    @Test
    public void testCopyRandomList() {
        TopInterviewProblems.RandomNode node5 = new TopInterviewProblems.RandomNode(1);
        TopInterviewProblems.RandomNode node4 = new TopInterviewProblems.RandomNode(10, node5);
        TopInterviewProblems.RandomNode node3 = new TopInterviewProblems.RandomNode(11, node4, node5);
        TopInterviewProblems.RandomNode node2 = new TopInterviewProblems.RandomNode(13, node3);
        TopInterviewProblems.RandomNode node1 = new TopInterviewProblems.RandomNode(7, node2);
        node2.random = node1;
        node4.random = node3;
        node5.random = node1;

        TopInterviewProblems.RandomNode node = problems.copyRandomList(node1);
        assertEquals(node.val, 7);
        assertEquals(node.next.val, 13);
        assertEquals(node.next.next.val, 11);
        assertEquals(node.next.next.next.val, 10);
        assertEquals(node.next.next.next.next.val, 1);

        TopInterviewProblems.RandomNode newNode = problems.copyRandomList_v2(node1);
        assertEquals(newNode.val, 7);
        assertEquals(newNode.next.val, 13);
        assertEquals(newNode.next.next.val, 11);
        assertEquals(newNode.next.next.next.val, 10);
        assertEquals(newNode.next.next.next.next.val, 1);
    }

    @Test
    public void testRotate() {
        int[] param = new int[] {1,2,3,4,5,6,7};
        problems.rotate(param, 3);
        int[] result = new int[] {5,6,7,1,2,3,4};
        assertArrayEquals(param, result);

        param = new int[] {-1,-100,3,99};
        result = new int[] {3,99,-1,-100};
        problems.rotate(param, 6);
        assertArrayEquals(param, result);
    }

    @Test
    public void testOddEvenList() {
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2, node1);
        ListNode node3 = new ListNode(3, node2);
        ListNode node4 = new ListNode(4, node3);
        ListNode node5 = new ListNode(5, node4);
        ListNode head = problems.oddEvenList(node5);
        assertEquals(head.val, 5);
        assertEquals(head.next.val, 3);
        assertEquals(head.next.next.val, 1);
        assertEquals(head.next.next.next.val, 4);
        assertEquals(head.next.next.next.next.val, 2);
        node1 = new ListNode(1);
        head = problems.oddEvenList(node1);
        assertEquals(head.val, 1);
        node2 = new ListNode(2, node1);
        head = problems.oddEvenList(node2);
        assertEquals(head.val, 2);
        assertEquals(head.next.val, 1);
        node3 = new ListNode(3, node2);
        head = problems.oddEvenList(node3);
        assertEquals(head.val, 3);
        assertEquals(head.next.val, 1);
        assertEquals(head.next.next.val, 2);

        head = ListProblemTests.buildCommonNode(Arrays.asList(1, 2, 3, 4, 5));
        ListNode node = problems.oddEvenList_v2(head);
        int[] rs = new int[] {1,3,5,2,4};
        assertArrayEquals(rs, ListProblemTests.toArray(node));
        head = ListProblemTests.buildCommonNode(Arrays.asList(2,1,3,5,6,4,7));
        rs = new int[] {2,3,6,7,1,5,4};
        assertArrayEquals(ListProblemTests.toArray(problems.oddEvenList_v2(head)), rs);
    }

    @Test
    public void testIncreasingTriplet() {
        int[] param = new int[] {1,2,3,4,5};
        assertTrue(problems.increasingTriplet(param));
        param = new int[] {5,4,3,2,1};
        assertFalse(problems.increasingTriplet(param));
        param = new int[] {2,1,5,0,4,6};
        assertTrue(problems.increasingTriplet(param));
    }

    @Test
    public void testGetSum() {
        assertEquals(problems.getSum(4, 5), 9);
        assertEquals(problems.getSum(4, 2), 6);
        assertEquals(problems.getSum(47, 22), 69);
    }

    @Test
    public void testKthSmallest() {
        int[][] matrix = new int[][] {{1,5,9}, {10,11,13}, {12,13,15}};
        assertEquals(problems.kthSmallest(matrix, 8), 13);
        matrix = new int[][] {{5}};
        assertEquals(problems.kthSmallest(matrix, 1), 5);
    }

    @Test
    public void testLongestSubstring() {
        assertEquals(problems.longestSubstring("aaabb", 3), 3);
        assertEquals(problems.longestSubstring("ababbc", 2), 5);
    }

    @Test
    public void testRob() {
        int[] param = new int[] {1,2,3,1};
        assertEquals(problems.rob(param), 4);
        param = new int[] {2,7,9,3,1};
        assertEquals(problems.rob(param), 12);
    }

    @Test
    public void testFirstMissingPositive() {
        int[] param = new int[] {1,2,0};
        assertEquals(problems.firstMissingPositive(param), 3);
        param = new int[] {3,4,-1,1};
        assertEquals(problems.firstMissingPositive(param), 2);
        param = new int[] {7,8,9,11,12};
        assertEquals(problems.firstMissingPositive(param), 1);
    }

    @Test
    public void testIsMatch() {
        assertFalse(problems.isMatch("aa", "a"));
        assertFalse(problems.isMatch("cb", "?a"));
        assertFalse(problems.isMatch("acdcb", "a*c?b"));
        assertTrue(problems.isMatch("aa", "*"));
        assertTrue(problems.isMatch("adceb", "*a*b"));
    }

    @Test
    public void testCombine() {
        List<List<Integer>> res = problems.combine(4, 2);
        assertEquals(res.size(), 6);
    }

    @Test
    public void testCombinationSum2() {
        int[] param = new int[] {10,1,2,7,6,1,5};
        List<List<Integer>> res = problems.combinationSum2(param, 8);
        assertEquals(res.size(), 4);
        param = new int[] {2,5,2,1,2};
        res = problems.combinationSum2(param, 5);
        assertEquals(res.size(), 2);
    }

    @Test
    public void testPermuteUnique() {
        int[] param = new int[] {1,1,2};
        List<List<Integer>> res = problems.permuteUnique(param);
        assertEquals(res.size(), 3);
        param = new int[] {3,3,0,3};
        assertEquals(problems.permuteUnique(param).size(), 4);
    }

    @Test
    public void testCountPrimes() {
        assertEquals(problems.countPrimes(10), 4);
        assertEquals(problems.countPrimes(0), 0);
        assertEquals(problems.countPrimes(1), 0);
    }

    @Test
    public void testIsHappy() {
        assertTrue(problems.isHappy(19));
        assertFalse(problems.isHappy(2));
    }
}
