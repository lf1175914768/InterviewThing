package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class HotProblemTests {

    private HotProblems problem;

    @Before
    public void before() {
        problem = new HotProblems();
    }

    @Test
    public void testLongestPalindrome() {
        String result = problem.longestPalindrome("aacabdkacaa");
        System.out.println(result);
    }

    @Test
    public void testLongestPalindrome_v2() {
        String result = problem.longestPalindrome_v2("aacabdkacaa");
        System.out.println(result);
    }

    @Test
    public void testRegexIsMatch() {
        boolean result = problem.isMatch("aab", "c*a*b");
        System.out.println(result);
        System.out.println(problem.isMatch("mississippi", "mis*is*p*."));
        System.out.println(problem.isMatch("ab", ".*"));
    }

    @Test
    public void testMaxArea() {
        int[] arr = {1, 8, 6, 2, 5, 4, 8, 3, 7};
        assertEquals(problem.maxArea(arr), 49);
        int[] arr2 = {1, 1};
        assertEquals(problem.maxArea(arr2), 1);
        int[] arr3 = {4, 3, 2, 1, 4};
        assertEquals(problem.maxArea(arr3), 16);
    }

    @Test
    public void testThreeSum() {
        int[] arr = {-1, 0, 1, 2, -1, -4};
        List<List<Integer>> result = problem.threeSum(arr);
        System.out.println(result);
    }

    @Test
    public void testLetterCombinations() {
        System.out.println(problem.letterCombinations("23"));
    }

    @Test
    public void testIsValidString() {
        assertTrue(problem.isValid("()"));
        assertTrue(problem.isValid("()[]{{}}"));
        assertTrue(problem.isValid("{[]}"));
        assertFalse(problem.isValid("(]"));
        assertFalse(problem.isValid("]"));
        assertFalse(problem.isValid("))"));
        assertFalse(problem.isValid("([)]"));
    }

    @Test
    public void testGenerateParenthesis() {
        assertEquals(problem.generateParenthesis(1).size(), 1);
        assertEquals(problem.generateParenthesis(2).size(), 2);
        assertEquals(problem.generateParenthesis(3).size(), 5);

        assertEquals(problem.generateParenthesis_v2(1).size(), 1);
        assertEquals(problem.generateParenthesis_v2(2).size(), 2);
        assertEquals(problem.generateParenthesis_v2(3).size(), 5);
    }

    @Test
    public void testMergeKLists() {
        List<HotProblems.ListNode> param = getListNodes();
        HotProblems.ListNode node = problem.mergeKLists(param.toArray(new HotProblems.ListNode[0]));
        while (node != null) {
            System.out.print(node.val + "  ");
            node = node.next;
        }
    }

    @Test
    public void testMergeKLists_v2() {
        List<HotProblems.ListNode> param = getListNodes();
        HotProblems.ListNode result = problem.mergeKLists_v2(param.toArray(new HotProblems.ListNode[0]));
        while (result != null) {
            System.out.print(result.val + "  ");
            result = result.next;
        }
    }

    @Test
    public void testNextPermutation() {
        int [] arr = {4,2,0,2,3,2,0};
        problem.nextPermutation(arr);
        for (int i = 0; i < arr.length; i++) {
            System.out.println(arr[i]);
        }
    }

    @Test
    public void testLongestValidParentheses() {
        assertEquals(problem.longestValidParentheses("(()"), 2);
        assertEquals(problem.longestValidParentheses("()(()"), 2);
        assertEquals(problem.longestValidParentheses(")()())"), 4);
        assertEquals(problem.longestValidParentheses(""), 0);
        assertEquals(problem.longestValidParentheses("(((()))))"), 8);
        assertEquals(problem.longestValidParentheses("))()())))((()))"), 6);


        assertEquals(problem.longestValidParentheses_v2("(()"), 2);
        assertEquals(problem.longestValidParentheses_v2("()(()"), 2);
        assertEquals(problem.longestValidParentheses_v2(")()())"), 4);
        assertEquals(problem.longestValidParentheses_v2(""), 0);
        assertEquals(problem.longestValidParentheses_v2("(((()))))"), 8);
        assertEquals(problem.longestValidParentheses_v2("))()())))((()))"), 6);

        assertEquals(problem.longestValidParentheses_v3("(()"), 2);
        assertEquals(problem.longestValidParentheses_v3("()(()"), 2);
        assertEquals(problem.longestValidParentheses_v3(")()())"), 4);
        assertEquals(problem.longestValidParentheses_v3(""), 0);
        assertEquals(problem.longestValidParentheses_v3("(((()))))"), 8);
        assertEquals(problem.longestValidParentheses_v3("))()())))((()))"), 6);
    }

    @Test
    public void testSearchRange() {
        int[] arr = {5,7,7,8,8,10};
        int[] arr2 = {5,7,7,8,8,9,10};
        assertEquals(problem.searchRange(arr, 8)[1], 4);
        assertEquals(problem.searchRange(arr, 8)[0], 3);
        assertEquals(problem.searchRange(arr, 6)[0], -1);
        assertEquals(problem.searchRange(arr, 6)[1], -1);
        assertEquals(problem.searchRange(arr2, 5)[1], 0);
        assertEquals(problem.searchRange(arr2, 5)[0], 0);
        assertEquals(problem.searchRange(arr2, 9)[0], 5);
        assertEquals(problem.searchRange(arr2, 9)[1], 5);
        assertEquals(problem.searchRange(arr2, 10)[1], 6);
        assertEquals(problem.searchRange(arr2, 10)[0], 6);
    }

    @Test
    public void testCombinationSum() {
        int[] arr = {2,3,6,7};
        List<List<Integer>> lists = problem.combinationSum(arr, 7);
        System.out.println(lists);

        System.out.println(problem.combinationSum_v2(arr, 7));
    }

    private List<HotProblems.ListNode> getListNodes() {
        List<HotProblems.ListNode> param = new ArrayList<>();
        HotProblems.ListNode node1 = new HotProblems.ListNode(5);
        HotProblems.ListNode node2 = new HotProblems.ListNode(4, node1);
        HotProblems.ListNode node3 = new HotProblems.ListNode(1, node2);
        param.add(node3);

        HotProblems.ListNode node4 = new HotProblems.ListNode(7);
        HotProblems.ListNode node5 = new HotProblems.ListNode(3, node4);
        HotProblems.ListNode node6 = new HotProblems.ListNode(1, node5);
        param.add(node6);

        HotProblems.ListNode node7 = new HotProblems.ListNode(10);
        HotProblems.ListNode node8 = new HotProblems.ListNode(6, node7);
        HotProblems.ListNode node9 = new HotProblems.ListNode(2, node8);
        param.add(node9);
        return param;
    }
}
