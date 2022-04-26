package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class HotProblemTests {

    private HotProblems problem;

    @Before
    public void before() {
        problem = new HotProblems();
    }

    @Test
    public void testBinarySearch() {
        int[] params = new int[] {1,2,3,4,5,6,7,8,9};
        assertEquals(problem.binarySearch(params, 4), 3);
        assertEquals(problem.binarySearch(params, 10), -1);
        assertEquals(problem.binarySearch(params, -1), -1);
        assertEquals(problem.binarySearch_v2(params, -1), -1);
        assertEquals(problem.binarySearch_v2(params, 10), -1);
        assertEquals(problem.binarySearch_v2(params, 5), 4);

        params = new int[] {1,2,3,3,3,3,8,9,15,28};
        assertEquals(problem.binaryLeftSearch(params, 3), 2);
        assertEquals(problem.binaryLeftSearch_v2(params, 3), 2);
        assertEquals(problem.binaryLeftSearch(params, 8), 6);
        assertEquals(problem.binaryLeftSearch_v2(params, 8), 6);
        assertEquals(problem.binaryLeftSearch(params, -1), -1);
        assertEquals(problem.binaryLeftSearch_v2(params, -1), -1);
        assertEquals(problem.binaryLeftSearch(params, 29), -1);
        assertEquals(problem.binaryLeftSearch_v2(params, 29), -1);
        assertEquals(problem.binaryLeftSearch(params, 16), -1);
        assertEquals(problem.binaryLeftSearch_v2(params, 16), -1);

        assertEquals(problem.binaryRightSearch(params, 3), 5);
        assertEquals(problem.binaryRightSearch_v2(params, 3), 5);
        assertEquals(problem.binaryRightSearch(params, 8), 6);
        assertEquals(problem.binaryRightSearch_v2(params, 8), 6);
        assertEquals(problem.binaryRightSearch(params, -1), -1);
        assertEquals(problem.binaryRightSearch_v2(params, -1), -1);
        assertEquals(problem.binaryRightSearch(params, 29), -1);
        assertEquals(problem.binaryRightSearch_v2(params, 29), -1);
        assertEquals(problem.binaryRightSearch(params, 16), -1);
        assertEquals(problem.binaryRightSearch_v2(params, 16), -1);
    }

    @Test
    public void testLongestPalindrome() {
        String result = problem.longestPalindrome("aacabdkacaa");
        System.out.println(result);
    }

    @Test
    public void testLongestPalindrome_v2() {
        assertEquals(problem.longestPalindrome_v2("aacabdkacaa"), "aca");
        assertEquals(problem.longestPalindrome_v3("aacabdkacaa"), "aca");
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
        List<ListNode> param = getListNodes();
        ListNode node = problem.mergeKLists(param.toArray(new ListNode[0]));
        while (node != null) {
            System.out.print(node.val + "  ");
            node = node.next;
        }
    }

    @Test
    public void testMergeKLists_v2() {
        List<ListNode> param = getListNodes();
        ListNode result = problem.mergeKLists_v2(param.toArray(new ListNode[0]));
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

    @Test
    public void testMinWindow() {
        assertEquals(problem.minWindow("ADOBECODEBANC", "ABC"), "BANC");
        assertEquals(problem.minWindow("a", "a"), "a");
        assertEquals(problem.minWindow("a", "aa"), "");
    }

    @Test
    public void testCheckInclusion() {
        assertTrue(problem.checkInclusion("ab", "eidbaooo"));
        assertFalse(problem.checkInclusion("ab", "eidboaoo"));
        assertFalse(problem.checkInclusion("abc", "ccccbbbbaaaa"));
    }

    @Test
    public void testLengthOfLongestSubstring() {
        assertEquals(problem.lengthOfLongestSubstring("abcabcbb"), 3);
        assertEquals(problem.lengthOfLongestSubstring("bbbbb"), 1);
        assertEquals(problem.lengthOfLongestSubstring("pwwkew"), 3);
    }

    @Test
    public void testNetworkDelayTime() {
        int[][] times = new int[3][3];
        times[0] = new int[] {2,1,1};
        times[1] = new int[] {2,3,1};
        times[2] = new int[] {3,4,1};
        assertEquals(problem.networkDelayTime(times, 4, 2), 2);
        times = new int[1][3];
        times[0] = new int[] {1,2,1};
        assertEquals(problem.networkDelayTime(times, 2, 1), 1);
        assertEquals(problem.networkDelayTime(times, 2, 2), -1);
    }

    @Test
    public void testMaxProfit() {
        int[] prices = new int[] {7,1,5,3,6,4};
        assertEquals(problem.maxProfit(prices), 5);
        assertEquals(problem.maxProfit_v1(prices), 5);

        prices = new int[] {7,6,4,3,1};
        assertEquals(problem.maxProfit(prices), 0);
        assertEquals(problem.maxProfit_v1(prices), 0);
    }

    @Test
    public void testMaxProfitWithInfinity() {
        int[] prices = new int[] {7,1,5,3,6,4};
        assertEquals(problem.maxProfit_infinity(prices), 7);
        assertEquals(problem.maxProfit_infinity_v2(prices), 7);

        prices = new int[] {7,6,4,3,1};
        assertEquals(problem.maxProfit_infinity(prices), 0);
        assertEquals(problem.maxProfit_infinity_v2(prices), 0);

        prices = new int[] {1,2,3,4,5};
        assertEquals(problem.maxProfit_infinity(prices), 4);
        assertEquals(problem.maxProfit_infinity_v2(prices), 4);
    }

    @Test
    public void testMaxProfitWithCool() {
        int[] prices = new int[] {3,3,5,0,0,3,1,4};
        assertEquals(problem.maxProfit_withCool(prices), 6);
        assertEquals(problem.maxProfit_withCool_v2(prices), 6);

        prices = new int[] {1,2,3,4,5};
        assertEquals(problem.maxProfit_withCool(prices), 4);
        assertEquals(problem.maxProfit_withCool_v2(prices), 4);

        prices = new int[] {7,6,4,3,1};
        assertEquals(problem.maxProfit_withCool(prices), 0);
        assertEquals(problem.maxProfit_withCool_v2(prices), 0);
    }

    @Test
    public void testReverse() {
//        assertEquals(problem.reverse(123), 321);
//        assertEquals(problem.reverse(-123), -321);
        assertEquals(problem.reverse(1200), 21);
    }

    private List<ListNode> getListNodes() {
        List<ListNode> param = new ArrayList<>();
        ListNode node1 = new ListNode(5);
        ListNode node2 = new ListNode(4, node1);
        ListNode node3 = new ListNode(1, node2);
        param.add(node3);

        ListNode node4 = new ListNode(7);
        ListNode node5 = new ListNode(3, node4);
        ListNode node6 = new ListNode(1, node5);
        param.add(node6);

        ListNode node7 = new ListNode(10);
        ListNode node8 = new ListNode(6, node7);
        ListNode node9 = new ListNode(2, node8);
        param.add(node9);
        return param;
    }

    @Test
    public void testPermute() {
        int[] param = new int[] {1,2,3};
        assertEquals(problem.permute(param).size(), 6);
        param = new int[] {0, 1};
        assertEquals(problem.permute(param).size(), 2);
        param = new int[] {1};
        assertEquals(problem.permute(param).size(), 1);
    }

    @Test
    public void testRotate() {
        int[][] matrix = new int[][] {{1,2,3}, {4,5,6}, {7,8,9}};
        problem.rotate(matrix);
        assertEquals(matrix[0][0], 7);
        assertEquals(matrix[0][1], 4);
        assertEquals(matrix[0][2], 1);
        assertEquals(matrix[1][0], 8);
        assertEquals(matrix[1][1], 5);
        assertEquals(matrix[1][2], 2);
    }

    @Test
    public void testCanJump() {
        int[] param = new int[] {2,3,1,1,4};
        assertTrue(problem.canJump(param));
        assertTrue(problem.canJump_v2(param));
        param = new int[] {3,2,1,0,4};
        assertFalse(problem.canJump(param));
        assertFalse(problem.canJump_v2(param));
    }

    @Test
    public void testMerge() {
        int[][] params = new int[][] {{1,3}, {2,6}, {8,10}, {15,18}};
        int[][] rs = problem.merge(params);
        assertEquals(problem.merge(params).length, 3);
        assertEquals(rs[0][0], 1);
        assertEquals(rs[0][1], 6);
        assertEquals(rs[1][0], 8);
        assertEquals(rs[1][1], 10);
    }

    @Test
    public void testUniquePaths() {
        assertEquals(problem.uniquePaths(3, 7), 28);
        assertEquals(problem.uniquePaths_v2(3, 7), 28);
        assertEquals(problem.uniquePaths(3, 2), 3);
        assertEquals(problem.uniquePaths_v2(3, 2), 3);
        assertEquals(problem.uniquePaths(7, 3), 28);
        assertEquals(problem.uniquePaths_v2(7, 3), 28);
        assertEquals(problem.uniquePaths(3, 3), 6);
        assertEquals(problem.uniquePaths_v2(3, 3), 6);
    }

    @Test
    public void testMinPathSum() {
        int[][] grid = new int[][] {{1,3,1}, {1,5,1}, {4,2,1}};
        assertEquals(problem.minPathSum(grid), 7);
        grid = new int[][] {{1,2,3}, {4,5,6}};
        assertEquals(problem.minPathSum(grid), 12);
    }

    @Test
    public void testClimbStairs() {
        assertEquals(problem.climbStairs(2), 2);
        assertEquals(problem.climbStairs(3), 3);
        assertEquals(problem.climbStairs(4), 5);
    }

    @Test
    public void testSortColors() {
        int[] params = new int[] {2,0,2,1,1,0};
        problem.sortColors(params);
        assertEquals(params[0], 0);
        assertEquals(params[1], 0);
        assertEquals(params[2], 1);
        assertEquals(params[3], 1);
        assertEquals(params[4], 2);
        assertEquals(params[5], 2);
        params = new int[] {2, 0 ,1};
        problem.sortColors(params);
        assertEquals(params[0], 0);
        assertEquals(params[1], 1);
        assertEquals(params[2], 2);
    }

    @Test
    public void testSubsets() {
        int[] params = new int[] {1,2,3};
        List<List<Integer>> results = problem.subsets(params);
        assertEquals(results.size(), 8);
    }

    @Test
    public void testSubsetsWithDup() {
        int[] params = new int[] {1,2,2};
        List<List<Integer>> results = problem.subsetsWithDup(params);
        assertEquals(results.size(), 6);
        params = new int[] {0};
        results = problem.subsetsWithDup(params);
        assertEquals(results.size(), 2);
    }

    @Test
    public void testExists() {
        char[][] board = new char[][] {{'A','B','C','E'}, {'S','F','C','S'}, {'A','D','E','E'}};
        assertTrue(problem.exist(board, "ABCCED"));
        assertTrue(problem.exist(board, "SEE"));
        assertFalse(problem.exist(board, "ABCB"));
        board = new char[][] {{'a'}};
        assertTrue(problem.exist(board, "a"));
    }

    @Test
    public void testLargestRectangleArea() {
        int[] params = new int[] {2,1,5,6,2,3};
        assertEquals(problem.largestRectangleArea(params), 10);
        assertEquals(problem.largestRectangleArea_v2(params), 10);
        params = new int[] {2,4};
        assertEquals(problem.largestRectangleArea(params), 4);
        assertEquals(problem.largestRectangleArea_v2(params), 4);
        params = new int[] {1,2,3,4,5};
        assertEquals(problem.largestRectangleArea(params), 9);
        assertEquals(problem.largestRectangleArea_v2(params), 9);
    }

    @Test
    public void testMaximalRectangle() {
        char[][] rectangle = new char[][] {{'1','0','1','0','0'}, {'1','0','1','1','1'}, {'1','1','1','1','1'}, {'1','0','0','1','0'}};
        assertEquals(problem.maximalRectangle(rectangle), 6);
        rectangle = new char[][] {{}};
        assertEquals(problem.maximalRectangle(rectangle), 0);
        rectangle = new char[][] {{'0'}};
        assertEquals(problem.maximalRectangle(rectangle), 0);
        rectangle = new char[][]{{'1'}};
        assertEquals(problem.maximalRectangle(rectangle), 1);
    }

    @Test
    public void testLongestConsecutive() {
        int[] params = new int[] {100,4,200,1,3,2};
        assertEquals(problem.longestConsecutive(params), 4);
        assertEquals(problem.longestConsecutive_v2(params), 4);
        params = new int[] {0,3,7,2,5,8,4,6,0,1};
        assertEquals(problem.longestConsecutive(params), 9);
        assertEquals(problem.longestConsecutive_v2(params), 9);
        params = new int[] {0,3,7,2,5,8,6,0,1,4};
        assertEquals(problem.longestConsecutive_v2(params), 9);
    }

    @Test
    public void testWordBreak() {
        List<String> words = new ArrayList<>();
        words.add("leet");
        words.add("code");
        assertTrue(problem.wordBreak("leetcode", words));
        assertTrue(problem.wordBreak_v2("leetcode", words));
        words.clear();
        words.add("apple");
        words.add("pen");
        assertTrue(problem.wordBreak("applepenapple", words));
        assertTrue(problem.wordBreak_v2("applepenapple", words));
        words = Arrays.asList("cats", "dog", "sand", "and", "cat");
        assertFalse(problem.wordBreak("catsandog", words));
        assertFalse(problem.wordBreak_v2("catsandog", words));
        words = Arrays.asList("a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa");
        assertFalse(problem.wordBreak("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", words));
        assertFalse(problem.wordBreak_v2("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", words));
    }

    @Test
    public void testDetectCycle() {
        ListNode node4 = new ListNode(4);
        ListNode node0 = new ListNode(0, node4);
        ListNode node2 = new ListNode(2, node0);
        ListNode node3 = new ListNode(3, node2);
        node4.next = node2;
        assertEquals(problem.detectCycle(node3).val, 2);
    }

    @Test
    public void testMaxProduct() {
        int[] params = new int[] {2,3,-2,4};
        assertEquals(problem.maxProduct(params), 6);
        assertEquals(problem.maxProduct_v2(params), 6);
        params = new int[] {-2,0,-1};
        assertEquals(problem.maxProduct(params), 0);
        assertEquals(problem.maxProduct_v2(params), 0);
    }

    @Test
    public void testMajorityElement() {
        int[] params = new int[] {3,2,3};
        assertEquals(problem.majorityElement(params), 3);
        params = new int[] {2,2,1,1,1,2,2};
        assertEquals(problem.majorityElement(params), 2);
    }

    @Test
    public void testReverseList() {
        ListNode node4 = new ListNode(4);
        ListNode node0 = new ListNode(0, node4);
        ListNode node2 = new ListNode(2, node0);
        ListNode node;
        assertEquals((node = problem.reverseList(node2)).val, 4);
        assertEquals(problem.reverseList_v2(node).val, 2);
    }

    @Test
    public void testFindKthLargest() {
        int[] pa = new int[] {3,2,1,5,6,4};
        assertEquals(problem.findKthLargest(pa, 2), 5);
        assertEquals(problem.findKthLargestManual(pa, 2), 5);
        assertEquals(problem.findKthLargest_v2(pa, 2), 5);
        assertEquals(problem.findKthLargest_v3(pa, 2), 5);
        pa = new int[] {3,2,3,1,2,4,5,5,6};
        assertEquals(problem.findKthLargest(pa, 4), 4);
        assertEquals(problem.findKthLargestManual(pa, 4), 4);
        assertEquals(problem.findKthLargest_v2(pa, 4), 4);
        assertEquals(problem.findKthLargest_v3(pa, 4), 4);
    }

    @Test
    public void testIsPalindrome() {
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2, node1);
        ListNode node22 = new ListNode(2, node2);
        ListNode node11 = new ListNode(1, node22);
        assertTrue(problem.isPalindrome(node11));
        node22.next = new ListNode(3, node2);
        assertTrue(problem.isPalindrome(node11));
    }

    @Test
    public void testProductExceptSelf() {
        int [] pa = new int[] {1,2,3,4};
        int[] res = problem.productExceptSelf(pa);
        assertEquals(res[0], 24);
        assertEquals(res[1], 12);
        assertEquals(res[2], 8);
        assertEquals(res[3], 6);

        pa = new int[] {-1,1,0,-3,3};
        res = new int[] {0,0,9,0,0};
        assertArrayEquals(problem.productExceptSelf(pa), res);
    }

    @Test
    public void testNumSquare() {
        assertEquals(problem.numSquares(12), 3);
        assertEquals(problem.numSquares(13), 2);
    }

    @Test
    public void testDecodeString() {
        assertEquals(problem.decodeString("3[a]2[bc]"), "aaabcbc");
        assertEquals(problem.decodeString("3[a2[c]]"), "accaccacc");
        assertEquals(problem.decodeString("2[abc]3[cd]ef"), "abcabccdcdcdef");
    }

    @Test
    public void testCountSubstrings() {
        assertEquals(problem.countSubstrings("abc"), 3);
        assertEquals(problem.countSubstrings("aaa"), 6);
    }

    @Test
    public void testFindUnsortedSubArray() {
        int[] pa = new int[] {2,6,4,8,10,9,15};
        assertEquals(problem.findUnsortedSubArray(pa), 5);
        pa = new int[] {1,2,3,4};
        assertEquals(problem.findUnsortedSubArray(pa), 0);
    }

    @Test
    public void testFindAnagrams() {
        Integer[] res = new Integer[] {0, 6};
        assertArrayEquals(problem.findAnagrams("cbaebabacd", "abc").toArray(new Integer[0]), res);
        res = new Integer[] {0,1,2};
        assertArrayEquals(problem.findAnagrams("abab", "ab").toArray(new Integer[0]), res);
        res = new Integer[] {};
        assertArrayEquals(problem.findAnagrams("dabsadl", "cw").toArray(new Integer[0]), res);
    }

    @Test
    public void testDailyTemperatures() {
        int[] pa = new int[] {73,74,75,71,69,72,76,73};
        int[] res = new int[] {1,1,4,2,1,1,0,0};
        assertArrayEquals(problem.dailyTemperatures(pa), res);
        assertArrayEquals(problem.dailyTemperatures_v2(pa), res);
        pa = new int[] {30,40,50,60};
        res = new int[] {1,1,1,0};
        assertArrayEquals(problem.dailyTemperatures(pa), res);
        assertArrayEquals(problem.dailyTemperatures_v2(pa), res);
        pa = new int[] {89,62,70,58,47,47,46,76,100,70};
        res = new int[] {8,1,5,4,3,2,1,1,0,0};
        assertArrayEquals(problem.dailyTemperatures(pa), res);
        assertArrayEquals(problem.dailyTemperatures_v2(pa), res);
    }

    @Test
    public void testFindDuplicate() {
        int[] pa = new int[] {1,3,4,2,2};
        assertEquals(problem.findDuplicate(pa), 2);
        pa = new int[] {3,1,3,4,2};
        assertEquals(problem.findDuplicate(pa), 3);
    }

    @Test
    public void testSumOfTwoString() {
        assertEquals(problem.sumOfTwoString("9999999", "1"), "10000000");
        assertEquals(problem.sumOfTwoString("9999999", "9"), "10000008");
        assertEquals(problem.sumOfTwoString("999999955555", "594"), "999999956149");
    }

    @Test
    public void testTrap() {
        int[] param = new int[] {0,1,0,2,1,0,1,3,2,1,2,1};
        assertEquals(problem.trap(param), 6);
        assertEquals(problem.trap_v2(param), 6);
        assertEquals(problem.trap_v3(param), 6);
        param = new int[] {4,2,0,3,2,5};
        assertEquals(problem.trap(param), 9);
        assertEquals(problem.trap_v2(param), 9);
        assertEquals(problem.trap_v3(param), 9);
    }

    @Test
    public void testSearch() {
        int[] param = new int[] {4,5,6,7,0,1,2};
        assertEquals(problem.search(param, 0), 4);
        assertEquals(problem.search(param, 3), -1);
        param = new int[] {1};
        assertEquals(problem.search(param, 0), -1);
    }

    @Test
    public void testMySqrt() {
        assertEquals(problem.mySqrt(9), 3);
        assertEquals(problem.mySqrt(8), 2);
    }

    @Test
    public void testSpiralOrder() {
        int[][] param = new int[][] {{1,2,3}, {4,5,6}, {7,8,9}};
        Integer[] res = new Integer[] {1,2,3,6,9,8,7,4,5};
        assertArrayEquals(problem.spiralOrder(param).toArray(new Integer[0]), res);
        param = new int[][] {{3},{2}};
        res = new Integer[] {3,2};
        assertArrayEquals(problem.spiralOrder(param).toArray(new Integer[0]), res);
        param = new int[][] {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}};
        res = new Integer[] {1,2,3,4,8,12,11,10,9,5,6,7};
        assertArrayEquals(problem.spiralOrder(param).toArray(new Integer[0]), res);
    }
}
