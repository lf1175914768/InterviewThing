package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DynamicProblemTests {

    private DynamicProblems problems;

    @Before
    public void init() {
        problems = new DynamicProblems();
    }

    @Test
    public void testLengthOfLIS() {
        int[] param = new int[] {10,9,2,5,3,7,101,18};
        int length = problems.lengthOfLIS(param);
        assertEquals(length, 4);
        assertEquals(problems.lengthOfLIS_v2(param), 4);
    }

    @Test
    public void testMinFallingPathSum() {
        int[][] matrix = new int[][] {{2,1,3}, {6,5,4}, {7,8,9}};
        assertEquals(problems.minFallingPathSum(matrix), 13);
        matrix = new int[][] {{-19,57}, {-40,-5}};
        assertEquals(problems.minFallingPathSum(matrix), -59);
        matrix = new int[][] {{-48}};
        assertEquals(problems.minFallingPathSum(matrix), -48);
        matrix = new int[][] {{17,82}, {1,-44}};
        assertEquals(problems.minFallingPathSum(matrix), -27);
    }

    @Test
    public void testFindTargetSumWays() {
        int[] params = new int[] {1,1,1,1,1};
        assertEquals(problems.findTargetSumWays(params, 3), 5);
        assertEquals(problems.findTargetSumWays_v2(params, 3), 5);
        assertEquals(problems.findTargetSumWays_v3(params, 3), 5);
        params = new int[] {1};
        assertEquals(problems.findTargetSumWays(params, 1), 1);
        assertEquals(problems.findTargetSumWays_v2(params, 1), 1);
        assertEquals(problems.findTargetSumWays_v3(params, 1), 1);
        params = new int[] {0,0,0,0,0,0,0,0,1};
        assertEquals(problems.findTargetSumWays(params, 1), 256);
        assertEquals(problems.findTargetSumWays_v2(params, 1), 256);
        assertEquals(problems.findTargetSumWays_v3(params, 1), 256);
        params = new int[] {100};
        assertEquals(problems.findTargetSumWays(params, -200), 0);
        assertEquals(problems.findTargetSumWays_v2(params, -200), 0);
        assertEquals(problems.findTargetSumWays_v3(params, -200), 0);
    }

    @Test
    public void testMinDistance() {
        assertEquals(problems.minDistance("horse", "ros"), 3);
        assertEquals(problems.minDistance_v2("horse", "ros"), 3);
        assertEquals(problems.minDistance_v3("horse", "ros"), 3);
        assertEquals(problems.minDistance("ros", "horse"), 3);
        assertEquals(problems.minDistance_v2("ros", "horse"), 3);
        assertEquals(problems.minDistance_v3("ros", "horse"), 3);
        assertEquals(problems.minDistance("intention", "execution"), 5);
        assertEquals(problems.minDistance_v2("intention", "execution"), 5);
        assertEquals(problems.minDistance_v3("intention", "execution"), 5);
        assertEquals(problems.minDistance("execution", "intention"), 5);
        assertEquals(problems.minDistance_v2("execution", "intention"), 5);
        assertEquals(problems.minDistance_v3("execution", "intention"), 5);
        assertEquals(problems.minDistance("rad", "apple"), 5);
        assertEquals(problems.minDistance_v2("rad", "apple"), 5);
        assertEquals(problems.minDistance_v3("rad", "apple"), 5);
        assertEquals(problems.minDistance("apple", "rad"), 5);
        assertEquals(problems.minDistance_v2("apple", "rad"), 5);
        assertEquals(problems.minDistance_v3("apple", "rad"), 5);
    }

    @Test
    public void testMaxEnvelopes() {
        int[][] params = new int[][] {{5,4}, {6,4}, {6,7}, {2,3}};
        assertEquals(problems.maxEnvelopes(params), 3);
    }

    @Test
    public void testMaxSubArray() {
        int[] param = new int[] {-2,1,-3,4,-1,2,1,-5,4};
        assertEquals(problems.maxSubArray(param), 6);
        assertEquals(problems.maxSubArray_v2(param), 6);
        param = new int[] {1};
        assertEquals(problems.maxSubArray(param), 1);
        assertEquals(problems.maxSubArray_v2(param), 1);
        param = new int[] {5,4,-1,7,8};
        assertEquals(problems.maxSubArray(param), 23);
        assertEquals(problems.maxSubArray_v2(param), 23);
    }

    @Test
    public void testLongestCommonSubsequence() {
        assertEquals(problems.longestCommonSubsequence("abcde", "ace"), 3);
        assertEquals(problems.longestCommonSubsequence_v2("abcde", "ace"), 3);
        assertEquals(problems.longestCommonSubsequence_v3("abcde", "ace"), 3);
        assertEquals(problems.longestCommonSubsequence("abc", "abc"), 3);
        assertEquals(problems.longestCommonSubsequence_v2("abc", "abc"), 3);
        assertEquals(problems.longestCommonSubsequence_v3("abc", "abc"), 3);
        assertEquals(problems.longestCommonSubsequence("abc", "def"), 0);
        assertEquals(problems.longestCommonSubsequence_v2("abc", "def"), 0);
        assertEquals(problems.longestCommonSubsequence_v3("abc", "def"), 0);
    }

    @Test
    public void testMinDistanceOfDeletion() {
        assertEquals(problems.minDistanceOfDeletion("sea", "eat"), 2);
        assertEquals(problems.minDistanceOfDeletion_v2("sea", "eat"), 2);
    }

    @Test
    public void testChange() {
        int[] coins = new int[] {1, 2, 5};
        assertEquals(problems.change(5, coins), 4);
        assertEquals(problems.change_v2(5, coins), 4);
        coins = new int[] {2};
        assertEquals(problems.change(3, coins), 0);
        assertEquals(problems.change_v2(3, coins), 0);
        coins = new int[] {10};
        assertEquals(problems.change(10, coins), 1);
        assertEquals(problems.change_v2(10, coins), 1);
    }

    @Test
    public void testRob2() {
        int[] param = new int[] {2,3,2};
        assertEquals(problems.rob2(param), 3);
        param = new int[] {1,2,3,1};
        assertEquals(problems.rob2(param), 4);
        param = new int[] {1,2,3};
        assertEquals(problems.rob2(param), 3);
    }

    @Test
    public void testMaximalSquare() {
        char[][] param = new char[][] {{'1','0','1','0','0'}, {'1','0','1','1','1'},
                {'1','1','1','1','1'}, {'1','0','0','1','0'}};
        assertEquals(problems.maximalSquare(param), 4);
        param = new char[][] {{'0','1'}, {'1','0'}};
        assertEquals(problems.maximalSquare(param), 1);
        param = new char[][] {{'0'}};
        assertEquals(problems.maximalSquare(param), 0);
    }

    @Test
    public void testDistinctSubSeqII() {
        assertEquals(problems.distinctSubSeqII("abc"), 7);
        assertEquals(problems.distinctSubSeqII("aba"), 6);
        assertEquals(problems.distinctSubSeqII("aaa"), 3);
    }
}
