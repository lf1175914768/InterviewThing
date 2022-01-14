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
}
