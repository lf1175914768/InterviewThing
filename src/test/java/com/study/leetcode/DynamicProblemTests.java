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
}
