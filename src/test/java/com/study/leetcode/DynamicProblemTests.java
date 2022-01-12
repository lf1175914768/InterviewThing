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
}
