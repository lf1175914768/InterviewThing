package com.study.leetcode;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

public class HotProblemTests {

    private HotProblems problem;

    @Before
    public void before() {
        problem = new HotProblems();
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
        int[] arr = {1,8,6,2,5,4,8,3,7};
        assertEquals(problem.maxArea(arr), 49);
        int[] arr2 = {1,1};
        assertEquals(problem.maxArea(arr2), 1);
        int[] arr3 = {4,3,2,1,4};
        assertEquals(problem.maxArea(arr3), 16);
    }

    @Test
    public void testThreeSum() {
        int[] arr = {-1,0,1,2,-1,-4};
        List<List<Integer>> result = problem.threeSum(arr);
        System.out.println(result);
    }

    @Test
    public void testLetterCombinations() {
        System.out.println(problem.letterCombinations("23"));
    }
}
