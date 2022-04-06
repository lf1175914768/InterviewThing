package com.study.leetcode;

import com.study.interview.array.NumArray;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * <p>description: 数组相关的测试类  </p>
 * <p>className:  ArrayProblemTests </p>
 * <p>create time:  2022/4/6 11:46 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class ArrayProblemTests {

    private ArrayProblems problems = new ArrayProblems();

    @Test
    public void testNumArray() {
        int[] param = new int[] {-2, 0, 3, -5, 2, -1};
        NumArray arr = new NumArray(param);
        assertEquals(arr.sumRange(0, 2), 1);
        assertEquals(arr.sumRange(2, 5), -1);
        assertEquals(arr.sumRange(0, 5), -3);
    }

    @Test
    public void testNumMatrix() {
        int[][] param = new int[][] {{3,0,1,4,2}, {5,6,3,2,1}, {1,2,0,1,5}, {4,1,0,1,7}, {1,0,3,0,5}};
        NumArray arr = new NumArray();
        arr.initMatrixSum(param);
        assertEquals(arr.sumRegion(2,1,4,3), 8);
        assertEquals(arr.sumRegion(1,1,2,2), 11);
        assertEquals(arr.sumRegion(1,2,2,4), 12);
    }

    @Test
    public void testSubarraySum() {
        int[] param = new int[] {1,1,1};
        assertEquals(problems.subarraySum(param, 2), 2);
        assertEquals(problems.subarraySum_v2(param, 2), 2);
        param = new int[] {1,2,3};
        assertEquals(problems.subarraySum(param, 3), 2);
        assertEquals(problems.subarraySum_v2(param, 3), 2);
        param = new int[] {-1, -1, 1};
        assertEquals(problems.subarraySum_v2(param, 0), 1);
        assertEquals(problems.subarraySum(param, 0), 1);
    }

    @Test
    public void testNextGreaterElements() {
        int[] param = new int[] {1,2,1};
        int[] res = new int[] {2,-1,2};
        assertArrayEquals(problems.nextGreaterElements(param), res);
    }

    @Test
    public void testMaxSlidingWindow() {
        int[] param = new int[] {1,3,-1,-3,5,3,6,7};
        int[] res = new int[] {3,3,5,5,6,7};
        assertArrayEquals(problems.maxSlidingWindow(param, 3), res);
    }
}
