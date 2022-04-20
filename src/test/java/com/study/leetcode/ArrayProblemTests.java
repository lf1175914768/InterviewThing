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

    @Test
    public void testMinEatingSpeed() {
        int[] params = new int[] {3,6,7,11};
        assertEquals(problems.minEatingSpeed(params, 8), 4);
        params = new int[] {30,11,23,4,20};
        assertEquals(problems.minEatingSpeed(params, 5), 30);
        assertEquals(problems.minEatingSpeed(params, 6), 23);
    }

    @Test
    public void testFoldStick() {
        int[] param = new int[] {3, 5, 13, 9, 12};
        assertEquals(problems.foldStick(param), 1);
        param = new int[] {3,12,13,9,12};
        assertEquals(problems.foldStick(param), 2);
        param = new int[] {3,13,12,9,12};
        assertEquals(problems.foldStick(param), 3);
        param = new int[] {3,13,60,7};
        assertEquals(problems.foldStick(param), 10);
        param = new int[] {3,63,7};
        assertEquals(problems.foldStick(param), 8);
        param = new int[] {9, 1};
        assertEquals(problems.foldStick(param), 8);
    }

    @Test
    public void testSearchMatrix() {
        int[][] matrix = new int[][] {{1,4,7,11,15}, {2,5,8,12,19}, {3,6,9,16,22}, {10,13,14,17,24}, {18,21,23,26,30}};
        assertTrue(problems.searchMatrix(matrix, 5));
        assertTrue(problems.searchMatrix_v2(matrix, 5));
        assertFalse(problems.searchMatrix(matrix, 20));
        assertFalse(problems.searchMatrix_v2(matrix, 20));
    }

    @Test
    public void testFindLength() {
        int[] nums1 = new int[] {1,2,3,2,1};
        int[] nums2 = new int[] {3,2,1,4,7};
        assertEquals(problems.findLength(nums1, nums2), 3);
        assertEquals(problems.findLength_v2(nums1, nums2), 3);
        nums1 = new int[] {0,0,0,0,0};
        nums2 = new int[] {0,0,0,0,0};
        assertEquals(problems.findLength(nums1, nums2), 5);
        assertEquals(problems.findLength_v2(nums1, nums2), 5);
    }

    @Test
    public void testMinSubArrayLen() {
        int[] param = new int[] {2,3,1,2,4,3};
        assertEquals(problems.minSubArrayLen(7, param), 2);
        param = new int[] {1,4,4};
        assertEquals(problems.minSubArrayLen(4, param), 1);
        param = new int[] {1,1,1,1,1,1,1,1};
        assertEquals(problems.minSubArrayLen(11, param), 0);
    }
}
