package com.study.leetcode;

import com.study.interview.array.NumArray;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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

    private final ArrayProblems problems = new ArrayProblems();

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
        assertArrayEquals(problems.nextGreaterElements_v2(param), res);
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

    @Test
    public void testFindMedianSortedArrays() {
        int[] param1 = new int[] {1,3}, param2 = new int[] {2};
        assertEquals(problems.findMedianSortedArrays(param1, param2), 2.0, 0.001);
        param1 = new int[] {1,2};
        param2 = new int[] {3,4};
        assertEquals(problems.findMedianSortedArrays(param1, param2), 2.5, 0.00001);
    }

    @Test
    public void testIntervalIntersection() {
        int[][] params1 = new int[][] {{0,2}, {5,10},{13,23},{24,25}};
        int[][] params2 = new int[][] {{1,5}, {8,12},{15,24}, {25,26}};
        int[][] res2 = new int[][] {{1,2},{5,5},{8,10},{15,23},{24,24},{25,25}};
        int[][] temp = problems.intervalIntersection(params1, params2);
        for (int i = 0; i < temp.length; i++) {
            assertArrayEquals(temp[i], res2[i]);
        }
        params1 = new int[][] {{1,3}, {5,9}};
        params2 = new int[][] {};
        assertEquals(problems.intervalIntersection(params1, params2).length, 0);
    }

    @Test
    public void testAdvantageCount() {
        int[] pa1 = new int[] {2,7,11,15}, pa2 = new int[] {1,10,4,11};
        int[] res = new int[] {2,11,7,15};
        assertArrayEquals(problems.advantageCount(pa1, pa2), res);
        pa1 = new int[] {12,24,8,32};
        pa2 = new int[] {13,25,32,11};
        res = new int[] {24,32,8,12};
        assertArrayEquals(problems.advantageCount(pa1, pa2), res);
    }

    @Test
    public void testLongestSubarray() {
        int[] param = new int[] {1,1,0,1};
        assertEquals(problems.longestSubarray(param), 3);
        param = new int[] {0,1,1,1,0,1,1,0,1};
        assertEquals(problems.longestSubarray(param), 5);
        param = new int[] {1,1,1};
        assertEquals(problems.longestSubarray(param), 2);
    }

    @Test
    public void testContainsNearbyAlmostDuplicate() {
        int[] param = new int[] {1,2,3,1};
        assertTrue(problems.containsNearbyAlmostDuplicate(param, 3, 0));
        param = new int[] {1,0,1,1};
        assertTrue(problems.containsNearbyAlmostDuplicate(param, 1, 2));
        param = new int[] {1,5,9,1,5,9};
        assertFalse(problems.containsNearbyAlmostDuplicate(param, 2,3));
    }

    @Test
    public void testNumSubarrayProductLessThanK() {
        int[] param = new int[] {10,5,2,6};
        assertEquals(problems.numSubarrayProductLessThanK(param, 100), 8);
        assertEquals(problems.numSubarrayProductLessThanK(param, 0), 0);
    }

    @Test
    public void testJump() {
        int[] param = new int[] {2,3,1,1,4};
        assertEquals(problems.jump(param), 2);
        assertEquals(problems.jump_v2(param), 2);
        param = new int[] {2,3,0,1,4};
        assertEquals(problems.jump(param), 2);
        assertEquals(problems.jump_v2(param), 2);
    }

    @Test
    public void testInsert() {
        int[][] param = new int[][] {{1,3}, {6,9}}, res = new int[][] {{1,5}, {6,9}};
        int[] newInterval = new int[] {2, 5};
        assertArrayEquals(problems.insert(param, newInterval), res);
        param = new int[][] {{1,2}, {3,5},{6,7},{8,10},{12,16}};
        newInterval = new int[] {4,8};
        res = new int[][] {{1,2},{3,10},{12,16}};
        assertArrayEquals(problems.insert(param, newInterval), res);
        param = new int[][] {}; newInterval = new int[] {5,7}; res = new int[][] {{5,7}};
        assertArrayEquals(problems.insert(param, newInterval), res);
        param = new int[][] {{1,5}}; newInterval = new int[] {2,3}; res = new int[][] {{1,5}};
        assertArrayEquals(problems.insert(param, newInterval), res);
        param = new int[][] {{1,5}}; newInterval = new int[] {2,7}; res = new int[][] {{1,7}};
        assertArrayEquals(problems.insert(param, newInterval), res);
    }

    @Test
    public void testSetZeroes() {
        int[][] matrix = new int[][] {{1,1,1},{1,0,1},{1,1,1}},
        res = new int[][] {{1,0,1},{0,0,0},{1,0,1}};
        problems.setZeroes(matrix);
        assertArrayEquals(matrix, res);
        matrix = new int[][] {{0,1,2,0}, {3,4,5,2}, {1,3,1,5}};
        res = new int[][] {{0,0,0,0}, {0,4,5,0}, {0,3,1,0}};
        problems.setZeroes(matrix);
        assertArrayEquals(matrix, res);
        matrix = new int[][] {{1,1,1},{1,0,1},{1,1,1}};
        res = new int[][] {{1,0,1},{0,0,0},{1,0,1}};
        problems.setZeroes_v2(matrix);
        assertArrayEquals(matrix, res);
        matrix = new int[][] {{0,1,2,0}, {3,4,5,2}, {1,3,1,5}};
        res = new int[][] {{0,0,0,0}, {0,4,5,0}, {0,3,1,0}};
        problems.setZeroes_v2(matrix);
        assertArrayEquals(matrix, res);
    }

    @Test
    public void testMinimumTotal() {
        List<List<Integer>> param = new ArrayList<>();
        param.add(Collections.singletonList(2));
        param.add(Arrays.asList(3,4));
        param.add(Arrays.asList(6,5,7));
        param.add(Arrays.asList(4,1,8,3));
        assertEquals(problems.minimumTotal(param), 11);
        param.clear();
        param.add(Collections.singletonList(-10));
        assertEquals(problems.minimumTotal(param), -10);
    }

    @Test
    public void testSolveSudoku() {
        char[][] param = new char[][] {
                {'5','3','.','.','7','.','.','.','.'},
                {'6','.','.','1','9','5','.','.','.'},
                {'.','9','8','.','.','.','.','6','.'},
                {'8','.','.','.','6','.','.','.','3'},
                {'4','.','.','8','.','3','.','.','1'},
                {'7','.','.','.','2','.','.','.','6'},
                {'.','6','.','.','.','.','2','8','.'},
                {'.','.','.','4','1','9','.','.','5'},
                {'.','.','.','.','8','.','.','7','9'}};
        char[][] res = new char[][] {
                {'5','3','4','6','7','8','9','1','2'},
                {'6','7','2','1','9','5','3','4','8'},
                {'1','9','8','3','4','2','5','6','7'},
                {'8','5','9','7','6','1','4','2','3'},
                {'4','2','6','8','5','3','7','9','1'},
                {'7','1','3','9','2','4','8','5','6'},
                {'9','6','1','5','3','7','2','8','4'},
                {'2','8','7','4','1','9','6','3','5'},
                {'3','4','5','2','8','6','1','7','9'}};

        ArrayProblems.Sudoku problem = new ArrayProblems.Sudoku();
        problem.solveSudoku(param);
        assertArrayEquals(param, res);
    }

    @Test
    public void testRemoveDuplicatesTwo() {
        int[] param = {0,0,1,1,1,1,2,3,3};
        int[] res = {0,0,1,1,2,3,3};
        assertArrayEquals(Arrays.copyOf(param, problems.removeDuplicatesTwo(param)), res);
        param = new int[] {0,0,1,1,1,1,2,3,3};
        assertArrayEquals(Arrays.copyOf(param, problems.removeDuplicatesTwo_v2(param)), res);
    }

    @Test
    public void testCanCompleteCircuit() {
        int[] gas = {1,2,3,4,5}, cost = {3,4,5,1,2};
        assertEquals(problems.canCompleteCircuit(gas, cost), 3);
        gas = new int[] {2,3,4}; cost = new int[] {3,4,3};
        assertEquals(problems.canCompleteCircuit(gas, cost), -1);
    }

    @Test
    public void testCandy() {
        int[] param = {1,0,2};
        assertEquals(problems.candy(param), 5);
        param = new int[] {1,2,2};
        assertEquals(problems.candy(param), 4);
    }

    @Test
    public void testEvalRPN() {
        String[] param = {"2","1","+","3","*"};
        assertEquals(problems.evalRPN(param), 9);
        param = new String[] {"4","13","5","/","+"};
        assertEquals(problems.evalRPN(param), 6);
        param = new String[] {"10","6","9","3","+","-11","*","/","*","17","+","5","+"};
        assertEquals(problems.evalRPN(param), 22);
    }

    @Test
    public void testTopKFrequent() {
        int[] param = {1,1,1,2,2,3}, res = {2,1};
        assertArrayEquals(problems.topKFrequent(param, 2), res);
        assertArrayEquals(problems.topKFrequent_v2(param, 2), res);
        param = new int[] {1}; res = new int[] {1};
        assertArrayEquals(problems.topKFrequent(param,1), res);
        assertArrayEquals(problems.topKFrequent_v2(param,1), res);
    }

    @Test
    public void testMinPatches() {
        int[] param = {1,3};
        assertEquals(problems.minPatches(param, 6), 1);
        assertEquals(problems.minPatches_v2(param, 6), 1);
        param = new int[] {1,5,10};
        assertEquals(problems.minPatches(param, 20), 2);
        assertEquals(problems.minPatches_v2(param, 20), 2);
        param = new int[] {1,2,2};
        assertEquals(problems.minPatches(param, 5), 0);
        assertEquals(problems.minPatches_v2(param, 5), 0);
    }

    @Test(timeout = 100)
    public void testWiggleMaxLength() {
        int[] param = {1,7,4,9,2,5};
        assertEquals(problems.wiggleMaxLength(param), 6);
        assertEquals(problems.wiggleMaxLength_v2(param), 6);
        assertEquals(problems.wiggleMaxLength_v3(param), 6);
        param = new int[] {1,17,5,10,13,15,10,5,16,8};
        assertEquals(problems.wiggleMaxLength(param), 7);
        assertEquals(problems.wiggleMaxLength_v2(param), 7);
        assertEquals(problems.wiggleMaxLength_v3(param), 7);
        param = new int[] {1,2,3,4,5,6,7,8,9};
        assertEquals(problems.wiggleMaxLength(param), 2);
        assertEquals(problems.wiggleMaxLength_v2(param), 2);
        assertEquals(problems.wiggleMaxLength_v3(param), 2);
    }

    @Test
    public void testFindMinArrowShots() {
        int[][] param = {{10,16},{2,8},{1,6},{7,12}};
        assertEquals(problems.findMinArrowShots(param), 2);
        param = new int[][] {{1,2},{3,4},{5,6},{7,8}};
        assertEquals(problems.findMinArrowShots(param), 4);
        param = new int[][] {{1,2},{2,3},{3,4},{4,5}};
        assertEquals(problems.findMinArrowShots(param), 2);
    }

    @Test
    public void testTriangleNumber() {
        int[] param = {2,2,3,4};
        assertEquals(problems.triangleNumber(param), 3);
        assertEquals(problems.triangleNumber_v2(param), 3);
        param = new int[] {4,2,3,4};
        assertEquals(problems.triangleNumber(param), 4);
        assertEquals(problems.triangleNumber_v2(param), 4);
    }

    @Test
    public void testLeastInterval() {
        char[] param = new char[] {'A','A','A','B','B','B'};
        assertEquals(problems.leastInterval(param, 2), 8);
        param = new char[] {'A','A','A','B','B','B'};
        assertEquals(problems.leastInterval(param, 0), 6);
        param = new char[] {'A','A','A','A','A','A','B','C','D','E','F','G'};
        assertEquals(problems.leastInterval(param, 2), 16);
    }

    @Test
    public void testIsPossible() {
        int[] param = {1,2,3,3,4,5};
        assertTrue(problems.isPossible(param));
        param = new int[] {1,2,3,3,4,4,5,5};
        assertTrue(problems.isPossible(param));
        param = new int[] {1,2,3,4,4,5};
        assertFalse(problems.isPossible(param));
    }

    @Test
    public void testLongestIncreasingPath() {
        int[][] matrix = {{9,9,4}, {6,6,8},{2,1,1}};
        assertEquals(problems.longestIncreasingPath(matrix), 4);
        matrix = new int[][] {{3,4,5}, {3,2,6}, {2,2,1}};
        assertEquals(problems.longestIncreasingPath(matrix), 4);
        matrix = new int[][] {{1}};
        assertEquals(problems.longestIncreasingPath(matrix), 1);
    }

    @Test
    public void testThreeNumClosest() {
        int[] param = {-1,2,1,-4};
        assertEquals(problems.threeSumClosest(param, 1), 2);
        param = new int[] {0,0,0};
        assertEquals(problems.threeSumClosest(param, 1), 0);
    }

    @Test
    public void testKthLargest() {
        ArrayProblems.KthLargest problem = new ArrayProblems.KthLargest(3, new int[] {4,5,8,2});
        assertEquals(problem.add(3), 4);
        assertEquals(problem.add(5), 5);
        assertEquals(problem.add(10), 5);
        assertEquals(problem.add(9), 8);
        assertEquals(problem.add(4), 8);
    }

    @Test
    public void testValidaStackSequences() {
        int[] pushed = {1,2,3,4,5}, popped = {4,5,3,2,1};
        assertTrue(problems.validateStackSequences(pushed, popped));
        assertTrue(problems.validateStackSequences_v2(pushed, popped));
        popped = new int[] {4,3,5,1,2};
        assertFalse(problems.validateStackSequences(pushed, popped));
        assertFalse(problems.validateStackSequences_v2(pushed, popped));
        popped = new int[] {5,4,3,2,1};
        assertTrue(problems.validateStackSequences(pushed, popped));
        assertTrue(problems.validateStackSequences_v2(pushed, popped));
    }

    @Test
    public void testGetLeastNumbers() {
        int[] arr = {3,2,1}, res = {1,2};
        assertArrayEquals(problems.getLeastNumbers(arr, 2), res);
        arr = new int[] {0,1,2,1};
        res = new int[] {0};
        assertArrayEquals(problems.getLeastNumbers(arr, 1), res);
    }
}
