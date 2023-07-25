package com.stud.algorithm;

import com.study.algorithm.Sort;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

public class SortTest {

    private final Sort sort = new Sort();
    private final int[] arr = new int[] {4,3,5,7,8,9,20,-1};

    @Test
    public void testBubbleSort() {
        int[] test = Arrays.copyOf(arr, arr.length), res = Arrays.copyOf(arr, arr.length);
        Arrays.sort(res);
        sort.bubbleSort(test);
        int[] test2 = Arrays.copyOf(arr, arr.length);
        sort.bubbleSortOptimize(test2);
        assertArrayEquals(test2, res);
        assertArrayEquals(test, res);
    }

    @Test
    public void testSelectionSort() {
        int[] test = Arrays.copyOf(arr, arr.length), res = Arrays.copyOf(arr, arr.length);
        Arrays.sort(res);
        sort.selectionSort(test);
        assertArrayEquals(test, res);
    }

    @Test
    public void testInsertionSort() {
        int[] test = Arrays.copyOf(arr, arr.length), res = Arrays.copyOf(arr, arr.length);
        Arrays.sort(res);
        sort.insertionSort(test);
        assertArrayEquals(test, res);
    }

    @Test
    public void testShellSort() {
        int[] test = Arrays.copyOf(arr, arr.length), res = Arrays.copyOf(arr, arr.length);
        Arrays.sort(res);
        sort.shellSort(test);
        assertArrayEquals(test, res);
    }

    @Test
    public void testMergeSort() {
        int[] test = Arrays.copyOf(arr, arr.length), res = Arrays.copyOf(arr, arr.length);
        Arrays.sort(res);
        int[] test2 = Arrays.copyOf(arr, arr.length);
        sort.mergeSort_v2(test2);
        assertArrayEquals(test2, res);
        sort.mergeSort(test);
        assertArrayEquals(test, res);
    }

    @Test
    public void testQuickSort() {
        int[] test = Arrays.copyOf(arr, arr.length), res = Arrays.copyOf(arr, arr.length);
        Arrays.sort(res);
        sort.quickSort(test);
        assertArrayEquals(test, res);
    }

    @Test
    public void testHeapSort() {
        int[] test = Arrays.copyOf(arr, arr.length), res = Arrays.copyOf(arr, arr.length);
        Arrays.sort(res);
        sort.heapSort(test);
        assertArrayEquals(test, res);
    }

}
