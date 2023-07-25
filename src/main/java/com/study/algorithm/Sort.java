package com.study.algorithm;

public class Sort {

    /**
     * 冒泡排序
     */
    public void bubbleSort(int[] arr) {
        int len = arr.length;
        for (int i = 0; i < len - 1; i++) {
            for (int j = 0; j < len - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr, j, j + 1);
                }
            }
        }
    }

    public void bubbleSortOptimize(int[] arr) {
        int len = arr.length;
        for (int i = 0; i < len - 1; i++) {
            boolean flag = true;
            for (int j = 0; j < len - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr, j , j + 1);
                    flag = false;
                }
            }
            if (flag)
                break;
        }
    }

    /**
     * 选择排序
     *
     */
    public void selectionSort(int[] arr) {
        int len = arr.length;
        for (int i = 0; i < len - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < len; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            if (minIndex != i) {
                swap(arr, minIndex, i);
            }
        }
    }

    /**
     * 插入排序
     */
    public void insertionSort(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            int j = i, val = arr[i];
            while (j > 0 && arr[j - 1] > val) {
                arr[j] = arr[j - 1];
                j--;
            }
            arr[j] = val;
        }
    }

    /**
     * 希尔排序
     */
    public void shellSort(int[] arr) {
        int len = arr.length;
        for (int gap = len / 2; gap >= 1; gap /= 2) {
            for (int i = gap; i < len; i++) {
                int tmp = arr[i];
                int j = i - gap;
                while (j >= 0 && arr[j] > tmp) {
                    arr[j + gap] = arr[j];
                    j -= gap;
                }
                arr[j + gap] = tmp;
            }
        }
    }

    /**
     * 归并排序
     * 采用的区间时 左闭右闭 区间的格式，所以递归终止的条件时 left >= right，表示只有一个元素时
     * <p>
     * 下面的一个方法采用的时 左闭右开 区间的格式，所以递归终止的条件时 left + 1 >= right
     * @see #mergeSort_v2(int[])
     */
    public void mergeSort(int[] arr) {
        mergeSort(arr, 0, arr.length - 1);
    }

    private void mergeSort(int[] arr, int left, int right) {
        if (left >= right) return;
        int middle = left + ((right - left) >>> 1);
        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);
        merge(arr, left, middle, right);
    }

    private void merge(int[] arr, int left, int middle, int right) {
        int[] tmp = new int[arr.length];
        int i = left, j = middle + 1, index = 0;
        while (i <= middle && j <= right) {
            if (arr[i] < arr[j]) {
                tmp[index++] = arr[i++];
            } else {
                tmp[index++] = arr[j++];
            }
        }
        while (j <= right) {
            tmp[index++] = arr[j++];
        }
        while (i <= middle) {
            tmp[index++] = arr[i++];
        }
        index = 0;
        while (left <= right) {
            arr[left++] = tmp[index++];
        }
    }

    public void mergeSort_v2(int[] arr) {
        mergeSort_v2Dfs(arr, 0, arr.length);
    }

    private void mergeSort_v2Dfs(int[] arr, int start, int end) {
        // 注意这里的 start + 1 >= end, 因为区间时左闭右开，表示只有一个 start 位置的数据时，中断递归
        if (start + 1 >= end) return;
        int middle = start + ((end - start) >>> 1);
        mergeSort_v2Dfs(arr, start, middle);
        mergeSort_v2Dfs(arr, middle, end);
        merge_V2(arr, start, middle, end);
    }

    private void merge_V2(int[] arr, int start, int middle, int end) {
        int[] tmp = new int[arr.length];
        int i = start, j = middle, index = 0;
        while (i < middle && j < end) {
            if (arr[i] < arr[j]) {
                tmp[index++] = arr[i++];
            } else {
                tmp[index++] = arr[j++];
            }
        }
        while (j < end) {
            tmp[index++] = arr[j++];
        }
        while (i < middle) {
            tmp[index++] = arr[i++];
        }
        index = 0;
        while (start < end) {
            arr[start++] = tmp[index++];
        }
    }

    /**
     * 快速排序
     */
    public void quickSort(int[] arr) {
        quickSort(arr, 0, arr.length - 1);
    }

    private void quickSort(int[] arr, int start, int end) {
        if (start >= end) {
            return;
        }
        int pivot = quickSortPartition(arr, start, end);
        quickSort(arr, start, pivot - 1);
        quickSort(arr, pivot + 1, end);
    }

    private int quickSortPartition(int[] arr, int start, int end) {
        int value = arr[start];
        while (start < end) {
            while (start < end && arr[end] >= value) {
                end--;
            }
            arr[start] = arr[end];
            while (start < end && arr[start] <= value) {
                start++;
            }
            arr[end] = arr[start];
        }
        arr[start] = value;
        return start;
    }

    /**
     * 堆排序
     */
    public void heapSort(int[] arr) {
        // 构建大顶堆
        for (int i = arr.length / 2 - 1; i >= 0; i--) {
            // 从第一个非叶子节点从下至上，从右至左调整结构
            adjustHeap(arr, i, arr.length);
        }
        for (int i = arr.length - 1; i > 0; i--) {
            // 将堆顶元素与末尾元素进行交换
            swap(arr, 0, i);
            // 重新对堆进行调整
            adjustHeap(arr, 0, i);
        }
    }

    private void adjustHeap(int[] arr, int i, int size) {
        int tmp = arr[i];
        for (int largest = i * 2 + 1; largest < size; largest = largest * 2 + 1) {
            if (largest + 1 < size && arr[largest] < arr[largest + 1]) {
                largest++;
            }
            if (arr[largest] > tmp) {
                arr[i] = arr[largest];
                i = largest;
            } else {
                break;
            }
        }
        arr[i] = tmp;
    }

    private void swap(int[] arr, int index1, int index2) {
        int temp = arr[index1];
        arr[index1] = arr[index2];
        arr[index2] = temp;
    }
}
