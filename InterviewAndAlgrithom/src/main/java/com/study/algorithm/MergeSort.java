package com.study.algorithm;

/**
 * 归并排序
 * @author Liufeng
 * @createData Created on: 2018年9月12日 上午11:27:22
 */
public class MergeSort {

	public void sort(int[] arr, int left, int right, int[] temp) {
		if(left < right) {
			int mid = left + (right - left) / 2;
			sort(arr, left, mid, temp);
			sort(arr, mid + 1, right, temp);
			merge(arr, left, mid, right, temp);
		}
	}

	private void merge(int[] arr, int left, int mid, int right, int[] temp) {
		int i = left, j = mid + 1;  //分别表示左右序列指针。
		int t = 0;
		while(i <= mid && j <= right) {
			if(arr[i] <= arr[j]) {
				temp[t++] = arr[i++];
			} else {
				temp[t++] = arr[j++];
			}
		}
		while(i <= mid) {
			temp[t++] = arr[i++];
		} 
		while(j <= right) {
			temp[t++] = arr[j++];
		}
		t = 0;
		while(left <= right) {
			arr[left++] = temp[t++];
		}
	}
	
}
