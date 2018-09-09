package com.study.algorithm;

public class QuickSort {
	
	public void sort(int[] array, int first, int last) {
		if(first >= last) {
			return;
		}
		int boundary = doSort(array, first, last);
		sort(array, first, boundary - 1);
		sort(array, boundary + 1, last);
	}
	
	public int doSort(int[] a, int first, int last) {
		int standard = a[first];
		int i = first, j = last;
		while(i < j) {
			while(i < j && a[j] >= standard) {  // 从后往前找小于基准数的位置
				j--;
			}
			a[i] = a[j];
			while(i < j && a[i] <= standard) {  // 从前往后找大于基准数的位置
				i++;
			}
			a[j] = a[i];
		}
		a[i] = standard;
		return i;
	}

}
