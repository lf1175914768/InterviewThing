package com.study.algorithm;

/**
 * 插入排序
 * @author LiuFeng
 *
 */
public class InsertSort {
	
	public void directInsertSort(int[] array) {
		for(int i = 1; i < array.length; i++) {
			int index = i - 1;
			int tmp = array[i];
			while(index >= 0 && array[index] > tmp) {
				array[index + 1] = array[index];
				index--;
			}
			array[index + 1] = tmp;
		}
	}

}
