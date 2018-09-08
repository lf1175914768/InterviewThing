package com.study.algorithm;

/**
 * 选择排序， 也很简单， 
 * @author LiuFeng
 *
 */
public class SelectSort {
	
	public void selectSort(int[] array) {
		for(int i = 0; i < array.length - 1; i++) {
			for(int j = i + 1; i < array.length; j++) {
				if(array[j] < array[i]) {
					int tmp = array[i];
					array[i] = array[j];
					array[j] = tmp;
				}
			}
		}
	}

}
