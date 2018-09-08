package com.study.algorithm;

/**
 * 冒泡排序， 太简单， 没什么可说的
 * 
 * @author LiuFeng
 *
 */
public class BubbleSort {
	
	/**
	 * 冒泡排序
	 * @param array
	 */
	public void bubbleSort(int[] array) {
		for(int i = 1; i < array.length; i++) {
			for(int j = 0; j < array.length - i; j++) {
				if(array[j] > array[j + 1]) {
					int tmp = array[j];
					array[j] = array[j + 1];
					array[j + 1] = tmp;
				}
			}
		}
	}

}
