package com.study.algorithm;

/**
 * 判断字符数组中是否所有的字符都只出现过一次， 
 * 这里需要用到堆排序算法，非递归实现
 * @author LiuFeng
 *
 */
public class HeapSort {
	
	public boolean isUnique2(char[] chas) {
		if(chas == null) {
			return true;
		}
		heapSort_1(chas);
		for(int i = 1; i < chas.length; i++) {
			if(chas[i] == chas[i - 1]) {
				return false;
			}
		}
		return true;
	}

	public void heapSort_1(char[] chas) {
		for(int i = 0; i < chas.length; i++) {
			heapInsert(chas, i);
		}
		for(int i = chas.length - 1; i > 0; i--) {
			swap(chas, 0, i);
			heapify(chas, 0, i);
		}
	}
	
	public void heapSort_2(char[] chas) {
		// Build the heap.
		for(int i = chas.length / 2 - 1; i >= 0; i--) {
			heapify(chas, i, chas.length);
		}
		// swap the first element and the i element, then rebuild.
		for(int i = chas.length - 1; i > 0; i--) {
			swap(chas, 0, i);
			heapify(chas, 0, i);
		}
	}

	public void heapInsert(char[] chas, int i) {
		int parent = 0;
		while(i != 0) {
			parent = (i - 1) / 2;
			if(chas[parent] < chas[i]) {
				swap(chas, parent, i);
				i = parent;
			} else {
				break;
			}
		}
	}

	public void heapify(char[] chas, int i, int size) {
		int left = i * 2 + 1;
		int right = i * 2 + 2;
		int largest = i;
		while(left < size) {
			if(chas[left] > chas[i]) {
				largest = left;
			}
			if(right < size && chas[right] > chas[largest]) {
				largest = right;
			}
			if(largest != i) {
				swap(chas, largest, i);
			} else {
				break;
			}
			i = largest;
			left = i * 2 + 1;
			right = i * 2 + 2;
		}
	}

	private void swap(char[] chas, int i, int i2) {
		char tmp = chas[i];
		chas[i] = chas[i2];
		chas[i2] = tmp;
	}

}
