package com.study.interview.chain;

/**
 * 将单向链表按某值划分成左边小、 中间相等、 右边大的形式。
 * 分为普通解法和高阶解法
 * @author LiuFeng
 *
 */
public class SeparateChain {
	
	/**
	 * 方法一： 
	 * @param head
	 * @param pivot
	 * @return
	 */
	public Node listPartition1(Node head, int pivot) {
		if(head == null) {
			return head;
		}
		Node cur = head;
		int i = 0;
		while(cur != null) {
			i++;
			cur = cur.next;
		}
		Node[] nodeArr = new Node[i];
		i = 0;
		cur = head;
		for(i = 0; i < nodeArr.length; i++) {
			nodeArr[i] = cur;
			cur = cur.next;
		}
		arrPartition(nodeArr, pivot);
		for(i = 1; i < nodeArr.length; i++) {
			nodeArr[i - 1].next = nodeArr[i];
		}
		nodeArr[i - 1].next = null;
		return nodeArr[0];
	}

	/**
	 * 快速排序的改进版
	 * @param nodeArr
	 * @param pivot
	 */
	public void arrPartition(Node[] nodeArr, int pivot) {
		int small = -1;
		int big = nodeArr.length;
		int index = 0;
		while(index != big) {
			if(nodeArr[index].value < pivot) {
				swap(nodeArr, ++small, index++);
			} else if(nodeArr[index].value == pivot) {
				index++;
			} else {
				swap(nodeArr, --big, index);
			}
		}
	}

	private void swap(Node[] nodeArr, int i, int index) {
		Node tmp = nodeArr[i];
		nodeArr[i] = nodeArr[index];
		nodeArr[index] = tmp;
	}
	
	/**
	 * 方法二 : 处了在原要求的基础上， 还增加了三个序列中的顺序与原始顺序一样。
	 * @param head
	 * @param pivot
	 * @return
	 */
	public Node listPartition2(Node head, int pivot) {
		Node sH = null;  //小的头
		Node sT = null; // 小的尾
		Node eH = null;   // 相等的头
		Node eT = null;   //相等的尾
		Node bH = null;   // 大的头
		Node bT = null;   // 大的尾
		Node next = null;    // 保存下一个节点
		// 所有的节点分进三个链表中
		while(head != null) {
			next = head.next;
			head.next = null;
			if(head.value < pivot) {
				if(sH == null) {
					sH = head;
					sT = head;
				} else {
					sT.next = head;
					sT = head;
				}
			} else if(head.value == pivot) {
				if(eH == null) {
					eH = head;
					eT = head;
				} else {
					eT.next = head;
					eT = head;
				}
			} else {
				if(bH == null) {
					bH = head;
					bT = head;
				} else {
					bT.next = head;
					bT = head;
				}
			}
			head = next;
		}
		
		// 小的和相等的重新连接
		if(sT != null) {
			sT.next = eH;
			eT = eT == null ? sT : eT;
		}
		// 所有的重新连接
		if(eT != null) {
			eT.next = bH;
		}
		return sH != null ? sH : eH != null ? eH : bH;
	}

}
