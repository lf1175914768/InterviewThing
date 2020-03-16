package com.study.interview.chain;

import java.util.HashSet;

/**
 * 删除无序单链表中值重复出现的节点
 * @author LiuFeng
 *
 */
public class DeleteRepeatNodes {
	
	public void removeRep_1(Node head) {
		if(head == null) {
			return ;
		}
		HashSet<Integer> set = new HashSet<Integer>();
		Node pre = head;
		Node cur = head.next;
		set.add(head.value);
		while(cur != null) {
			if(set.contains(cur)) {
				pre.next = cur.next;
			} else {
				set.add(cur.value);
				pre = cur;
			}
			cur = cur.next;
		}
	}

}
