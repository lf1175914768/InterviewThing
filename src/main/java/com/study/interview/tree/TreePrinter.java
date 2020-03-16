package com.study.interview.tree;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * 二叉树的按层打印与 zigzag打印 
 * @author LiuFeng
 *
 */
public class TreePrinter {
	
	public void printByLevel_1(Node head) {
		if(head == null) {
			return;
		}
		Queue<Node> queue = new LinkedList<Node>();
		queue.offer(head);
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		List<Integer> listTmp = new ArrayList<Integer>();
		listTmp.add(head.value);
		result.add(listTmp);
 		while(!queue.isEmpty()) {
			Queue<Node> currentLevel = new LinkedList<Node>();
			listTmp = new ArrayList<Integer>();
			while(!queue.isEmpty()) {
				Node current = queue.poll();
				if(current.left != null) {
					currentLevel.offer(current.left);
					listTmp.add(current.left.value);
				}
				if(current.right != null) {
					currentLevel.offer(current.right);
					listTmp.add(current.right.value);
				}
			}
			if(listTmp.size() > 0) result.add(listTmp);
			queue = currentLevel;
		}
 		for(List<Integer> ite : result) {
 			for(Integer it : ite) {
 				System.out.print(it + "  ");
 			}
 			System.out.println();
 		}
	}
	
	/****
	 * 按层打印
	 * @param head
	 */
	public void printByLevel_2(Node head) {
		if(head == null) {
			return;
		}
		Queue<Node> queue = new LinkedList<Node>();
		int level = 1;
		Node last = head;
		Node nLast = null;
		queue.offer(head);
		System.out.print("Level " + (level++) + " : ");
		while(!queue.isEmpty()) {
			head = queue.poll();
			System.out.print(head.value + " ");
			if(head.left != null) {
				queue.offer(head.left);
				nLast = head.left;
			} 
			if(head.right != null) {
				queue.offer(head.right);
				nLast = head.right;
			}
			if(head == last && !queue.isEmpty()) {
				System.out.print("\nLevel " + (level++) + " : ");
				last = nLast;
			}
		}
		System.out.println();
	}
	
	//-----------------------------------------------------
	//  ZigZag 打印
	//-----------------------------------------------------
	
	public void printByZigZag(Node head) {
		if(head == null) {
			return;
		}
		Deque<Node> dp = new LinkedList<Node>();
		int level = 1;
		Node last = head;
		Node nLast = null;
		boolean lr = true;
		dp.offerFirst(head);
		printLevelAndOrientation(level++, lr);
		while(!dp.isEmpty()) {
			if(lr) {
				head = dp.pollFirst();
				if(head.left != null) {
					nLast = nLast == null ? head.left : nLast;
					dp.offerLast(head.left);
				} 
				if(head.right != null) {
					nLast = nLast == null ? head.right : nLast;
					dp.offerLast(head.right);
				}
			} else {
				head = dp.pollLast();
				if(head.right != null) {
					nLast = nLast == null ? head.right : nLast;
					dp.offerFirst(head.right);
				} 
				if(head.left != null) {
					nLast = nLast == null ? head.left : nLast;
					dp.offerFirst(head.left);
				}
			}
			System.out.print(head.value + " ");
			if(head == last && !dp.isEmpty()) { 
				lr = !lr;
				last = nLast;
				nLast = null;
				System.out.println();
				printLevelAndOrientation(level++, lr);
			}
		}
		System.out.println();
	}

	private void printLevelAndOrientation(int level, boolean lr) {
		System.out.print("Level " + level + " from ");
		System.out.print(lr ? "left to right: " : "right to left: ");
	}

}
