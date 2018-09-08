package com.study.interview.tree;

import java.util.HashMap;
import java.util.Map;

/**
 * 找到二叉树中符合搜索二叉树条件的最大拓扑结构
 * @author LiuFeng
 *
 */
public class MaxTopoSize {
	
	//-----------------------------------------
	//  比较复杂的第一种方法， O(N的平方)  时间复杂度大
	//-----------------------------------------
	public int bstTopoSize_1(Node head) {
		if(head == null) {
			return 0;
		}
		int max = maxTopo(head, head);
		max = Math.max(bstTopoSize_1(head.left), max);
		max = Math.max(bstTopoSize_1(head.right), max);
		return max;
	}

	public int maxTopo(Node head, Node n) {
		if(head != null && n != null && isBSTNode(head, n, n.value)) {
			return maxTopo(head, n.left) + maxTopo(head, n.right) + 1;
		}
		return 0;
	}

	public boolean isBSTNode(Node head, Node n, int value) {
		if(head == null) {
			return false;
		} 
		if(head == n) {
			return true;
		}
		return isBSTNode(head.value > value ? head.left : head.right, n, value);
	}
	
	//——----------------------------------------------------
	//  时间复杂度大， 采用 拓扑贡献记录的方法, More complicated
	//——----------------------------------------------------
	
	public int bstTopoSize_2(Node head) {
		Map<Node, Record> map = new HashMap<Node, Record>();
		return posOrder(head, map);
	}
	
	public int posOrder(Node head, Map<Node, Record> map) {
		if(head == null) {
			return 0;
		}
		int ls = posOrder(head.left, map);
		int rs = posOrder(head.right, map);
		modifyMap(head.left, head.value, map, true);
		modifyMap(head.right, head.value, map, false);
		Record lr = map.get(head.left);
		Record rr = map.get(head.right);
		int lbst = lr == null ? 0 : lr.l + lr.r + 1;
		int rbst = rr == null ? 0 : rr.l + rr.r + 1;
		map.put(head, new Record(lbst, rbst));
		return Math.max(lbst + rbst + 1, Math.max(ls, rs));
	}

	public int modifyMap(Node n, int v, Map<Node, Record> map, boolean s) {
		if(n == null || !map.containsKey(n)) {
			return 0;
		}
		Record r = map.get(n);
		if((s && n.value > v) || ((!s) && n.value < v)) {
			map.remove(n);
			return r.l + r.r + 1;
		} else {
			int minus = modifyMap(s ? n.right : n.left, v, map, s);
			if(s) {
				r.r = r.r - minus;
			} else {
				r.l = r.l - minus;
			}
			map.put(n, r);
			return minus;
		}
	}

	private class Record {
		int l;
		int r;
		public Record(int left, int right) {
			this.l = left;
			this.r = right;
		}
	}

}
