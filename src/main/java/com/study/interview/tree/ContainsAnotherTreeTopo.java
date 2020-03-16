package com.study.interview.tree;

/**
 * 判断 t1 树是否包含 t2 树全部的拓扑结构
 * @author LiuFeng
 *
 */
public class ContainsAnotherTreeTopo {
	
	public boolean contains(Node t1, Node t2) {
		return check(t1, t2) || contains(t1.left, t2) || contains(t1.right, t2);
	}
	
	public boolean check(Node h, Node t2) {
		if(t2 == null) {
			return true;
		}
		if(h == null || h.value != t2.value) {
			return false;
		}
		return check(h.left, t2.left) && check(h.right, t2.right);
	}

}
