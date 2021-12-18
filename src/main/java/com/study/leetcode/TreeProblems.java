package com.study.leetcode;

import java.util.List;

public class TreeProblems {

    // -------反转二叉树 start >>--------

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        // 前序遍历
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    // -------反转二叉树 << end --------

    // -------填充每一个节点的下一个右侧节点指针 start >>--------

    public Node connect(Node root) {
        if (root == null)
            return null;
        connectTwoNode(root.left, root.right);
        return root;
    }

    private void connectTwoNode(Node left, Node right) {
        if (left == null || right == null)
            return;
        left.next = right;
        // 链接相同父节点的两个子节点
        connectTwoNode(left.left, left.right);
        connectTwoNode(right.left, right.right);
        // 连接跨越父节点的两个子节点
        connectTwoNode(left.right, right.left);
    }


    // -------填充每一个节点的下一个右侧节点指针 << end --------


    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------

    ///////-------------helper class-------------------

    static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int val, Node left, Node right) {
            this(val, left, right, null);
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }


}
