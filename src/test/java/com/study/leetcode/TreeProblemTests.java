package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public class TreeProblemTests {

    private TreeProblems problem;

    @Before
    public void init() {
        problem = new TreeProblems();
    }

    @Test
    public void testInvertTree() {
        TreeNode node1 = new TreeNode(1);
        TreeNode node3 = new TreeNode(3);
        TreeNode node2 = new TreeNode(2, node1, node3);

        TreeNode node6 = new TreeNode(6);
        TreeNode node9 = new TreeNode(9);
        TreeNode node7 = new TreeNode(7, node6, node9);

        TreeNode node4 = new TreeNode(4, node2, node7);
        TreeNode root = problem.invertTree(node4);
        assertEquals(root, node4);
        assertEquals(root.left, node7);
        assertEquals(root.right, node2);
        assertEquals(node2.left, node3);
        assertEquals(node2.right, node1);
    }

    @Test
    public void testConnect() {
        TreeProblems.Node node4 = new TreeProblems.Node(4);
        TreeProblems.Node node5 = new TreeProblems.Node(5);
        TreeProblems.Node node2 = new TreeProblems.Node(2, node4, node5);

        TreeProblems.Node node6 = new TreeProblems.Node(6);
        TreeProblems.Node node7 = new TreeProblems.Node(7);
        TreeProblems.Node node3 = new TreeProblems.Node(3, node6, node7);
        TreeProblems.Node node1 = new TreeProblems.Node(1, node2, node3);

        TreeProblems.Node root = problem.connect(node1);
        assertEquals(root.left.next, root.right);
        assertNull(root.next);
        assertEquals(node4.next, node5);
        assertEquals(node5.next, node6);
        assertEquals(node6.next, node7);
        assertNull(node7.next);
    }
}
