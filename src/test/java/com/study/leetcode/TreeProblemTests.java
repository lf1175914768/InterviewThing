package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

public class TreeProblemTests {

    private TreeProblems problem;

    @Before
    public void init() {
        problem = new TreeProblems();
    }

    @Test
    public void testInvertTree() {
        TreeNode node = buildCommonTree();
        TreeNode left = node.left;
        TreeNode root = problem.invertTree(node);
        assertEquals(root.val, 4);
        assertEquals(root.left.val, 7);
        assertEquals(root.right.val, 2);
        assertEquals(left.left.val, 3);
        assertEquals(left.right.val, 1);
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

    @Test
    public void testFlatten() {
        TreeNode node = buildCommonTree();
        problem.flatten(node);
        assertNull(node.left);
        assertEquals(node.right.val, 2);
        assertNull(node.right.left);
        assertEquals(node.right.right.val, 1);
        assertEquals(node.right.right.right.val, 3);
        assertEquals(node.right.right.right.right.val, 7);
        assertEquals(node.right.right.right.right.right.val, 6);
        assertEquals(node.right.right.right.right.right.right.val, 9);
        assertNull(node.right.right.right.right.right.left);

        node = buildCommonTree();
        problem.flatten_v2(node);
        assertNull(node.left);
        assertEquals(node.right.val, 2);
        assertNull(node.right.left);
        assertEquals(node.right.right.val, 1);
        assertEquals(node.right.right.right.val, 3);
        assertEquals(node.right.right.right.right.val, 7);
        assertEquals(node.right.right.right.right.right.val, 6);
        assertEquals(node.right.right.right.right.right.right.val, 9);
        assertNull(node.right.right.right.right.right.left);

        node = buildCommonTree();
        problem.flatten_v3(node);
        assertNull(node.left);
        assertEquals(node.right.val, 2);
        assertNull(node.right.left);
        assertEquals(node.right.right.val, 1);
        assertEquals(node.right.right.right.val, 3);
        assertEquals(node.right.right.right.right.val, 7);
        assertEquals(node.right.right.right.right.right.val, 6);
        assertEquals(node.right.right.right.right.right.right.val, 9);
        assertNull(node.right.right.right.right.right.left);
    }

    @Test
    public void testConstructMaximumBinaryTree() {
        int[] nums = new int[]{3, 2, 1, 6, 0, 5};
        TreeNode root = problem.constructMaximumBinaryTree(nums);
        assertEquals(root.val, 6);
        assertEquals(root.left.val, 3);
        assertEquals(root.right.val, 5);
        assertEquals(root.left.right.val, 2);
        assertNull(root.left.left);
        assertNull(root.left.right.left);
        assertEquals(root.left.right.right.val, 1);
        assertEquals(root.right.left.val, 0);
        assertNull(root.right.right);
    }

    @Test
    public void testBuildTree() {
        int[] preOrder = new int[]{3, 9, 20, 15, 7};
        int[] inOrder = new int[]{9, 3, 15, 20, 7};
        TreeNode root = problem.buildTree(preOrder, inOrder);
        assertEquals(root.val, 3);
        assertEquals(root.left.val, 9);
        assertEquals(root.right.val, 20);
        assertNull(root.left.left);
        assertNull(root.left.right);
        assertEquals(root.right.left.val, 15);
        assertEquals(root.right.right.val, 7);

        TreeNode node = problem.buildTree_v2(preOrder, inOrder);
        assertEquals(node.val, 3);
        assertEquals(node.left.val, 9);
        assertEquals(node.right.val, 20);
        assertNull(node.left.left);
        assertNull(node.left.right);
        assertEquals(node.right.left.val, 15);
        assertEquals(node.right.right.val, 7);
    }

    @Test
    public void testBuildTreeByPost() {
        int[] inorder = new int[]{9,3,15,20,7};
        int[] postOrder = new int[]{9,15,7,20,3};
        TreeNode root = problem.buildTreeByPost(inorder, postOrder);
        assertEquals(root.val, 3);
        assertEquals(root.left.val, 9);
        assertNull(root.left.left);
        assertNull(root.left.right);
        assertEquals(root.right.val, 20);
        assertEquals(root.right.left.val, 15);
        assertEquals(root.right.right.val, 7);
    }

    @Test
    public void testKthSmallest() {
        TreeNode node2 = new TreeNode(2);
        TreeNode node1 = new TreeNode(1, null, node2);
        TreeNode node4 = new TreeNode(4);
        TreeNode node3 = new TreeNode(3, node1, node4);

        assertEquals(problem.kthSmallest(node3, 1), 1);
        assertEquals(problem.kthSmallest(node3, 2), 2);
        assertEquals(problem.kthSmallest(node3, 3), 3);
        assertEquals(problem.kthSmallest(node3, 4), 4);
    }

    @Test
    public void testConvertBST() {
        TreeNode node8 = new TreeNode(8);
        TreeNode node7 = new TreeNode(7, null, node8);
        TreeNode node5 = new TreeNode(5);
        TreeNode node6 = new TreeNode(6, node5, node7);

        TreeNode node0 = new TreeNode(0);
        TreeNode node3 = new TreeNode(3);
        TreeNode node2 = new TreeNode(2, null, node3);
        TreeNode node1 = new TreeNode(1, node0, node2);

        TreeNode node4 = new TreeNode(4, node1, node6);
        TreeNode root = problem.convertBST(node4);
        assertEquals(root.val, 30);
        assertEquals(root.left.val, 36);
        assertEquals(root.left.left.val, 36);
        assertEquals(root.left.right.val, 35);
        assertEquals(root.left.right.right.val, 33);
        assertEquals(root.right.val, 21);
        assertEquals(root.right.left.val, 26);
        assertEquals(root.right.right.val, 15);
    }

    @Test
    public void testisValidBSTTree() {
        TreeNode node1 = new TreeNode(1);
        TreeNode node3 = new TreeNode(3);
        TreeNode node2 = new TreeNode(2, node1, node3);

        TreeNode node6 = new TreeNode(6);
        TreeNode node4 = new TreeNode(4, node3, node6);
        TreeNode node5 = new TreeNode(5, node1, node4);

        assertTrue(problem.isValidBST(node2));
        assertFalse(problem.isValidBST(node5));
    }

    @Test
    public void testInsertIntoBST() {
        TreeNode node1 = new TreeNode(1);
        TreeNode node3 = new TreeNode(3);
        TreeNode node2 = new TreeNode(2, node1, node3);

        TreeNode node7 = new TreeNode(7);
        TreeNode node4 = new TreeNode(4, node2, node7);
        assertEquals(problem.insertIntoBST(node4, 6).right.left.val, 6);
    }

    @Test
    public void deleteNode() {
        TreeNode node2 = new TreeNode(2);
        TreeNode node4 = new TreeNode(4);
        TreeNode node3 = new TreeNode(3, node2, node4);

        TreeNode node7 = new TreeNode(7);
        TreeNode node6 = new TreeNode(6, null, node7);
        TreeNode node5 = new TreeNode(5, node3, node6);
        TreeNode root = problem.deleteNode(node5, 3);
        assertEquals(root.left.val, 4);
        assertEquals(root.left.left.val, 2);
        assertNull(root.left.right);
    }

    @Test
    public void testGenerateTrees() {
        List<TreeNode> treeNodes = problem.generateTrees(1);
        assertEquals(treeNodes.size(), 1);
        List<TreeNode> treeNodes2 = problem.generateTrees(2);
        assertEquals(treeNodes2.size(), 2);
        assertEquals(problem.generateTrees(3).size(), 5);
        assertEquals(problem.generateTrees_v2(1).size(), 1);
        assertEquals(problem.generateTrees_v2(2).size(), 2);
        assertEquals(problem.generateTrees_v2(3).size(), 5);
    }

    @Test
    public void testInOrderTraversal() {
        TreeNode root = buildCommonTree();
        List<Integer> list = problem.inorderTraversal(root);
        assertEquals(list.get(0).intValue(), 1);
        assertEquals(list.get(1).intValue(), 2);
        assertEquals(list.get(2).intValue(), 3);
        assertEquals(list.get(3).intValue(), 4);
        assertEquals(list.get(4).intValue(), 6);
        assertEquals(list.get(5).intValue(), 7);
        assertEquals(list.get(6).intValue(), 9);

        List<Integer> result = problem.inorderTraversal(null);
        assertEquals(result.size(), 0);
    }

    @Test
    public void testRecoverTree() {
        TreeNode node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(3, null, node2);
        TreeNode node1 = new TreeNode(1, node3, null);

        problem.recoverTree(node1);
        assertEquals(node1.val, 3);
        assertEquals(node2.val, 2);
        assertEquals(node3.val, 1);
    }

    @Test
    public void testIsSameTree() {
        TreeNode node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(3);
        TreeNode node1 = new TreeNode(1, node2, node3);

        TreeNode node21 = new TreeNode(2);
        TreeNode node31 = new TreeNode(3);
        TreeNode node11 = new TreeNode(1, node21, node31);
        assertTrue(problem.isSameTree(node11, node1));

        node1 = new TreeNode(1, node2, null);
        node11 = new TreeNode(1, null, node21);
        assertFalse(problem.isSameTree(node11, node1));

        node3 = new TreeNode(1);
        node1 = new TreeNode(1, node2, node3);

        node31 = new TreeNode(1);
        node11 = new TreeNode(1, node31, node21);
        assertFalse(problem.isSameTree(node1, node11));
    }

    @Test
    public void testIsSymmetric() {
        TreeNode node3 = new TreeNode(3);
        TreeNode node4 = new TreeNode(4);
        TreeNode node2 = new TreeNode(2, node3, node4);

        TreeNode node31 = new TreeNode(3);
        TreeNode node41 = new TreeNode(4);
        TreeNode node21 = new TreeNode(2, node41, node31);

        TreeNode node1 = new TreeNode(1, node2, node21);
        assertTrue(problem.isSymmetric(node1));
        assertTrue(problem.isSymmetric_v2(node1));
    }

    @Test
    public void testLevelOrder() {
        TreeNode root = buildCommonTree();
        List<List<Integer>> result = problem.levelOrder(root);
        assertEquals(result.size(), 3);
        assertEquals(result.get(0).size(), 1);
        assertEquals(result.get(1).size(), 2);
        assertEquals(result.get(2).size(), 4);
        assertEquals(result.get(0).get(0).intValue(), 4);
        assertEquals(result.get(1).get(0).intValue(), 2);
        assertEquals(result.get(1).get(1).intValue(), 7);
        assertEquals(result.get(2).get(0).intValue(), 1);
        assertEquals(result.get(2).get(1).intValue(), 3);
        assertEquals(result.get(2).get(2).intValue(), 6);
        assertEquals(result.get(2).get(3).intValue(), 9);
    }

    @Test
    public void testHasPathSum() {
        TreeNode node7 = new TreeNode(7);
        TreeNode node2 = new TreeNode(2);
        TreeNode node11 = new TreeNode(11, node7, node2);
        TreeNode node4_v2 = new TreeNode(4, node11, null);

        TreeNode node1 = new TreeNode(1);
        TreeNode node4 = new TreeNode(4, null, node1);
        TreeNode node13 = new TreeNode(13);
        TreeNode node8 = new TreeNode(8, node13, node4);
        TreeNode node5 = new TreeNode(5, node4_v2, node8);
        assertTrue(problem.hasPathSum(node5, 22));

        node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(3);
        node1 = new TreeNode(1, node2, node3);
        assertFalse(problem.hasPathSum(node1, 5));

        assertFalse(problem.hasPathSum(null, 0));
    }

    @Test
    public void testPathSum() {
        TreeNode node7 = new TreeNode(7);
        TreeNode node2 = new TreeNode(2);
        TreeNode node11 = new TreeNode(11, node7, node2);
        TreeNode node4_v2 = new TreeNode(4, node11, null);

        TreeNode node1 = new TreeNode(5);
        TreeNode node4 = new TreeNode(4, null, node1);
        TreeNode node13 = new TreeNode(13);
        TreeNode node8 = new TreeNode(8, node13, node4);
        TreeNode node5 = new TreeNode(5, node4_v2, node8);
        List<List<Integer>> lists = problem.pathSum(node5, 22);
        assertEquals(lists.size(), 2);
        List<List<Integer>> lists2 = problem.pathSum_v2(node5, 22);
        assertEquals(lists2.size(), 2);
    }

    /**
     * build tree
     *             4
     *         2      7
     *      1   3  6     9
     *
     * @return root node
     */
    private TreeNode buildCommonTree() {
        TreeNode node1 = new TreeNode(1);
        TreeNode node3 = new TreeNode(3);
        TreeNode node2 = new TreeNode(2, node1, node3);

        TreeNode node6 = new TreeNode(6);
        TreeNode node9 = new TreeNode(9);
        TreeNode node7 = new TreeNode(7, node6, node9);

        return new TreeNode(4, node2, node7);
    }

}