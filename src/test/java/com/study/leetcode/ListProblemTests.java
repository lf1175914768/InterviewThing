package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ListProblemTests {

    private ListProblem problem;

    @Before
    public void init() {
        problem = new ListProblem();
    }

    @Test
    public void testReverseList() {
        ListNode node1 = new ListNode(5);
        ListNode node2 = new ListNode(4, node1);
        ListNode node3 = new ListNode(3, node2);
        ListNode node4 = new ListNode(2, node3);
        ListNode node5 = new ListNode(1, node4);
        ListNode head = problem.reverseList(node5);
        assertEquals(head, node1);
        assertEquals(head.next, node2);
        assertEquals(node2.next, node3);
        head = problem.reverseList_v2(node1);
        assertEquals(head, node5);
        assertEquals(node5.next, node4);
    }

    @Test
    public void testReverseBetween() {
        ListNode node1 = new ListNode(5);
        ListNode node2 = new ListNode(4, node1);
        ListNode node3 = new ListNode(3, node2);
        ListNode node4 = new ListNode(2, node3);
        ListNode node5 = new ListNode(1, node4);
        ListNode head = problem.reverseBetween(node5, 2, 4);
        assertEquals(head, node5);
        assertEquals(node5.next, node2);
        assertEquals(node2.next, node3);
        assertEquals(node3.next, node4);
        assertEquals(node4.next, node1);
    }

    @Test
    public void testReverseKGroup() {
        ListNode node1 = new ListNode(5);
        ListNode node2 = new ListNode(4, node1);
        ListNode node3 = new ListNode(3, node2);
        ListNode node4 = new ListNode(2, node3);
        ListNode node5 = new ListNode(1, node4);
        ListNode head = problem.reverseKGroup(node5, 2);
        assertEquals(head, node4);
        assertEquals(node4.next, node5);
        assertEquals(node5.next, node2);
        assertEquals(node2.next, node3);
        assertEquals(node3.next, node1);
        head = problem.reverseKGroup(node4, 3);
        assertEquals(head, node2);
        assertEquals(node2.next, node5);
        assertEquals(node5.next, node4);
        assertEquals(node4.next, node3);
        assertEquals(node3.next, node1);
    }
}
