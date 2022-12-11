package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

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

    @Test
    public void testPartition() {
        ListNode head = buildCommonNode(Arrays.asList(1, 4, 3, 2, 5, 2));
        int[] rs = new int[] {1,2,2,4,3,5};
        assertArrayEquals(toArray(problem.partition(head, 3)), rs);
    }

    @Test
    public void testRemoveDuplicates() {
        int[] param = new int[] {1,1,2};
        assertEquals(problem.removeDuplicates(param), 2);
        param = new int[] {0,0,1,1,1,2,2,3,3,4};
        assertEquals(problem.removeDuplicates(param), 5);
    }

    @Test
    public void testDeleteDuplicates() {
        ListNode head = buildCommonNode(Arrays.asList(1, 1, 2));
        int[] rs = new int[] {1,2};
        assertArrayEquals(toArray(problem.deleteDuplicates(head)), rs);
    }

    @Test
    public void testRemoveNthFromEnd() {
        ListNode head = buildCommonNode(Arrays.asList(1, 2, 3, 4, 5));
        ListNode resHead = problem.removeNthFromEnd(head, 2);
        int[] res = new int[] {1,2,3,5};
        assertArrayEquals(toArray(resHead), res);
        head = buildCommonNode(Collections.singletonList(1));
        res = new int[0];
        assertArrayEquals(toArray(problem.removeNthFromEnd(head, 1)), res);
        head = buildCommonNode(Arrays.asList(1,2));
        res = new int[] {1};
        assertArrayEquals(toArray(problem.removeNthFromEnd(head, 1)), res);
    }

    @Test
    public void testSwapPairs() {
        ListNode head = buildCommonNode(Arrays.asList(1, 2, 3, 4));
        int[] res = new int[] {2,1,4,3};
        assertArrayEquals(toArray(problem.swapPairs(head)), res);
    }

    @Test
    public void testRemoveElements() {
        ListNode head = buildCommonNode(Arrays.asList(1,2,6,3,4,5,6));
        int[] res = new int[] {1,2,3,4,5};
        assertArrayEquals(toArray(problem.removeElements(head, 6)), res);
        head = null;
        assertNull(problem.removeElements(head, 1));
        head = buildCommonNode(Arrays.asList(7,7,7,7));
        assertNull(problem.removeElements(head, 7));
        head = buildCommonNode(Arrays.asList(1,2,6,3,4,5,6));
        assertArrayEquals(toArray(problem.removeElements_v2(head, 6)), res);
        head = null;
        assertNull(problem.removeElements_v2(head, 1));
        head = buildCommonNode(Arrays.asList(7,7,7,7));
        assertNull(problem.removeElements_v2(head, 7));
    }

    static int[] toArray(ListNode head) {
        List<Integer> rs = new ArrayList<>();
        for (ListNode p = head; p != null; p = p.next) {
            rs.add(p.val);
        }
        int[] tmp = new int[rs.size()];
        for (int i = 0; i < rs.size(); i++) {
            tmp[i] = rs.get(i);
        }
        return tmp;
    }

    static ListNode buildCommonNode(List<Integer> arr) {
        return buildCommonNode(arr.toArray(new Integer[0]));
    }

    static ListNode buildCommonNode(Integer[] arr) {
        ListNode pre = null, cur = null;
        for (int i = arr.length - 1; i >= 0; i--) {
            cur = new ListNode(arr[i], pre);
            pre = cur;
        }
        return cur;
    }
}
