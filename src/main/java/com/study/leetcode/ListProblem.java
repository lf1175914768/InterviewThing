package com.study.leetcode;


public class ListProblem {

    // -------反转链表 start >>--------

    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode last = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return last;
    }

    public ListNode reverseList_v2(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode cur = head, next, pre = null;
        while (cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    // -------反转链表 << end --------

    // -------反转链表II start >>--------

    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        ListNode q = dummyHead, p = dummyHead.next;
        for (int i = 0; i < left - 1; i++) {
            q = q.next;
            p = p.next;
        }

        for (int i = 0; i < right - left; i++) {
            ListNode removed = p.next;
            p.next = p.next.next;

            removed.next = q.next;
            q.next = removed;
        }
        return dummyHead.next;
    }

    // -------反转链表II << end --------

    // -------K个一组翻转链表 start >>--------

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        // 区间 [head, b) 包含 k个待反转元素
        ListNode b = head;
        for (int i = 0; i < k; i++) {
            // 不足k个，不需要反转， base case
            if (b == null)
                return head;
            b = b.next;
        }
        ListNode newNode = reverseKGroup_doReverse(head, b);
        head.next = reverseKGroup(b, k);
        return newNode;
    }

    private ListNode reverseKGroup_doReverse(ListNode start, ListNode end) {
        ListNode pre = null, cur = start, next;
        while (cur != end) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    // -------K个一组翻转链表 << end --------


}
