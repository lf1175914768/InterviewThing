package com.study.leetcode;


import java.util.List;

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

    // -------排序链表 start >>--------

    /**
     * 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
     *
     * 实现自顶向下归并排序
     *
     * 对应 leetcode 中第 148 题
     */
    public ListNode sortList(ListNode head) {
        return sortList(head, null);
    }

    private ListNode sortList(ListNode head, ListNode tail) {
        if (head == null) {
            return head;
        }
        if (head.next == tail) {
            head.next = null;
            return head;
        }
        ListNode slow = head, fast = head;
        while (fast != tail) {
            fast = fast.next;
            slow = slow.next;
            if (fast != tail) {
                fast = fast.next;
            }
        }
        ListNode mid = slow;
        ListNode list1 = sortList(head, mid);
        ListNode list2 = sortList(mid, tail);
        return sortListMerge(list1, list2);
    }

    private ListNode sortListMerge(ListNode head1, ListNode head2) {
        ListNode dummyHead = new ListNode(0);
        ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
        while (temp1 != null && temp2 != null) {
            if (temp1.val <= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        if (temp1 != null) {
            temp.next = temp1;
        } else if (temp2 != null) {
            temp.next = temp2;
        }
        return dummyHead.next;
    }

    // -------排序链表 << end --------

    // -------分隔链表 start >>--------

    /**
     * 给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。
     * 你应当 保留 两个分区中每个节点的初始相对位置。
     *
     * 对应 leetcode 中第 86 题。
     */
    public ListNode partition(ListNode head, int x) {
        ListNode large = new ListNode(0), lNode = large;
        ListNode small = new ListNode(0), sNode = small;
        for (ListNode p = head; p != null; p = p.next) {
            if (p.val < x) {
                sNode.next = p;
                sNode = sNode.next;
            } else {
                lNode.next = p;
                lNode = lNode.next;
            }
        }
        sNode.next = large.next;
        lNode.next = null;
        return small.next;
    }

    // -------分隔链表 << end --------
}
