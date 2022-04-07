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

    /**
     *  给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。
     *  请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
     *
     * 解题思路：
     * 1.我们定义两个指针，分别称之为 g 和 p。
     * 我们首先根据方法的参数 m 确定 g 和 p的位置。将 g 移动到第一个要反转的节点的前面，将 p 移动到第一个要
     * 反转的节点的位置上。我们以 m = 2，n = 4 为例。
     * 2.将 p 后面的元素删除，然后添加到 g 的后面，也即头插法。
     * 3.根据 m 和 n，重复步骤2
     * 4.返回dummy.next
     *
     * 对应 leetcode 中第 92 题。
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        ListNode q = dummyHead, p = dummyHead.next;
        for (int i = 0; i < left - 1; i++) {
            q = q.next;
            p = p.next;
        }

        // 采用头插法
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

    // -------删除有序数组中的重复项 start >>--------

    /**
     * 给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。
     * 由于在某些语言中不能改变数组的长度，所以必须将结果放在数组nums的第一部分。更规范地说，如果在删除重复项之后有 k 个元素，那么 nums 的前 k 个元素应该保存最终结果。
     * 将最终结果插入 nums 的前 k 个位置后返回 k 。
     * 不要使用额外的空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
     *
     * 对应 leetcode 中第 26 题。
     */
    public int removeDuplicates(int[] nums) {
        int start = 0, end = 1;
        while (end < nums.length) {
            if (nums[end] != nums[start]) {
                start++;
                nums[start] = nums[end];
            }
            end++;
        }
        return start + 1;
    }

    // -------删除有序数组中的重复项 << end --------

    // -------删除排序链表中的重复元素 start >>--------

    /**
     * 给定一个已排序的链表的头 head ， 删除所有重复的元素，使每个元素只出现一次 。返回 已排序的链表 。
     *
     * 对应  leetcode 中第 83 题。
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;
        ListNode slow = head, fast = head;
        while (fast != null) {
            if (fast.val != slow.val) {
                slow.next = fast;
                slow = slow.next;
            }
            fast = fast.next;
        }
        // 断开与后面的连接
        slow.next = null;
        return head;
    }

    // -------删除排序链表中的重复元素 << end --------

}
