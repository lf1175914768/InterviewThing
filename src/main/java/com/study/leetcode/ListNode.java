package com.study.leetcode;

public class ListNode {
    int val;
    ListNode next;

    @Override
    public String toString() {
        return String.valueOf(val);
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}
