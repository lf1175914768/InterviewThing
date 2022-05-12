package com.study.leetcode;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

/**
 * <p>description: 设计类的题目  </p>
 * <p>className:  DesignProblems </p>
 * <p>create time:  2022/5/12 11:32 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class DesignProblems {


    // -------二叉搜索树迭代器 start >>--------

    /**
     * 实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：
     * BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 的根节点 root 会作为构造函数的一部分给出。
     * 指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。
     * boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则返回 false 。
     * int next()将指针向右移动，然后返回指针处的数字。
     * 注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返回 BST 中的最小元素。
     *
     * 你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序遍历中至少存在一个下一个数字。
     *
     * 对应 leetcode 中第 173 题。
     */
    static final class BSTIterator {
        private TreeNode cur;
        private final Stack<TreeNode> stack;

        public BSTIterator(TreeNode root) {
            stack = new Stack<>();
            cur = root;
        }

        public int next() {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            int ret = cur.val;
            cur = cur.right;
            return ret;
        }

        public boolean hasNext() {
            return cur != null || !stack.isEmpty();
        }
    }

    // -------二叉搜索树迭代器 << end --------

    // -------实现Trie（前缀树） start >>--------

    /**
     * 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。
     *
     * 请你实现 Trie 类：
     * Trie() 初始化前缀树对象。
     * void insert(String word) 向前缀树中插入字符串 word 。
     * boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
     * boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。
     *
     * 对应 leetcode 中第 208 题。
     */
    static final class Trie {

        private final TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        public void insert(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                if (node.next[c - 'a'] == null) {
                    node.next[c - 'a'] = new TrieNode();
                }
                node = node.next[c - 'a'];
            }
            node.isEnd = true;
        }

        public boolean search(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                node = node.next[c - 'a'];
                if (node == null) return false;
            }
            return node.isEnd;
        }

        public boolean startsWith(String prefix) {
            TrieNode node = root;
            for (char c : prefix.toCharArray()) {
                node = node.next[c - 'a'];
                if (node == null) return false;
            }
            return true;
        }

        private static class TrieNode {
            private boolean isEnd;
            TrieNode[] next;
            private TrieNode() {
                isEnd = false;
                next = new TrieNode[26];
            }
        }
    }

    // -------实现Trie（前缀树） << end --------

    // -------添加与搜索单词 - 数据结构设计 start >>--------

    /**
     * 请你设计一个数据结构，支持 添加新单词 和 查找字符串是否与任何先前添加的字符串匹配 。
     *
     * 实现词典类 WordDictionary ：
     *
     * WordDictionary() 初始化词典对象
     * void addWord(word) 将 word 添加到数据结构中，之后可以对它进行匹配
     * bool search(word) 如果数据结构中存在字符串与 word 匹配，则返回 true ；否则，返回  false 。word 中可能包含一些 '.' ，每个 . 都可以表示任何一个字母。
     *
     * 使用 前缀树的方法进行解答。
     *
     * 对应 leetcode 中第 211 题。
     */
    final static class WordDictionary {

        private final Node root;

        public WordDictionary() {
            root = new Node();
        }

        public void addWord(String word) {
            Node node = root;
            for (char ch : word.toCharArray()) {
                if (node.next[ch - 'a'] == null) {
                    node.next[ch - 'a'] = new Node();
                }
                node = node.next[ch - 'a'];
            }
            node.isEnd = true;
        }

        public boolean search(String word) {
            return backTrack(word, 0, root);
        }

        private boolean backTrack(String word, int start, Node cur) {
            if (cur == null) return false;
            if (start == word.length()) {
                return cur.isEnd;
            }
            if (word.charAt(start) == '.') {
                for (int j = 0; j < 26; j++) {
                    if (backTrack(word, start + 1, cur.next[j])) {
                        return true;
                    }
                }
                return false;
            } else {
                return backTrack(word, start + 1, cur.next[word.charAt(start) - 'a']);
            }
        }

        private final static class Node {
            boolean isEnd;
            Node[] next = new Node[26];
        }
    }

    // -------添加与搜索单词 - 数据结构设计 << end --------

    // -------LFU缓存 start >>--------

    /**
     * 实现 LFUCache 类：
     *
     * LFUCache(int capacity) - 用数据结构的容量 capacity 初始化对象
     * int get(int key) - 如果键 key 存在于缓存中，则获取键的值，否则返回 -1 。
     * void put(int key, int value) - 如果键 key 已存在，则变更其值；如果键不存在，请插入键值对。当缓存达到其容量 capacity 时，
     * 则应该在插入新项之前，移除最不经常使用的项。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，应该去除 最近最久未使用 的键。
     * 为了确定最不常使用的键，可以为缓存中的每个键维护一个 使用计数器 。使用计数最小的键是最久未使用的键。
     * 当一个键首次插入到缓存中时，它的使用计数器被设置为 1 (由于 put 操作)。对缓存中的键执行 get 或 put 操作，使用计数器的值将会递增。
     *
     * 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
     *
     * 对应  leetcode 中第 460 题。
     */
    final static class LFUCache {
        private final Map<Integer, Node> cache;
        private final DoublyLinkedList first, last;
        int size;
        private final int capacity;

        public LFUCache(int capacity) {
            cache = new HashMap<>(capacity);
            first = new DoublyLinkedList();
            last = new DoublyLinkedList();
            first.next = last;
            last.prev = first;
            this.capacity = capacity;
        }

        public int get(int key) {
            Node node = cache.get(key);
            if (node == null) return -1;
            // 该 key 访问频次 +1
            freqInc(node);
            return node.value;
        }

        public void put(int key, int value) {
            if (capacity == 0) return;
            Node node = cache.get(key);
            if (node != null) {
                node.value = value;
                freqInc(node);
            } else {
                // 若 key 不存在
                if (size == capacity) {
                    Node removed = last.removeNode(last.prev.tail.prev);
                    cache.remove(removed.key);
                    size--;
                    if (removed.list.head.next == removed.list.tail) {
                        removeList(removed.list);
                    }
                }
                Node newNode = new Node(key, value);
                cache.put(key, newNode);
                if (last.prev.freq != 1) {
                    DoublyLinkedList newList = new DoublyLinkedList(1);
                    addList(newList, last.prev);
                }
                last.prev.addNode(newNode);
            }
            size++;
        }

        private void freqInc(Node node) {
            // 将 node 从原freq对应的双向链表里移除，如果链表空了则删除链表
            DoublyLinkedList list = node.list;
            DoublyLinkedList prev = list.prev;
            list.removeNode(node);
            if (list.head.next == list.tail) {
                removeList(list);
            }
            // 将node加入新 freq 对应的双向链表，若该链表不存在，则先创建该链表
            node.freq++;
            if (prev.freq != node.freq) {
                DoublyLinkedList newList = new DoublyLinkedList(node.freq);
                addList(newList, prev);
                newList.addNode(node);
            } else {
                prev.addNode(node);
            }
        }

        private void removeList(DoublyLinkedList list) {
            list.prev.next = list.next;
            list.next.prev = list.prev;
        }

        private void addList(DoublyLinkedList newList, DoublyLinkedList prev) {
            newList.next = prev.next;
            newList.next.prev = newList;
            newList.prev = prev;
            prev.next = newList;
        }

        static final class Node {
            int key, value, freq = 1;
            Node prev, next;
            DoublyLinkedList list;     // 该Node所在频次的双向链表
            private Node() {}

            private Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }

        static final class DoublyLinkedList {
            int freq;  // 该双向链表表示的频次
            DoublyLinkedList prev;    // 该双向链表的前缀链表 (prev.freq < this.freq)
            DoublyLinkedList next;    // 该双线链表的后继链表 (next.freq > this.freq)
            Node head, tail;

            private DoublyLinkedList() {
                this(0);
            }

            private DoublyLinkedList(int freq) {
                head = new Node();
                tail = new Node();
                tail.prev = head;
                head.next = tail;
                this.freq = freq;
            }

            private Node removeNode(Node node) {
                node.prev.next = node.next;
                node.next.prev = node.prev;
                return node;
            }

            private void addNode(Node node) {
                node.next = head.next;
                node.prev = head;
                head.next.prev = node;
                head.next = node;
                node.list = this;
            }
        }
    }

    // -------LFU缓存 << end --------

}
