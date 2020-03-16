package com.study.interview.tree;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Created by Liufeng on 2020/1/12.
 */
public class AhoCorasickAutomation {

    /**
     * AC自动机 是由 Trie树 和 类似于 KMP 算法组合而成的
     */

    /**
     * 这里的AC自动机只处理英文类型的字符串，所以数组的长度是128
     */
    private static final int ASCII = 128;

    /**
     *  AC自动机的根节点，根节点不存储任何字符串
     */
    private Node root;

    /**
     * 带查找的目标字符串集合
     */
    private List<String> target;

    public AhoCorasickAutomation(List<String> target) {
        root = new Node();
        this.target = target;
        buildTrieTree();
        buildACFromTrie();
    }

    /**
     * 在文本串中查找所有的目标字符串
     * @param text
     * @return
     */
    public Map<String, List<Integer>> find(String text) {
        Map<String, List<Integer>> result = new HashMap<>();

        for(String s : target) {
            result.put(s, new LinkedList<>());
        }

        Node curr = root;
        int i = 0, length = text.length();
        while(i < length) {
            // 文本串中的字符
            char ch = text.charAt(i);

            if(curr.table[ch] != null) {
                // 若文本串中字符和AC自动机中的字符相等，自动机进入下一状态
                curr = curr.table[ch];

                if(curr.isWord()) {
                    // 存放这一个 word 在 text中出现的位置
                    result.get(curr.str).add(i - curr.str.length() + 1);
                }

                /**
                 * 这里很容易被忽视，因为一个目标串中的中间某部分字符串可能
                 * 正好是另一个完整的字符串，即使当前节点不表示一个目标字符串的
                 * 终点，但到当前节点为止，可能恰好包含了一个字符串
                 */
                if(curr.fail != null && curr.fail.isWord()) {
                    result.get(curr.fail.str).add(i - curr.fail.str.length() + 1);
                }

                i++;
            } else {
                // 若不相等，找到下一个应该比较的状态
                curr = curr.fail;

                /**
                 * 若到根节点还没有找到，说明文本串中以ch作为结束的字符片段
                 * 不是任何目标字符串的 prefix， 所以状态机重置，比较下一个字符
                 */
                if(curr == null) {
                    curr = root;
                    i++;
                }
            }
        }
        return result;
    }

    /**
     * 由目标字符串构建Trie树
     */
    private void buildTrieTree() {
        for(String targetStr : this.target) {
            Node curr = root ;
            for(int i = 0; i < targetStr.length(); i++) {
                char ch = targetStr.charAt(i);
                if(curr.table[ch] == null) {
                    curr.table[ch] = new Node();
                }
                curr = curr.table[ch];
            }
            // 将每个目标字符串的最后一个字符对应的节点变成终点
            curr.str = targetStr;
        }
    }

    /**
     * 由Trie树构建一个Ac自动机，本质是一个自动机，相当于构建KMP算法的next数组
     */
    private void buildACFromTrie() {

        // 广度优先遍历所使用的队列
        LinkedList<Node> queue = new LinkedList<>();

        for (Node x : root.table) {
            if(x != null) {
                // 根节点的所有孩子节点的fail都指向根节点
                x.fail = root;
                // 所有根节点的孩子节点入列
                queue.addLast(x);
            }
        }

        while(!queue.isEmpty()) {
            // 确定出列节点的所有孩子节点的fail的指向
            Node p = queue.removeFirst();
            for(int i = 0; i < p.table.length; i++) {
                if(p.table[i] != null) {
                    // 孩子节点入列
                    queue.addLast(p.table[i]);
                    // 从 p.fail 开始找起
                    Node failTo = p.fail;
                    while(true) {
                        // 说明找到了根节点还没有找到
                        if(failTo == null) {
                            p.table[i].fail = root;
                            break;
                        }

                        // 说明有公共的前缀，并且在这里假如 failTo 的孩子节点如果
                        // 有和 p.table[i] 相同的话，那么肯定是存储在 第i个位置的
                        if(failTo.table[i] != null) {
                            p.table[i].fail = failTo.table[i];
                            break;
                        } else {
                            // 继续向上查找
                            failTo = failTo.fail;
                        }
                    }
                }
            }
        }
    }

    /**
     * 内部静态类，用于表示AC自动机的每个节点，
     * key表示字符串，value表示目标字符串在文本串中出现的位置
     */
    private static class Node {
        String str;

        // fail 指针，表示当前节点的孩子节点不能匹配文本串中的某个字符串时，下一个应该查找的节点
        Node fail;

        Node[] table = new Node[ASCII];

        boolean isWord() {
            return str != null;
        }
    }
}
