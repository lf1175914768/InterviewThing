package com.study.leetcode;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * <p>description:   </p>
 * <p>className:  DesignProblemTests </p>
 * <p>create time:  2022/5/12 11:33 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class DesignProblemTests {


    @Test
    public void testBSTIterator() {
        TreeNode root = TreeProblemTests.buildCommonTree();
        DesignProblems.BSTIterator iterator = new DesignProblems.BSTIterator(root);
        assertEquals(iterator.next(), 1);
        assertEquals(iterator.next(), 2);
        assertEquals(iterator.next(), 3);
        assertTrue(iterator.hasNext());
        assertEquals(iterator.next(), 4);
        assertEquals(iterator.next(), 6);
        assertEquals(iterator.next(), 7);
        assertTrue(iterator.hasNext());
        assertEquals(iterator.next(), 9);
        assertFalse(iterator.hasNext());
        assertFalse(iterator.hasNext());
    }

    @Test
    public void testTrie() {
        DesignProblems.Trie trie = new DesignProblems.Trie();
        trie.insert("apply");
        assertTrue(trie.search("apply"));
        assertFalse(trie.search("app"));
        assertTrue(trie.startsWith("app"));
        trie.insert("app");
        assertTrue(trie.search("app"));
    }

    @Test
    public void testWordDictionary() {
        DesignProblems.WordDictionary dictionary = new DesignProblems.WordDictionary();
        dictionary.addWord("bad");
        dictionary.addWord("dad");
        dictionary.addWord("mad");
        assertFalse(dictionary.search("pad"));
        assertTrue(dictionary.search("bad"));
        assertTrue(dictionary.search(".ad"));
        assertTrue(dictionary.search("b.."));
        assertTrue(dictionary.search("m.d"));
        dictionary = new DesignProblems.WordDictionary();
        assertFalse(dictionary.search("a"));
    }

    @Test
    public void testLFUCache() {
        DesignProblems.LFUCache cache = new DesignProblems.LFUCache(2);
        cache.put(1,1);
        cache.put(2,2);
        assertEquals(cache.get(1), 1);
        cache.put(3,3);
        assertEquals(cache.get(2), -1);
        assertEquals(cache.get(3), 3);
        cache.put(4,4);
        assertEquals(cache.get(1), -1);
        assertEquals(cache.get(3), 3);
        assertEquals(cache.get(4), 4);
    }

}
