package com.study.leetcode;

import org.junit.Test;
import static org.junit.Assert.*;

public class StringProblemTests {

    private StringProblems problems = new StringProblems();

    @Test
    public void testMinAddToMakeValid() {
        assertEquals(problems.minAddToMakeValid("())"), 1);
        assertEquals(problems.minAddToMakeValid("((("), 3);
    }

    @Test
    public void testIsSubsequence() {
        assertTrue(problems.isSubsequence("abc", "ahbgdc"));
        assertTrue(problems.isSubsequence_v2("abc", "ahbgdc"));
        assertTrue(problems.isSubsequence_v2("abc", "ddddddadddbeeeedck"));
        assertFalse(problems.isSubsequence("axc", "ahbgdc"));
        assertFalse(problems.isSubsequence_v2("axc", "ahbgdc"));
    }

    @Test
    public void testCompareVersion() {
        assertEquals(problems.compareVersion("0.1", "1.1"), -1);
        assertEquals(problems.compareVersion("1.01", "1.001"), 0);
        assertEquals(problems.compareVersion("1.0", "1.000"), 0);
    }

    @Test
    public void testNumDistinct() {
        assertEquals(problems.numDistinct("rabbbit", "rabbit"), 3);
        assertEquals(problems.numDistinct("babgbag", "bag"), 5);
    }

    @Test
    public void testReverseWords() {
        assertEquals(problems.reverseWords("the sky is blue"), "blue is sky the");
        assertEquals(problems.reverseWords("  hello world  "), "world hello");
        assertEquals(problems.reverseWords("a good   example"), "example good a");
    }

    @Test
    public void testRestoreIpAddresses() {
        String[] res = new String[] {"255.255.11.135", "255.255.111.35"};
        assertArrayEquals(problems.restoreIpAddresses("25525511135").toArray(new String[0]), res);
        res = new String[]  {"0.0.0.0"};
        assertArrayEquals(problems.restoreIpAddresses("0000").toArray(new String[0]), res);
        res = new String[] {"1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"};
        assertArrayEquals(problems.restoreIpAddresses("101023").toArray(new String[0]), res);
    }

    @Test
    public void testConvert() {
        assertEquals(problems.convert("PAYPALISHIRING", 3), "PAHNAPLSIIGYIR");
        assertEquals(problems.convert("PAYPALISHIRING", 4), "PINALSIGYAHRPI");
        assertEquals(problems.convert("A", 1), "A");
    }

    @Test
    public void testCharacterReplacement() {
        assertEquals(problems.characterReplacement("ABAB", 2), 4);
        assertEquals(problems.characterReplacement("AABABBA", 1), 4);
        assertEquals(problems.characterReplacement("AABCABBB", 2), 6);
    }

    @Test
    public void testFindSubstring() {
        String[] words = new String[] {"foo","bar"};
        Integer[] res = new Integer[] {0, 9};
        assertArrayEquals(problems.findSubstring("barfoothefoobarman", words).toArray(new Integer[0]), res);
        words = new String[] {"word","good","best","word"};
        res = new Integer[] {};
        assertArrayEquals(problems.findSubstring("wordgoodgoodgoodbestword", words).toArray(new Integer[0]), res);
        words = new String[] {"bar","foo","the"};
        res = new Integer[] {6,9,12};
        assertArrayEquals(problems.findSubstring("barfoofoobarthefoobarman", words).toArray(new Integer[0]), res);
    }

    @Test
    public void testIntToRoman() {
        assertEquals(problems.intToRoman(3), "III");
        assertEquals(problems.intToRoman(4), "IV");
        assertEquals(problems.intToRoman(9), "IX");
        assertEquals(problems.intToRoman(58), "LVIII");
        assertEquals(problems.intToRoman(1994), "MCMXCIV");
    }

    @Test
    public void testMultiply() {
        assertEquals(problems.multiply("2", "3"), "6");
        assertEquals(problems.multiply_v2("2", "3"), "6");
        assertEquals(problems.multiply("123", "456"), "56088");
        assertEquals(problems.multiply_v2("123", "456"), "56088");
    }

    @Test
    public void testFullJustify() {
        String[] words = {"This", "is", "an", "example", "of", "text", "justification."};
        String[] res = {"This    is    an", "example  of text", "justification.  "};
        assertArrayEquals(problems.fullJustify(words, 16).toArray(new String[0]), res);
        words = new String[] {"What","must","be","acknowledgment","shall","be"};
        res = new String[] {"What   must   be",    "acknowledgment  ", "shall be        "};
        assertArrayEquals(problems.fullJustify(words, 16).toArray(new String[0]), res);
        words = new String[] {"Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"};
        res = new String[] {"Science  is  what we","understand      well","enough to explain to","a  computer.  Art is", "everything  else  we", "do                  "};
        assertArrayEquals(problems.fullJustify(words, 20).toArray(new String[0]), res);
    }

    @Test(timeout = 100)
    public void testSimplifyPath() {
        assertEquals(problems.simplifyPath("/home/"), "/home");
        assertEquals(problems.simplifyPath_v2("/home/"), "/home");
        assertEquals(problems.simplifyPath("/../"), "/");
        assertEquals(problems.simplifyPath_v2("/../"), "/");
        assertEquals(problems.simplifyPath("/a/./b/../../c/"), "/c");
        assertEquals(problems.simplifyPath_v2("/a/./b/../../c/"), "/c");
    }

    @Test
    public void testRemoveDuplicateLetters() {
        assertEquals(problems.removeDuplicateLetters("bcabc"), "abc");
        assertEquals(problems.removeDuplicateLetters("cbacdcbc"), "acdb");
    }
}
