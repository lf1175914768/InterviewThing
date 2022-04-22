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
}
