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
}
