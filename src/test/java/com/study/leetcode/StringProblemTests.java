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
}
