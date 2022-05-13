package com.study.leetcode;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * <p>description: 数学技巧类的题目测试类  </p>
 * <p>className:  MathProblemTests </p>
 * <p>create time:  2022/5/13 13:50 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class MathProblemTests {

    private final MathProblems problem = new MathProblems();

    @Test
    public void testNthUglyNumber() {
        assertEquals(problem.nthUglyNumber(10), 12);
        assertEquals(problem.nthUglyNumber(1), 1);
    }
}
