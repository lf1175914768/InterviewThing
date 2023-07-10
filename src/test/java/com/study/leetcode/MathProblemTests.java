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

    @Test
    public void testIntegerReplacement() {
        assertEquals(problem.integerReplacement(8), 3);
    }

    @Test
    public void testLexicalOrder() {
        Integer[] res = new Integer[] {1,10,11,12,13,2,3,4,5,6,7,8,9};
        assertArrayEquals(problem.lexicalOrder(13).toArray(new Integer[0]), res);
        assertArrayEquals(problem.lexicalOrder_v2(13).toArray(new Integer[0]), res);
        res = new Integer[] {1,2};
        assertArrayEquals(problem.lexicalOrder(2).toArray(new Integer[0]), res);
        assertArrayEquals(problem.lexicalOrder_v2(2).toArray(new Integer[0]), res);
    }

    @Test
    public void testCountDigitOne() {
        assertEquals(problem.countDigitOne(12), 5);
        assertEquals(problem.countDigitOne(13), 6);
    }
}
