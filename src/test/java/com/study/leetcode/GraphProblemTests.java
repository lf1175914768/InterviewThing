package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/**
 * <p>description: 图相关算法测试类  </p>
 * <p>className:  GraphProblemTests </p>
 * <p>create time:  2022/1/4 11:44 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class GraphProblemTests {

    private GraphProblems problems;

    @Before
    public void init() {
        problems = new GraphProblems();
    }

    @Test
    public void testAllPathsSourceTarget() {
        int[][] graph = new int[][] {{1,2}, {3}, {3}, {}};
        List<List<Integer>> res = problems.allPathsSourceTarget(graph);
        assertEquals(res.size(), 2);
        assertEquals(res.get(0).get(0).intValue(), 0);
        assertEquals(res.get(0).get(1).intValue(), 1);
        assertEquals(res.get(0).get(2).intValue(), 3);
        assertEquals(res.get(1).get(0).intValue(), 0);
        assertEquals(res.get(1).get(1).intValue(), 2);
        assertEquals(res.get(1).get(2).intValue(), 3);
    }

    @Test
    public void testCanFinish() {
        int[][] prerequisites = new int[][] {{1, 0}};
        assertTrue(problems.canFinish(2, prerequisites));
        assertTrue(problems.canFinish_v2(2, prerequisites));
        prerequisites = new int[][] {{1, 0}, {0, 1}};
        assertFalse(problems.canFinish(2, prerequisites));
        assertFalse(problems.canFinish_v2(2, prerequisites));
    }

    @Test
    public void testFindOrder() {
        int[][] prerequisites = new int[][] {{1, 0}};
        int[] order = problems.findOrder(2, prerequisites);
        assertEquals(order[0], 0);
        assertEquals(order[1], 1);
    }
}
