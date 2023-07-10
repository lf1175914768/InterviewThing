package com.study.leetcode;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
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

    @Test
    public void testCloneGraph() {
        GraphProblems.Node node1 = new GraphProblems.Node(1, new ArrayList<>());
        GraphProblems.Node node2 = new GraphProblems.Node(2, new ArrayList<>());
        GraphProblems.Node node3 = new GraphProblems.Node(3, new ArrayList<>());
        GraphProblems.Node node4 = new GraphProblems.Node(4, new ArrayList<>());
        node1.neighbors.add(node2);
        node1.neighbors.add(node4);
        node2.neighbors.add(node1);
        node2.neighbors.add(node3);
        node3.neighbors.add(node2);
        node3.neighbors.add(node4);
        node4.neighbors.add(node1);
        node4.neighbors.add(node3);

        GraphProblems.Node finalNode = problems.cloneGraph(node2);
        assertEquals(finalNode.val, 2);
        Integer[] res = {1, 3};
        assertArrayEquals(finalNode.neighbors.stream().map(item -> item.val).toArray(Integer[]::new), res);
        GraphProblems.Node node_v2 = problems.cloneGraph_v2(node3);
        assertEquals(node_v2.val, 3);
        res = new Integer[] {2, 4};
        assertArrayEquals(node_v2.neighbors.stream().map(item -> item.val).toArray(Integer[]::new), res);
    }
}
