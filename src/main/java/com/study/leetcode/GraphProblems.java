package com.study.leetcode;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * <p>description: 图相关的算法  </p>
 * <p>className:  GraphProblems </p>
 * <p>create time:  2022/1/4 11:41 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class GraphProblems {

    // -------所有可能的路径 start >>--------

    /**
     * 给你一个有 n 个节点的 有向无环图（DAG），请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）
     * 二维数组的第 i 个数组中的单元都表示有向图中 i 号节点所能到达的下一些节点，空就是没有下一个结点了。
     * 译者注：有向图是有方向的，即规定了 a→b 你就不能从 b→a 。
     *
     * 解题思路：
     * 以 0 为起点遍历图，同时记录遍历过的路径，当遍历到终点时将路径记录下来即可。
     *
     * @param graph input graph
     * @return list of path
     */
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new LinkedList<>();
        allPathsTraverse(graph, 0, new LinkedList<>(), res);
        return res;
    }

    private void allPathsTraverse(int[][] graph, int s, Deque<Integer> path, List<List<Integer>> res) {
        // 添加节点 s 到路径
        path.addLast(s);
        int n = graph.length;
        if (s == n - 1) {
            // 到达终点
            res.add(new LinkedList<>(path));
            path.removeLast();
            return;
        }
        // 递归每个相邻节点
        for (int v : graph[s]) {
            allPathsTraverse(graph, v, path, res);
        }

        // 从路径中移除节点s
        path.removeLast();
    }

    // -------所有可能的路径 << end --------

    // -------课程表 start >>--------

    /**
     * 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
     * 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程 bi 。
     *
     * 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
     * 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
     *
     * 解题思路：
     * 看到依赖问题，首先想到的就是把问题转化成【有向图】这种数据结构，只要图中存在环，那就说明存在循环依赖。
     *
     * 对应 leetcode 中第 207 题。
     *
     * @param numCourses number of courses
     * @param prerequisites prerequisites
     * @return whether has cycle
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // 构建图
        List<Integer>[] graph = buildGraph(numCourses, prerequisites);
        // 记录遍历过的节点，防止走回头路
        boolean[] visited = new boolean[numCourses];
        // 记录一次 traverse 递归经过的节点
        boolean[] onPath = new boolean[numCourses];
        AtomicBoolean hasCycle = new AtomicBoolean(false);

        for (int i = 0; i < numCourses; i++) {
            // 遍历图中的所有节点
            if (hasCycle.get()) {
                break;
            }
            canFinishTraverse(graph, i, visited, onPath, hasCycle);
        }
        // 只要没有循环依赖就可以完成所有课程
        return !hasCycle.get();
    }

    private void canFinishTraverse(List<Integer>[] graph, int s, boolean[] visited, boolean[] onPath, AtomicBoolean hasCycle) {
        if (onPath[s]) {
            // 出现环
            hasCycle.set(true);
        }
        if (visited[s] || hasCycle.get()) {
            // 如果已经遍历过，或者找到了环，那么停止遍历
            return;
        }
        // 前序遍历位置
        visited[s] = true;
        onPath[s] = true;
        for (Integer t : graph[s]) {
            canFinishTraverse(graph, t, visited, onPath, hasCycle);
        }
        onPath[s] = false;
    }

    private List<Integer>[] buildGraph(int numCourses, int[][] prerequisites) {
        // 图中共有 numCourses 个节点
        List<Integer>[] graph = new LinkedList[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new LinkedList<>();
        }
        for (int[] edge : prerequisites) {
            int from = edge[1];
            int to = edge[0];
            // 修完课程 from 才能修课程 to
            // 在图中添加一条从 from 到 to 的有向边
            graph[from].add(to);
        }
        return graph;
    }

    /**
     * 流程：
     * 1、统计课程安排图中的每个节点的入度，生成入度表 indegree
     * 2、借助一个队列 queue，将所有入度为 0 的节点入队。
     * 3、当 queue 非空时，依次将队首节点入队，在课程安排图中删除此节点 pre:
     *    3.1、并不是真正从邻接表中删除此节点 pre，而是将此节点对应所有邻接节点 cur 的入度 - 1，即 indegree[cur] -= 1
     *    3.2、当入度 -1后邻接节点 cur的入度为 0，说明 cur所有的前驱节点已经被“删除”，此时将 cur 入队
     * 4、在每次 pre 出队时，执行 numCourses--
     *    4.1、若整个课程安排图是有向无环图，则所有节点一定都入队并出队过，即完成拓扑排序。换个角度说，若课程安排图中存在环，一定有节点的入度始终不为0.
     *    4.2、因此，拓扑排序出队次数等于课程个数，返回 numCourse == 0 判断课程是否可以安排成功。
     *
     * 对应 leetcode 中第 207 题。
     *
     * @param numCourses number of courses
     * @param prerequisites prerequisites
     * @return whether has cycle
     */
    public boolean canFinish_v2(int numCourses, int[][] prerequisites) {
        int[] inDegrees = new int[numCourses];
        List<List<Integer>> adjacency = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            adjacency.add(new ArrayList<>());
        }
        // get the indegree and adjacency of every course
        for (int[] cp : prerequisites) {
            inDegrees[cp[0]]++;
            adjacency.get(cp[1]).add(cp[0]);
        }
        // get all the courses with the indegree of 0
        for (int i = 0; i < numCourses; i++) {
            if (inDegrees[i] == 0)
                queue.add(i);
        }
        // BFS topSort
        while (!queue.isEmpty()) {
            int pre = queue.poll();
            numCourses--;
            for (Integer cur : adjacency.get(pre)) {
                if (--inDegrees[cur] == 0)
                    queue.add(cur);
            }
        }
        return numCourses == 0;
    }

    // -------课程表 << end --------

    // -------课程表II start >>--------

    /**
     * 现在你总共有 numCourses 门课需要选，记为0到numCourses - 1。给你一个数组prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修bi 。
     * 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
     * 返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组
     *
     * 解题思路：
     * 先说最重要的部分：
     * 1、【拓扑排序】是专门应用于有向图的算法；
     * 2、BFS的写法就叫【拓扑排序】，这里还用到了贪心算法的思想，贪的点是：当前让入度是 0 的那些节点入队。
     * 3、【拓扑排序】的结果不唯一。
     * 4、删除节点的操作，通过【入度数组】体现，这个技巧要掌握。
     * 5、【拓扑排序】的一个附加效果是：<strong>能够顺带检测有向图中是否有环</strong>，这个知识点非常重要。
     *
     * 对应 leetcode 中第 210 题
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses <= 0) {
            return new int[0];
        }
        List<Set<Integer>> adj = new ArrayList<>();
        // [1, 0]  0 -> 1
        int[] inDegree = new int[numCourses];
        Queue<Integer> queue = new LinkedList<>();
        int[] res = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            adj.add(new HashSet<>());
        }
        for (int[] edge : prerequisites) {
            int from = edge[1], to = edge[0];
            inDegree[to]++;
            adj.get(from).add(to);
        }
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0)
                queue.offer(i);
        }
        int count = 0;
        while (!queue.isEmpty()) {
            Integer head = queue.poll();
            res[count++] = head;
            for (Integer cur : adj.get(head)) {
                if (--inDegree[cur] == 0)
                    queue.offer(cur);
            }
        }
        if (count == numCourses) {
            return res;
        }
        return new int[0];
    }

    // -------课程表II << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------
}
