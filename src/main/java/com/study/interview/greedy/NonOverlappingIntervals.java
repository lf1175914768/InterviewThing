package com.study.interview.greedy;

import java.util.Arrays;
import java.util.Comparator;

/**
 * 计算让一组区间不重叠所需要移除的区间个数。
 * Input: [ [1,2], [1,2], [1,2] ]
 * Output: 2
 * Explanation: You need to remove two [1,2] to make the rest of intervals non-overlapping.
 *
 * Input: [ [1,2], [2,3] ]
 * Output: 0
 * Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
 *
 * @date: 2020/3/24 22:51
 * @author: Liufeng
 **/
public class NonOverlappingIntervals {

    public int eraseOverlapIntervals(int[][] intervals) {
        if(intervals == null || intervals.length == 0) {
            return 0;
        }
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[1]));
        int cnt = 1, end = intervals[0][1];
        for(int i = 1; i < intervals.length; i++) {
            if(intervals[i][0] < end) {
                continue;
            }
            end = intervals[i][1];
            cnt++;
        }
        return intervals.length - cnt;
    }

    /**
     *气球在一个水平数轴上摆放，可以重叠，飞镖垂直投向坐标轴，使得路径上的气球都被刺破。求解最小的投飞镖次数使所有气球都被刺破。
     * Input: [[10,16], [2,8], [1,6], [7,12]]
     * Output: 2
     */
    public int findMinArrowsShot(int[][] points) {
        if(points == null || points.length == 0)
            return 0;
        Arrays.sort(points, Comparator.comparingInt(o -> o[1]));
        int cnt = 1, end = points[0][1];
        for(int i = 1; i < points.length; i++) {
            if(points[i][0] <= end)
                continue;
            cnt++;
            end = points[i][1];
        }
        return cnt;
    }

}
