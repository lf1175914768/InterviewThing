package com.study.interview.greedy;

import java.util.Arrays;

/**
 * 每个孩子都有一个满足度 grid，每个饼干都有一个大小 size，只有饼干的大小大于等于一个孩子的满足度，
 * 该孩子才会获得满足。求解最多可以获得满足的孩子数量。
 * Input: grid[1,3] , size[1,2,4]
 * Output: 2
 * @date: 2020/3/24 22:44
 * @author: Liufeng
 **/
public class AssignCookie {

    public int findContentChildren(int[] grid, int[] size) {
        if(grid == null || size == null)
            return 0;
        Arrays.sort(grid);
        Arrays.sort(size);
        int gi = 0, si = 0;
        while(gi < grid.length && si < size.length) {
            if(grid[gi] <= size[si]) {
                ++gi;
            }
            ++si;
        }
        return gi;
    }

}
