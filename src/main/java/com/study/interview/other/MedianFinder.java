package com.study.interview.other;

import java.util.PriorityQueue;

/**
 * <p>description: 查找数据流的中位数  </p>
 * <p>className:  MedianFinder </p>
 * <p>create time:  2022/4/29 17:42 </p>
 *
 * 对应 leetcode 中第 295 题。
 *
 * @author feng.liu
 * @since 1.0
 **/
public class MedianFinder {

    /*
     * 使用两个堆来进行实现
     * 要求中位数的话，可以分别使用大顶堆和小顶堆的和除以2来实现。
     *
     *             .
     *            / \
     *           /   \                  ______        ______
     *          /_____\                 \    /       /      \
     *         /       \                 \  /       /________\
     *        /         \                 .
     *       /___________\
     */

    /**
     * large 是一个小顶堆，但是其中保存的元素都是偏大的数据
     */
    private PriorityQueue<Integer> large;
    /**
     * small 是一个大顶堆，但是其中保存的元素都是偏小的数据
     */
    private PriorityQueue<Integer> small;

    public MedianFinder() {
        // 小顶堆
        large = new PriorityQueue<>();
        // 大顶堆
        small = new PriorityQueue<>((a, b) -> b - a);
    }

    public double findMedian() {
        // 如果元素不一样多，多的那个堆的堆顶元素就是中位数
        if (large.size() < small.size()) {
            return small.peek();
        } else if (large.size() > small.size()) {
            return large.peek();
        }
        // 如果元素一样多，两个堆堆顶元素的平均数就是中位数
        return (small.peek() + large.peek()) / 2.0;
    }

    /**
     * 我们的梯形和小倒三角形冲中间切开得到的， 那么梯形中的最小宽度要大于等于小倒三角的最大宽度，
     * 这样他俩才能拼成一个大的倒三角。
     * 也就是说，添加元素的时候，不仅要维护 large 和 small 的元素个数之差不超过 1，还要维护 large
     * 堆的堆顶元素要大于等于 small堆的堆顶元素。
     *
     * 所以加单说的话，想要往 large 里面添加元素，不能直接添加，而是要先往 small 里面添加，
     * 然后再把 small 的堆顶元素加到 large 中；向 small 中添加元素同理。
     */
    public void addNum(int num) {
        if (small.size() >= large.size()) {
            small.offer(num);
            large.offer(small.poll());
        } else {
            large.offer(num);
            small.offer(large.poll());
        }
    }
}
