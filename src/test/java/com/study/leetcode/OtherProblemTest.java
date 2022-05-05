package com.study.leetcode;

import com.study.interview.other.MedianFinder;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * <p>description:   </p>
 * <p>className:  OtherProblemTest </p>
 * <p>create time:  2022/5/5 17:05 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class OtherProblemTest {

    @Test
    public void testMedianFinder() {
        MedianFinder finder = new MedianFinder();
        finder.addNum(1);
        finder.addNum(2);
        assertEquals(finder.findMedian(), 1.5, 0.001);
        finder.addNum(3);
        assertEquals(finder.findMedian(), 2.0, 0.0001);
        finder.addNum(1);
        assertEquals(finder.findMedian(), 1.5, 0.0001);

    }
}
