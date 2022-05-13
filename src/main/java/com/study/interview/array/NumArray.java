package com.study.interview.array;

/**
 * <p>description:   </p>
 * <p>className:  NumArray </p>
 * <p>create time:  2022/4/6 11:30 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class NumArray {

    /*
     * 给定一个整数数组  nums，处理以下类型的多个查询:
     * 计算索引 left 和 right （包含 left 和 right）之间的 nums 元素的 和 ，其中 left <= right
       实现 NumArray 类：

       NumArray(int[] nums) 使用数组 nums 初始化对象
       int sumRange(int i, int j) 返回数组 nums 中索引 left 和 right 之间的元素的 总和 ，
       包含 left 和 right 两点（也就是 nums[left] + nums[left + 1] + ... + nums[right] )
     */

    private int[] preSum;

    public NumArray() {}

    public NumArray(int[] nums) {
        preSum = new int[nums.length + 1]; 
        preSum[0] = 0;
        for (int i = 1; i < preSum.length; i++) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }
    }

    public int sumRange(int left, int right) {
        return preSum[right + 1] - preSum[left];
    }

    private int[][] matrixSum;

    public void initMatrixSum(int[][] matrix) {
        int rows = matrix.length, cols = matrix[0].length;
        matrixSum = new int[rows + 1][cols + 1];
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                matrixSum[i][j] = matrixSum[i - 1][j] + matrixSum[i][j - 1]
                        + matrix[i - 1][j - 1] - matrixSum[i - 1][j - 1];
            }
        }
    }

    public int sumRegion(int x1, int y1, int x2, int y2) {
        return matrixSum[x2 + 1][y2 + 1] - matrixSum[x2 + 1][y1] - matrixSum[x1][y2 + 1] + matrixSum[x1][y1];
    }
}
