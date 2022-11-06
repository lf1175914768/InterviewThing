package com.study.leetcode;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * <p>description: 数组相关的问题  </p>
 * <p>className:  ArrayProblems </p>
 * <p>create time:  2022/4/6 11:28 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class ArrayProblems {

    // -------和为K的子数组 start >>--------

    /**
     * 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。
     *
     * 首先使用前缀和技巧 解答。
     *
     * 对应 leetcode 中第 560 题。
     */
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        int[] preSum = new int[n + 1];
        for (int i = 1; i < preSum.length; i++) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }

        int res = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                // 子数组 nums[j..i - 1] 的元素和
                if (preSum[i] - preSum[j] == k)
                    res++;
            }
        }
        return res;
    }

    /**
     * 上面的方法 优化版
     *
     * 优化的思路是： 我直接记录下有几个 preSum[j] = preSum[i] - k ，直接更新结果，避免内层的循环
     *
     * 其中 preSum 保存的是 前缀和 以及 对应出现的次数，
     * 比如对于 nums   [3, 5, 2, -2, 4, 1]
     * 对应的 前缀和 [0, 3, 8, 10, 8, 12, 13], 因为第一个数字一定是0，所以有 base case：preSum.put(0, 1)
     */
    public int subarraySum_v2(int[] nums, int k) {
        int n = nums.length;
        // map: 前缀和 -> 该前缀和出现的次数
        Map<Integer, Integer> preSum = new HashMap<>();
        // base case
        preSum.put(0, 1);
        int res = 0, sum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            // 这是我们想找的前缀和 nums[0..j]
            int sum0_j = sum - k;
            if (preSum.containsKey(sum0_j)) {
                res += preSum.get(sum0_j);
            }
            // 把前缀和 nums[0..i] 加入并记录出现次数
            preSum.put(sum, preSum.getOrDefault(sum, 0) + 1);
        }
        return res;
    }

    // -------和为K的子数组 << end --------

    // -------下一个更大的元素II start >>--------

    /**
     * 给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。
     * 数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1 。
     *
     * 使用单调栈的方法进行解答
     *
     * 对应 leetcode 中第 503 题。
     */
    public int[] nextGreaterElements(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[nums.length];
        int len = nums.length;
        for (int i = 2 * len - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums[i % len]) {
                stack.pop();
            }
            if (stack.isEmpty()) {
                res[i % len] = -1;
            } else {
                res[i % len] = stack.peek();
            }
            stack.push(nums[i % len]);
        }
        return res;
    }

    // -------下一个更大的元素II << end --------

    // -------滑动窗口最大值 start >>--------

    /**
     * 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
     * 返回 滑动窗口中的最大值 。
     *
     * 对应 leetcode 中第 239 题
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        // 双向队列，保存当前窗口最大值的数组位置，保证队列中数组位置的数值按照从大到小的顺序排列
        Deque<Integer> queue = new LinkedList<>();
        // 结果数组
        int[] res = new int[nums.length - k + 1];
        for (int i = 0; i < nums.length; i++) {
            while (!queue.isEmpty() && nums[queue.peekLast()] <= nums[i]) {
                queue.pollLast();
            }
            queue.offerLast(i);
            // 判断当前队列中队首的值是否有效，也就是是否在窗口中，不满足的话，需要移除
            if (queue.peekFirst() <= i - k) {
                queue.pollFirst();
            }
            // 当窗口长度为 k 时，保存当前窗口最大值
            if (i + 1 >= k) {
                res[i - k + 1] = nums[queue.peekFirst()];
            }
        }
        return res;
    }

    // -------滑动窗口最大值 << end --------

    // -------爱吃香蕉的珂珂 start >>--------

    /**
     * 珂珂喜欢吃香蕉。这里有 N 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 H 小时后回来。
     * 珂珂可以决定她吃香蕉的速度 K （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 K 根。如果这堆香蕉少于 K 根，她将吃掉这堆的所有香蕉，
     * 然后这一小时内不会再吃更多的香蕉。 珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
     * 返回她可以在 H 小时内吃掉所有香蕉的最小速度 K（K 为整数）。
     *
     * 根据题意可以知道：
     * 珂珂吃香蕉的速度越小，耗时越多。反之，速度越大，耗时越少，这是题目的单调性。
     * 我们要找的是速度，因为题目限制了珂珂一个小时之内只能选择一堆香蕉吃，因此速度最大值就是这几堆香蕉中，数量最多的那一堆。速度的最小值是 1，
     * 其实还可以在分析一下下界是多少，由于二分搜索的时间复杂度很低，严格的分析不是很有必要。
     * 还是因为珂珂一个小时之内只能选择一堆香蕉吃，因此：每堆香蕉吃完的耗时 = 这堆香蕉的数量 / 珂珂一小时吃香蕉的数量。根据题意，这里的 / 在不能
     * 整除的时候，需要 上取整。
     * 注意：
     * 当 【二分查找】算法猜测的速度恰好使得珂珂在规定的时间内吃完所有的香蕉的时候，还应该去尝试更小的速度是不是还可以保证在规定的时间内吃完香蕉。
     * 这是因为题目问的是 【最小速度】。
     *
     * 对应 leetcode 中第 875 题。
     */
    public int minEatingSpeed(int[] piles, int h) {
        int maxVal = 1;
        for (int pile : piles) {
            maxVal = Math.max(maxVal, pile);
        }
        // 速度最小的时候，耗时最长
        int left = 1;
        // 速度最大的时候，耗时最短
        int right = maxVal;
        while (left < right) {
            int mid = left + ((right - left) >>> 1);
            if (minEatingSpeedFunction(piles, mid) > h) {
                // 耗时太多，说明速度太慢了，下一轮搜索区间是 [mid + 1, right)
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    private int minEatingSpeedFunction(int[] piles, int speed) {
        int hours = 0;
        for (int i = 0; i < piles.length; i++) {
            hours += piles[i] / speed;
            if (piles[i] % speed > 0) {
                hours++;
            }
        }
        return hours;
    }

    // -------爱吃香蕉的珂珂 << end --------

    // -------折木棍 start >>--------

    /**
     * 折木棍
     *
     * 在你的面前从左到右摆放着 nn 根长短不一的木棍，你每次可以折断一根木棍，并将折断后得到的两根木棍一左一右放在原来的位置
     * （即若原木棍有左邻居，则两根新木棍必须放在左邻居的右边，若原木棍有右邻居，新木棍必须放在右邻居的左边，所有木棍保持左右排列）。
     * 折断后的两根木棍的长度必须为整数，且它们之和等于折断前的木棍长度。你希望最终从左到右的木棍长度单调不减，那么你需要折断多少次呢？
     *
     * 使用贪心的方法进行解答。
     * 从后往前遍历，当当前位置木棍长度比后面的长时，就需要将其折成n份，策略是折成的 n 份中最小值尽量大，而最大值不超过后面的数。
     */
    public int foldStick(int[] nums) {
        int len = nums.length;
        if (len == 0) return 0;
        int res = 0, maxHeight = nums[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            if (nums[i] > maxHeight) {
                if (nums[i] % maxHeight == 0) {
                    int stick = nums[i] / maxHeight;
                    res += (stick - 1);
                } else {
                    // 查看当前长度应该分成几份
                    int stick = nums[i] / maxHeight + 1;
                    res += (stick - 1);
                    // 因为要分成 stick 份，所以一份的长度至少为 nums[i] / stick
                    maxHeight = nums[i] / stick;
                }
            } else {
                maxHeight = nums[i];
            }
        }
        return res;
    }

    // -------折木棍 << end --------

    // -------搜索二维矩阵II start >>--------

    /**
     * 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
     *
     * 每行的元素从左到右升序排列。
     * 每列的元素从上到下升序排列。
     *
     * 解题思路：
     * 看到有序，第一反应就是二分查找，最直接的做法，就是一行一行的进行二分查找。
     * 此外，结合有序的性质，一些情况下可以提前结束。
     *
     * 对应 leetcode 中第 240 题。
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) return false;
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][0] > target) break;
            if (matrix[i][matrix[i].length - 1] < target) continue;
            int col = searchMatrixBinarySearch(matrix[i], target);
            if (col != -1) {
                return true;
            }
        }
        return false;
    }

    private int searchMatrixBinarySearch(int[] matrix, int target) {
        int left = 0, right = matrix.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (matrix[mid] == target) {
                return mid;
            } else if (matrix[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    /**
     * 数组从左到右 和 从上到下都是升序的，如果从右上角出发开始遍历呢？
     * 会发现每次都是向左数字会变小，向下数字会变大， 有点和二分查找树相似。二分查找树的话，是向左数字变小，向右数字变大。
     * 所以我们可以把 target 和当前值比较。
     *  如果 target 的值大于当前值，那么就向下走，
     *  如果 target 的值小于当前值，那么就向左走。
     *  如果相等的话，直接返回 true。
     */
    public boolean searchMatrix_v2(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) return false;
        for (int i = 0, j = matrix[0].length - 1; i < matrix.length && j >= 0;) {
            if (matrix[i][j] > target) {
                j--;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                return true;
            }
        }
        return false;
    }

    // -------搜索二维矩阵II << end --------

    // -------最长重复子数组 start >>--------

    /**
     * 给两个整数数组 nums1 和 nums2 ，返回 两个数组中 公共的 、长度最长的子数组的长度 。
     *
     * 动态规划思想是希望连续的，也就是说上一个状态和下一个状态（自变量）之间有关系而且连续。
     * 公共子数组相当于子串是 连续的。
     * 定义 dp[i][j]：表示第一个数组 A 前 i 个元素和数组B前 j 个元素组成的最长公共子数组（相当于子串）的长度。
     * 我们在计算 dp[i][j] 的时候：
     * 1、若当前两个元素相同，即 A[i] == B[j],则说明当前元素可以构成公共子数组，所以还要加上 他们的前一个元素构成的最长公共子数组的长度
     * （在原来的基础上 + 1），此时，状态转移方程： dp[i][j] = dp[i - 1][j - 1] + 1;
     * 2、若当前两个元素不同，即 A[i] != B[j], 则说明当前元素无法构成公共子数组。因为公共子数组必须是连续的，儿此时的元素值不同，
     * 相当于直接断开了，此时窗台转移方程： dp[i][j] = 0;
     *
     * 对应 leetcode 中第 718 题。
     */
    public int findLength(int[] nums1, int[] nums2) {
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        int res = 0;
        for (int i = 1; i <= nums1.length; i++) {
            for (int j = 1; j <= nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = 0;
                }
                res = Math.max(res, dp[i][j]);
            }
        }
        return res;
    }

    public int findLength_v2(int[] nums1, int[] nums2) {
        int n = nums1.length, m = nums2.length;
        int ret = 0;
        for (int i = 0; i < n; i++) {
            int len = Math.min(m, n - i);
            int maxLen = findLengthMax(nums1, nums2, i, 0, len);
            ret = Math.max(ret, maxLen);
        }
        for (int i = 0; i < m; i++) {
            int len = Math.min(n, m - i);
            int maxLen = findLengthMax(nums1, nums2, 0, i, len);
            ret = Math.max(ret, maxLen);
        }
        return ret;
    }

    private int findLengthMax(int[] nums1, int[] nums2, int aStart, int bStart, int len) {
        int ret = 0, k = 0;
        for (int i = 0; i < len; i++) {
            if (nums1[aStart + i] == nums2[bStart + i]) {
                k++;
            } else {
                k = 0;
            }
            ret = Math.max(ret, k);
        }
        return ret;
    }

    // -------最长重复子数组 << end --------

    // -------长度最小的子数组 start >>--------

    /**
     * 给定一个含有 n 个正整数的数组和一个正整数 target 。
     * 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
     *
     * 使用 滑动窗口的方法进行解答。
     *
     * 对应 leetcode 中第 209 题。
     */
    public int minSubArrayLen(int target, int[] nums) {
        int left = 0, right = 0;
        int len = nums.length, sum = 0, res = Integer.MAX_VALUE;
        while (right < len) {
            int num = nums[right];
            right++;
            sum += num;
            while (sum >= target) {
                res = Math.min(res, right - left);
                num = nums[left];
                sum -= num;
                left++;
            }
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    // -------长度最小的子数组 << end --------

    // -------寻找两个正序数组的中位数 start >>--------

    /**
     * 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
     * 算法的时间复杂度应该为 O(log (m+n)) 。
     *
     * 解题思路：
     * 我们可以将数组进行切分。一个长度为 m 的数组，有 0 到 m 总共 m + 1 个位置可以且切。
     * 我们把数组 nums1 和 nums2 分别在 i 和 j 进行切割。
     * 将 i 的左边和 j 的左边组合成【左半部分】，将 i 的右边和 j 的右边组合成 【右半部分】。
     * 1、当 nums1 数组和 nums2 数组的总长度是 偶数时，如果我们能够保证左半部分的长度等于右半部分的长度
     *   i + j = m - i + n - j，也就是  j = (m + n) / 2 - i;
     *   左半部分最大的值小于等于右半部分最小的值 max(nums1[i - 1], nums2[j - 1]) <= min(nums1[i], nums2[j])。
     *  那么这种情况下，中位数就可以表示如下： （左半部分最大值 + 右半部分最小值） / 2.
     *  也就是 (max(nums1[i - 1], nums2[j - 1]) + min(nums1[i], nums2[j])) / 2.
     * 2、当 nums1 数组和 nums2 数组的总长度是奇数时，如果我们能够保证 左半部分的长度比右半部分的长度大 1.
     *    i + j = m - i + n - j + 1, 也就是 j = (m + n + 1) / 2 - i;
     *    那么这种情况下，中位数就是左半部分的最大值，也就是左半部分比右半部分多出的哪一个数  max(nums1[i - 1], nums2[j - 1])
     *
     * 上面的第一个条件我们其实可以合并为 j = (m + n + 1) / 2 - i,因为如果 m + n是偶数，由于我们取得是 int 值，所以加1不会影响结果。
     * 当然，由于 0 <= i <= m, 为了保证 0 <= j <= n, 我们必须保证 m <= n;
     * m <= n, i < m, j = (m + n + 1) / 2 - i >= (m + m + 1) / 2 - i > (m + m + 1) / 2 - m = 0.
     * m <= n, i > 0, j = (m + n + 1) / 2 - i <= (n + n + 1) / 2 - i < (n + n + 1) / 2 = n.
     *
     * 剩下的是如何保证 max(nums1[i - 1], nums2[j - 1]) <= min(nums1[i], nums2[j])， 因为 nums1 数组和 nums2 数组是有序的，所以
     * nums1[i - 1] <= nums1[i], nums2[j - 1] <= nums2[j] 这是天然的，所以，我们只需要保证 nums2[j - 1] <= nums1[i] 和 nums1[i - 1] <= nums2[j],
     * 所以我们分两种情况讨论：
     * 1、nums2[j - 1] > nums1[i],为了不越界，要保证 j > 0, i < m, 此时很明显，我们需要增加 i，为了保证数量的平衡还要减少 j，幸运的是
     * j = (m + n + 1) / 2 - i, i 增大， j自然会减少。
     * 2、nums1[i - 1] > nums2[j],为了不越界，要保证 i > 0, j < n,此时和上面相反，我们要减少 i，增加 j。
     * 上边两种情况，我们把边界都排除了，需要单独讨论。
     * 当 i = 0，或者 j = 0， 也就是切在了最前边。
     * 此时 左半部分当 j = 0时，最大的值就是 nums1[i - 1]; 当 i = 0时，最大的值就是 nums2[j - 1], 右半部分最小值和之前一样。
     * 当 i = m，或者 j = n，也就是切在了最后边。
     * 此时 左半部分最大值和之前一样，右半部分当 j = n时，最小值就是 nums1[i]; 当 i = m 时，最小值就是 nums2[j]
     *
     * 对应 leetcode 中第 4 题。
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        if (m > n) {
            return findMedianSortedArrays(nums2, nums1);
        }
        int iMin = 0, iMax = m;
        while (iMin <= iMax) {
            int i = (iMin + iMax) / 2;
            int j = (m + n + 1) / 2 - i;
            if (j != 0 && i != m && nums2[j - 1] > nums1[i]) {
                // i 需要增大
                iMin = i + 1;
            } else if (i != 0 && j != n && nums1[i - 1] > nums2[j]) {
                // i 需要减小
                iMax = i - 1;
            } else {
                // 达到要求，并且将边界条件列出来单独考虑。
                int maxLeft = 0;
                if (i == 0) {
                    maxLeft = nums2[j - 1];
                } else if (j == 0) {
                    maxLeft = nums1[i - 1];
                } else {
                    maxLeft = Math.max(nums1[i - 1], nums2[j - 1]);
                }
                if (((m + n) & 1) == 1) {
                    // 奇数的话不需要考虑右半部分
                    return maxLeft;
                }
                int minRight = 0;
                if (i == m) minRight = nums2[j];
                else if (j == n) minRight = nums1[i];
                else minRight = Math.min(nums1[i], nums2[j]);
                return (maxLeft + minRight) / 2.0;
            }
        }
        return 0.0;
    }

    // -------寻找两个正序数组的中位数 << end --------

    // -------区间列表的交集 start >>--------

    /**
     * 区间列表的交集
     *
     * 我们用 [a1, a2], [b1, b2]表示在A和B中的两个区间，如果这两个区间有交集，需要满足 b2 >= a1 && a2 >= b1,
     * 假设交集区间是 [c1,c2]，那么 c1 = max(a1, b1), c2 = min(a2, b2)
     *
     * 对应 leetcode 中第 986 题。
     */
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> res = new LinkedList<>();
        for (int i = 0, j = 0; i < firstList.length && j < secondList.length; ) {
            int a1 = firstList[i][0], a2 = firstList[i][1];
            int b1 = secondList[j][0], b2 = secondList[j][1];
            if (b2 >= a1 && a2 >= b1) {
                res.add(new int[] {Math.max(a1, b1), Math.min(a2, b2)});
            }
            if (a2 < b2) {
                i++;
            } else {
                j++;
            }
        }
        return res.toArray(new int[0][0]);
    }

    // -------区间列表的交集 << end --------

    // -------优势洗牌 start >>--------

    /**
     * 给定两个大小相等的数组 nums1 和 nums2，nums1 相对于 nums 的优势可以用满足 nums1[i] > nums2[i] 的索引 i 的数目来描述。
     * 返回 nums1 的任意排列，使其相对于 nums2 的优势最大化。
     *
     * 这道题就像田忌赛马的场景。nums1就是田忌的马，nums2就是齐王的马，数组中的元素就是马的战斗力。
     * 最优策略就是将齐王和田忌的马按照战斗力排序，然后按照战斗力排名一一对比：
     * 如果田忌的马能赢，那就比赛。如果赢不了，那就换一个垫底的人来送人头，保存实力。
     *
     * 对应 leetcode 中第 870 题。
     */
    public int[] advantageCount(int[] nums1, int[] nums2) {
        // 给 nums2 进行倒序排序
        PriorityQueue<int[]> queue = new PriorityQueue<>((int[] o1, int[] o2) -> o2[1] - o1[1]);
        for (int i = 0; i < nums2.length; i++) {
            queue.offer(new int[] {i, nums2[i]});
        }
        Arrays.sort(nums1);
        int left = 0, right = nums1.length - 1;
        int[] res = new int[nums1.length];
        while (!queue.isEmpty()) {
            int[] pair = queue.poll();
            int index = pair[0], val = pair[1];
            if (val < nums1[right]) {
                res[index] = nums1[right];
                right--;
            } else {
                res[index] = nums1[left];
                left++;
            }
        }
        return res;
    }

    // -------优势洗牌 << end --------

    // -------删掉一个元素以后全为1的最长数组 start >>--------

    /**
     * 给你一个二进制数组 nums ，你需要从中删掉一个元素。
     * 请你在删掉元素的结果数组中，返回最长的且只包含 1 的非空子数组的长度。
     * 如果不存在这样的子数组，请返回 0 。
     *
     * 使用滑动窗口库的方法进行解答。
     *
     * 对应 leetcode 中第 1493 题。
     */
    public int longestSubarray(int[] nums) {
        int left = 0, right = 0, len = nums.length;
        int zeroCount = 0, res = 0;
        while (right < len) {
            if (nums[right] == 0) zeroCount++;
            right++;
            while (zeroCount > 1) {
                if (nums[left] == 0) zeroCount--;
                left++;
            }
            res = Math.max(res, right - left);
        }
        return res - 1;
    }

    // -------删掉一个元素以后全为1的最长数组 << end --------

    // -------存在重复元素III start >>--------

    /**
     * 给你一个整数数组 nums 和两个整数 k 和 t 。请你判断是否存在 两个不同下标 i 和 j，使得 abs(nums[i] - nums[j]) <= t ，
     * 同时又满足 abs(i - j) <= k 。
     * 如果存在则返回 true，不存在返回 false。
     *
     * 解题思路：
     * 根据题意，对于任意一个位置 i （假设其值为 u），我们其实是希望在下标范围为 [max(0,i-k), i)内找到值范围在 [u-t,u+t]的数。
     * 我们希望使用一个【有序集合】去维护长度为 k 的滑动窗口内的数，该数据结构最好支持高效【查询】与【插入/删除】操作：
     * 【查询】：能够在【有序集合】中应用【二分查找】，快速找到【小于等于u的最大值】和【大于等于u的最小值】
     * 【插入/删除】：在往【有序集合】中添加或删除元素时，能够在低于线性的复杂度内完成。
     *
     * 而红黑树能很好的解决上面的问题，每次红黑树的平衡调整引发的旋转的次数 能够限制到【最多三次】。
     * 当【查询】动作和【插入/删除】动作频率相当时，比较好的选择是使用【红黑树】。
     *
     * 对应 leetcode 中第 220 题。
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        int len = nums.length;
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i = 0; i < len; i++) {
            int u = nums[i];
            Integer floor = ts.floor(u);
            Integer ceiling = ts.ceiling(u);
            if (floor != null && u - floor <= t) return true;
            if (ceiling != null && ceiling - u <= t) return true;
            // 将当前数加到 ts 中，并移除下标范围不在窗口中的值
            ts.add(u);
            if (i >= k) {
                ts.remove(nums[i - k]);
            }
        }
        return false;
    }

    // -------存在重复元素III << end --------

    // -------乘积小于K的子数组 start >>--------

    /**
     * 给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。
     *
     * 使用 滑动窗口的方法进行解答
     *
     * 对应 leetcode 中第 713 题。
     */
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        // 因为乘积要严格小于 k， nums[i] >= 1, 所以当k = 1时，同样返回 0
        if (k <= 1) return 0;
        int left = 0, right = 0, multi = 1, res = 0;
        while (right < nums.length) {
            multi *= nums[right];
            right++;
            while (multi >= k) {
                multi /= nums[left];
                left++;
            }
            // 每次右指针位移到一个新位置，应该加上 right - left 种数组组合
            // nums[right - 1]
            // nums[right - 2], nums[right - 1]
            // nums[left], ... nums[right - 3], nums[right - 2], nums[right - 1]
            res += right - left;
        }
        return res;
    }

    // -------乘积小于K的子数组 << end --------

    // -------跳跃游戏II start >>--------

    /**
     * 给你一个非负整数数组 nums ，你最初位于数组的第一个位置。
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
     * 假设你总是可以到达数组的最后一个位置。
     *
     * 使用 贪心 的思想进行解答。
     *
     * 对应 leetcode 中第 45 题。
     */
    public int jump(int[] nums) {
        int start = 0, end = 1, res = 0;
        while (end < nums.length) {
            int maxPos = 0;
            for (int i = start; i < end; i++) {
                maxPos = Math.max(maxPos, i + nums[i]);
            }
            start = end;
            end = maxPos + 1;
            res++;
        }
        return res;
    }

    /**
     * 从上面的代码观察发现，其实被 while 包含的 for 循环中， i 是从头跑到尾的。
     * 只需要在一次跳跃完成时，更新下一次 能跳到最远的距离。
     * 并以此刻作为时机来更新跳跃次数。 就可以在一次for 循环中处理。
     */
    public int jump_v2(int[] nums) {
        int end = 0, maxPos = 0, res = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            maxPos = Math.max(maxPos, i + nums[i]);
            if (i == end) {
                end = maxPos;
                res++;
            }
        }
        return res;
    }

    // -------跳跃游戏II << end --------

    // -------插入区间 start >>--------

    /**
     * 给你一个 无重叠的 ，按照区间起始端点排序的区间列表。
     * 在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。
     *
     * 对应 leetcode 中第 57 题。
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int[][] res = new int[intervals.length + 1][2];
        int i = 0, idx = 0;
        // 首先将新区间左边且相离的区间加入结果集
        while (i < intervals.length && intervals[i][1] < newInterval[0]) {
            res[idx++] = intervals[i++];
        }
        // 接着判断当前区间是否与新区间重叠，重叠的话就进行合并，知道遍历到当前区间在新区间的右边且相离，
        // 将最终合并后的新区间假如结果集
        while (i < intervals.length && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i++;
        }
        res[idx++] = newInterval;
        // 最后将新区间右边且相离的区间加入结果集
        while (i < intervals.length) {
            res[idx++] = intervals[i++];
        }
        return Arrays.copyOf(res, idx);
    }

    // -------插入区间 << end --------

    // -------矩阵置零 start >>--------

    /**
     * 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
     *
     * 两遍扫 matrix，第一遍用集合记录哪些行， 哪些列有 0， 第二遍置 0.
     *
     * 对应 leetcode 中第 73 题。
     */
    public void setZeroes(int[][] matrix) {
        Set<Integer> rowZero = new HashSet<>();
        Set<Integer> colZero = new HashSet<>();
        int row = matrix.length, col = matrix[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == 0) {
                    rowZero.add(i);
                    colZero.add(j);
                }
            }
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (rowZero.contains(i) || colZero.contains(j)) {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    public void setZeroes_v2(int[][] matrix) {
        int row = matrix.length, col = matrix[0].length;
        boolean firstRowHasZero = false, firstColHasZero = false;
        // 第一行是否有 0
        for (int i = 0; i < col; i++) {
            if (matrix[0][i] == 0) {
                firstRowHasZero = true;
                break;
            }
        }
        // 第一列是否有 0
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == 0) {
                firstColHasZero = true;
                break;
            }
        }
        // 将第一行第一列作为标志位
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        // 置 0
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (firstRowHasZero) {
            for (int j = 0; j < col; j++) {
                matrix[0][j] = 0;
            }
        }
        if (firstColHasZero) {
            for (int i = 0; i < row; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    // -------矩阵置零 << end --------

    // -------三角形最小路径和 start >>--------

    /**
     * 给定一个三角形 triangle ，找出自顶向下的最小路径和。
     * 每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
     * 也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
     *
     * 使用 动态规划的方法进行解答。
     * 定义 dp[i][j] 表示从点 (i,j) 到底边的最小路径和。
     * 那么有 状态转移方程： dp[i][j] = min(dp[i+1][j], dp[i+1][j+1])
     *
     * 对应 leetcode 中第 120 题。
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        int len = triangle.size();
        int[][] dp = new int[len + 1][len + 1];
        for (int i = len - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                dp[i][j] = Math.min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0][0];
    }

    // -------三角形最小路径和 << end --------

    // -------解数独 start >>--------

    final static class Sudoku {
        // 二进制中 1 表示对应位置已经有值了
        private final int[] rows = new int[9];
        private final int[] cols = new int[9];
        private final int[][] cells = new int[3][3];

        /**
         * 编写一个程序，通过填充空格来解决数独问题。
         * 数独的解法需 遵循如下规则：
         *
         * 数字 1-9 在每一行只能出现一次。
         * 数字 1-9 在每一列只能出现一次。
         * 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
         * 数独部分空格内已填入了数字，空白格用 '.' 表示。
         *
         * 对应 leetcode 中第 37 题。
         */
        public void solveSudoku(char[][] board) {
            int cnt = 0;
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[i].length; j++) {
                    char c = board[i][j];
                    if (c == '.') {
                        cnt++;
                    } else {
                        int n = c - '1';
                        solveSudokuFillNumber(i, j, n, true);
                    }
                }
            }
            solveSudokuBackTrace(board, cnt);
        }

        private boolean solveSudokuBackTrace(char[][] board, int cnt) {
            if (cnt == 0) return true;
            // 获取当前 候选项最少（即限制最多）的格子下标
            int[] pos = solveSudokuGetMinOkMaskCountPos(board);
            int x = pos[0], y = pos[1];
            // okMask 值为1的位表示对应的数字当前可以填入
            int okMask = solveSudokuGetOkMask(x, y);

            for (char c = '1'; c <= '9'; c++) {
                int index = c - '1';
                if (solveSudokuTestMask(okMask, index)) {
                    solveSudokuFillNumber(x, y, index, true);
                    board[x][y] = c;
                    if (solveSudokuBackTrace(board, cnt - 1)) return true;  // 题目假定唯一解
                    board[x][y] = '.';
                    solveSudokuFillNumber(x, y, index, false);
                }
            }
            return false;
        }

        /**
         * mask 二进制， 低 index位 是否为 1
         */
        private boolean solveSudokuTestMask(int mask, int index) {
            return (mask & (1 << index)) != 0;
        }

        private int solveSudokuGetOkMask(int x, int y) {
            return ~(rows[x] | cols[y] | cells[x/3][y/3]);
        }

        /**
         * 获取候选项最少的位置
         */
        private int[] solveSudokuGetMinOkMaskCountPos(char[][] board) {
            int[] res = new int[2];
            int min = 10;
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[i].length; j++) {
                    if (board[i][j] == '.') {
                        int okMask = solveSudokuGetOkMask(i, j);
                        int count = solveSudokuGetOneCountInMask(okMask);
                        if (count < min) {
                            min = count;
                            res[0] = i;
                            res[1] = j;
                        }
                    }
                }
            }
            return res;
        }

        /**
         * mask 二进制低 9 位中 1 的个数
         */
        private int solveSudokuGetOneCountInMask(int mask) {
            int res = 0;
            for (int i = 0; i < 9; i++) {
                int test = 1 << i;
                if ((mask & test) != 0) {
                    res++;
                }
            }
            return res;
        }

        private void solveSudokuFillNumber(int x, int y, int n, boolean fill) {
            // true set 1, false set 0
            rows[x] = fill ? rows[x] | (1 << n) : rows[x] & ~(1 << n);
            cols[y] = fill ? cols[y] | (1 << n) : cols[y] & ~(1 << n);
            cells[x / 3][y / 3] = fill ? cells[x / 3][y / 3] | (1 << n) : cells[x / 3][y / 3] & ~(1 << n);
        }
    }

    // -------解数独 << end --------

    // -------删除有序数组中的重复项II start >>--------

    /**
     * 给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 最多出现两次 ，返回删除后数组的新长度。
     * 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
     *
     * 使用 快慢指针的方式进行解答。
     *
     * 对应 leetcode 中第 80 题。
     */
    public int removeDuplicatesTwo(int[] nums) {
        int fast = 1, slow = 0, count = 1;
        while (fast < nums.length) {
            if (nums[slow] != nums[fast]) {
                slow++;
                nums[slow] = nums[fast];
                count = 1;
            } else {
                count++;
                if (count == 2) {
                    slow++;
                    nums[slow] = nums[fast];
                }
            }
            fast++;
        }
        return slow + 1;
    }

    public int removeDuplicatesTwo_v2(int[] nums) {
        return removeDuplicatesTwoProcess(nums, 2); 
    }

    private int removeDuplicatesTwoProcess(int[] nums, int k) {
        int index = 0;
        for (int num : nums) {
            if (index < k || nums[index - k] != num) {
                nums[index++] = num;
            }
        }
        return index;
    }

    // -------删除有序数组中的重复项II << end --------

    // -------加油站 start >>--------

    /**
     * 在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
     * 你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
     * 给定两个整数数组 gas 和 cost ，如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则 保证 它是 唯一 的。
     *
     * 该题可以使用 图的思想来分析，
     * 可以画一个 折线图用来表示总油量剩余值，若要满足题目的要求，跑完全程在回到起点，总油量剩余值的任意部分都需要在 X 轴以上，且跑到终点时，
     * 总剩余量 >= 0.
     * 为了让黑色折线图任意部分都在 X 轴以上，我们需要向上移动黑色折线图，直到所有的点都在 X 轴或者 X轴 以上。此时，处在 X轴的点即为出发点。
     * 也即黑色折线图的最低值的位置。
     *
     * 对应 leetcode 中第 134 题。
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int len = gas.length, spare = 0, minSpare = Integer.MAX_VALUE;
        int res = 0;
        for (int i = 0; i < len; i++) {
            spare += gas[i] - cost[i];
            if (spare < minSpare) {
                res = i;
                minSpare = spare;
            }
        }
        return spare < 0 ? -1 : (res + 1) % len;
    }

    // -------加油站 << end --------

    // -------分发糖果 start >>--------

    /**
     * n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。
     * 你需要按照以下要求，给这些孩子分发糖果：
     * 每个孩子至少分配到 1 个糖果。
     * 相邻两个孩子评分更高的孩子会获得更多的糖果。
     * 请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。
     *
     * 解题思路：
     * 规则定义： 设学生 A 和学生 B左右相邻，A在B左边；
     *  左规则：当 ratings(B) > ratings(A) 时，B的糖比A 的糖数量多。
     *  右规则：当 ratings(A) > ratings(B) 时，A的糖比B的糖数量多。
     *
     * 算法流程：
     * 1.先从左至右遍历学生成绩 ratings， 按照以下规则给糖，并记录在 left 中，
     *   1.先给所有学生 1 颗糖；
     *   2.若 ratings(i) > ratings(i - 1)， 则第 i 名学生糖比第 i - 1名学生多1个
     *   3.若 ratings(i) <= ratings(i - 1),则第 i 名学生糖不变。（交由从右向左遍历时处理）
     *  经过此规则后，可以保证所有学生的糖数量满足左规则。
     * 2.同理，在此规则下从右至左遍历学生成绩并记录在 right 中，可以保证所有学生糖数量满足右规则。
     * 3.最终，取以上2轮遍历 left 和 right 对应学生糖果数的最大值，这样则同时满足左规则和右规则，即得到每个同学的最少糖果数量。
     *
     * 对应 leetcode 中第 135 题。
     */
    public int candy(int[] ratings) {
        int[] left = new int[ratings.length], right = new int[ratings.length];
        Arrays.fill(left, 1);
        Arrays.fill(right, 1);
        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                left[i] = left[i - 1] + 1;
            }
        }
        for (int j = ratings.length - 2; j >= 0; j--) {
            if (ratings[j] > ratings[j + 1]) {
                right[j] = right[j + 1] + 1;
            }
        }
        int res = 0;
        for (int i = 0; i < ratings.length; i++) {
            res += Math.max(left[i], right[i]);
        }
        return res;
    }

    // -------分发糖果 << end --------

    // -------逆波兰表达式求值 start >>--------

    /**
     * 根据 逆波兰表达式， 求表达式的值。
     * 有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
     * 注意 两个整数之间的除法只保留整数部分。
     * 可以保证给定的逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
     *
     * 对应 leetcode 中第 137 题。
     */
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        Set<String> set = Stream.of("+", "-", "*", "/").collect(Collectors.toSet());
        for (String token : tokens) {
            if (set.contains(token)) {
                Integer p1 = stack.pop();
                Integer p2 = stack.pop();
                switch (token) {
                    case "+":
                        stack.push(p1 + p2);
                        break;
                    case "-":
                        stack.push(p2 - p1);
                        break;
                    case "*":
                        stack.push(p1 * p2);
                        break;
                    case "/":
                        stack.push(p2 / p1);
                        break;
                    default:
                        break;
                }
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.peek();
    }

    // -------逆波兰表达式求值 << end --------

    // -------前K个高频元素 start >>--------

    /**
     * 给你一个整数数组 nums 和一个整数 k，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
     *
     * 使用 最小堆的 方法进行解答。
     *
     * 时间复杂度： o(nlogK), 空间复杂度： o(n)
     *
     * 对应 leetcode 中第 336 题。
     */
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Integer> queue = new PriorityQueue<>(Comparator.comparingInt(map::get));
        for (Integer key : map.keySet()) {
            if (queue.size() < k) {
                queue.offer(key);
            } else if (map.get(key) > map.get(queue.peek())) {
                queue.poll();
                queue.offer(key);
            }
        }
        int[] res = new int[k];
        int i = 0;
        while (!queue.isEmpty()) {
            res[i++] = queue.poll();
        }
        return res;
    }

    /**
     * 使用 桶排序的方法进行解答。
     */
    public int[] topKFrequent_v2(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        List<Integer>[] list = new List[nums.length + 1];
        for (Integer key : map.keySet()) {
            int i = map.get(key);
            if (list[i] == null) {
                list[i] = new ArrayList<>();
            }
            list[i].add(key);
        }
        int[] res = new int[k];
        int index = k;
        for (int i = list.length - 1; i >= 0 && index > 0; i--) {
            if (list[i] != null) {
                for (Integer num : list[i]) {
                    res[--index] = num;
                }
            }
        }
        return Arrays.copyOfRange(res, index, k);
    }

    // -------前K个高频元素 << end --------

    // -------按要求补齐数组 start >>--------

    /**
     * 给定一个已排序的正整数数组 nums ，和一个正整数 n 。从 [1, n] 区间内选取任意个数字补充到 nums 中，使得 [1, n] 区间内的任何数字都可以用 nums 中某几个数字的和来表示。
     * 请返回 满足上述要求的最少需要补充的数字个数 。
     *
     * 使用 贪心的方法进行解答。
     * 首先，题目中说数组是排序的。假设数组中前 k 个数字能组成的数字范围是 [1, total]，当我们添加数组中第 k + 1 个数字 nums[k] 的时候，范围就变成了
     * [1, total] U [1 + nums[k], total + nums[k]] U [nums[k], nums[k]], 这是一个并集，可以合并成 [1, total] U [nums[k], total + nums[k]]。
     * 我们仔细观察一下：
     * 1、如果左边的 total < nums[k] - 1,那么他们中间肯定会有空缺，不能构成完整的 [1, tatal + nums[k]]。这个时候我们需要添加一个数字
     * total + 1。先构成一个更大的范围 [1, total * 2 + 1]。这里为什么是添加 total + 1而不是添加 total，举个例子，比如可以构成的数字范围是
     * [1,5], 如果需要添加一个构成更大范围的，我们应该选择 6 而不是 5.
     * 2、如果左边的 total >= nums[k] - 1, 那么就可以构成完整的 [1, total + nums[k]],就不需要在添加数字了。
     *
     * 对应 leetcode 中第 330 题。
     */
    public int minPatches(int[] nums, int n) {
        int total = 0, index = 0, count = 0;
        while (total < n) {
            if (index < nums.length && nums[index] <= total + 1) {
                // 如果数组能够组成的数字范围是 [1, total], 那么加上 nums[index]
                // 就变成了 [1, total] U [nums[index], total + nums[index]]
                // 这种情况下结果就是 [1, total + nums[index]]
                total += nums[index++];
            } else {
                total += (total + 1);
                count++;
            }
        }
        return count;
    }

    /**
     * 上面组成数字的范围是闭区间，我们还可以改成开区间[1,total), 原理都一样，稍作修改即可。
     */
    public int minPatches_v2(int[] nums, int n) {
        int total = 1, index = 0, count = 0;
        while (total <= n) {
            if (index < nums.length && nums[index] <= total) {
                // 如果数组能组成的数字范围是 [1,total),那么加上 nums[index]
                // 就变成了 [1,total) U [nums[index], total + nums[index])
                // 结果就是 [1, total + nums[index])
                total += nums[index++];
            } else {
                // 添加一个新数字，并且 count + 1
                total <<= 1;
                count++;
            }
        }
        return count;
    }

    // -------按要求补齐数组 << end --------

    // -------摆动序列 start >>--------

    /**
     * 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。
     * 例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。
     * 相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
     * 子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。
     * 给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。
     *
     * 采用动态规划的方法进行解答：
     * 我们可以先进行一些约定：
     * 1、某个序列被称为【上升摆动序列】，当且仅当该序列是摆动序列，且最后一个元素呈上升趋势。如序列 [1,3,2,4] 即为【上升摆动序列】。
     * 2、某个序列被称为【下降摆动序列】，当且仅当该序列是摆动序列，且最后一个元素呈下降趋势。如序列 [4,2,3,1] 即为【下降摆动序列】。
     *
     * 每当我们选择一个元素作为摆动序列的一部分时，这个元素要么是上升的，要么是下降的，这取决于前一个元素的大小。那么列出状态表达式为：
     * 1、up[i]表示以前 i 个元素中的某一个为结尾的最长的 【上升摆动序列】的长度。
     * 2、down[i]表示以前 i 个元素中的某一个为结尾的最长的【下降摆动序列】的长度。
     * 下面以 up[i]为例，说明其状态转移规则：
     * 1、当 nums[i] <= nums[i - 1]时，我们无法选出更长的【上升摆动序列】的方案。因为对于任何以 nums[i] 结尾的【上升摆动序列】，
     *   我们都可以将 nums[i] 替换为 nums[i - 1]，使其成为以 nums[i - 1] 结尾的【上升摆动序列】。
     * 2、当 nums[i] > nums[i - 1]时，我们既可以从 up[i - 1] 进行转移，也可以从 down[i - 1] 进行转移。下面我们证明从 down[i - 1]
     * 转移时必然合法的，即必然存在一个 down[i - 1] 对应的最长的 【下降摆动序列】的末尾元素小于 nums[i]。
     *   a、我们记这个末尾元素在原序列中的下标为 j，假设从 j 往前的第一个【谷】为 nums[k],我们总可以让 j 移动到 k，使得这个最长的【下降摆动序列】
     *   的末尾元素为 【谷】。
     *   b、然后我们可以证明在这个末尾元素为 【谷】的情况下，这个末尾元素必然是从 nums[i] 往前的第一个 【谷】。证明非常简单，我们使用反证法，
     *   如果这个末尾元素不是从 nums[i] 往前的第一个【谷】，那么我们总可以在末尾元素和 nums[i]之间取得一对 【峰】与【谷】，接在这个【下降摆动序列】
     *   后，使其更长。
     *   c、这样我们知道必然存在一个 down[i - 1]对应的最长的【下降摆动序列】的末尾元素Wie nums[i] 往前的第一个 【谷】，这个【谷】
     *   必然小于 nums[i]。 证毕。
     *
     * 这样我们可以用同样的方法说明 down[i]的状态转移规则，最终的状态转移方程为：
     *         | up[i - 1],                          nums[i] <= nums[i - 1]
     * up[i] = |
     *         | max(up[i - 1], down[i - 1] + 1),    nums[i] > nums[i - 1]
     *
     *           | down[i - 1],                        nums[i] >= nums[i - 1]
     * down[i] = |
     *           | max(down[i - 1], up[i - 1] + 1),    nums[i] < nums[i - 1]
     *
     * 最终的答案就是 up[n - 1] 和 down[n - 1]中的较大值，其中 n 是序列的长度。
     *
     * 对应  leetcode 中第 376 题。
     */
    public int wiggleMaxLength(int[] nums) {
        int n = nums.length;
        if (n < 2) {
            return n;
        }
        int[] up = new int[n], down = new int[n];
        up[0] = down[0] = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] > nums[i - 1]) {
                up[i] = Math.max(up[i - 1], down[i - 1] + 1);
                down[i] = down[i - 1];
            } else if (nums[i] < nums[i - 1]) {
                up[i] = up[i - 1];
                down[i] = Math.max(up[i - 1] + 1, down[i - 1]);
            } else {
                up[i] = up[i - 1];
                down[i] = down[i - 1];
            }
        }
        return Math.max(up[n - 1], down[n - 1]);
    }

    /**
     * 上面方法的 压缩版
     */
    public int wiggleMaxLength_v2(int[] nums) {
        int n = nums.length;
        if (n < 2) return n;
        int up = 1, down = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] > nums[i - 1]) {
                up = Math.max(up, down + 1);
            } else if (nums[i] < nums[i - 1]) {
                down = Math.max(down, up + 1);
            }
        }
        return Math.max(up, down);
    }

    /**
     * 使用 贪心的方法进行解答。
     *
     * 观察这个序列可以发现，我们不断的交错选择【峰】与【谷】，可使得该序列尽可能长
     */
    public int wiggleMaxLength_v3(int[] nums) {
        int n = nums.length;
        if (n < 2) return n;
        int prevDiff = nums[1] - nums[0];
        int ret = prevDiff != 0 ? 2 : 1;
        for (int i = 2; i < n; i++) {
            int diff = nums[i] - nums[i - 1];
            if ((diff > 0 && prevDiff <= 0) || (diff < 0 && prevDiff >= 0)) {
                ret++;
                prevDiff = diff;
            }
        }
        return ret;
    }

    // -------摆动序列 << end --------

    // -------用最少数量的箭引爆气球 start >>--------

    /**
     * 有一些球形气球贴在一堵用 XY 平面表示的墙面上。墙面上的气球记录在整数数组 points ，其中points[i] = [xstart, xend] 表示水平直径在
     *  xstart 和 xend之间的气球。你不知道气球的确切 y 坐标。
     *
     * 一支弓箭可以沿着 x 轴从不同点 完全垂直 地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend，
     * 且满足  xstart ≤ x ≤ xend，则该气球会被 引爆 。可以射出的弓箭的数量 没有限制 。 弓箭一旦被射出之后，可以无限地前进。
     * 给你一个数组 points ，返回引爆所有气球所必须射出的 最小 弓箭数 
     *
     * 对应 leetcode 中第 452 题。
     */
    public int findMinArrowShots(int[][] points) {
        if (points.length == 0) return 0;
        Arrays.sort(points, Comparator.comparingInt(p -> p[1]));
        int res = 1;
        int pre = points[0][1];
        for (int i = 1; i < points.length; i++) {
            if (points[i][0] > pre) {
                res++;
                pre = points[i][1];
            }
        }
        return res;
    }

    // -------用最少数量的箭引爆气球 << end --------

    // -------有效三角形的个数 start >>--------

    /**
     * 给定一个包含非负整数的数组 nums ，返回其中可以组成三角形三条边的三元组个数。
     *
     * 采用固定前两个位置，后一个位置采用二分查找的方法，
     * 时间复杂度：O(n2logn)
     *
     * 对应 leetcode 中第 611 题。
     */
    public int triangleNumber(int[] nums) {
        if (nums.length < 3) return 0;
        Arrays.sort(nums);
        int res = 0;
        for (int i = 0; i < nums.length - 2; i++) {
            for (int j = i + 1; j < nums.length - 1; j++) {
                int sum = nums[i] + nums[j];
                int left = j + 1, right = nums.length;
                while (left < right) {
                    int middle = left + (right - left) / 2;
                    if (nums[middle] < sum) {
                        left = middle + 1;
                    } else {
                        right = middle;
                    }
                }
                res += left - j - 1;
            }
        }
        return res;
    }

    /**
     * 采用双指针的方法进行解答。
     *
     * 首先对数组排序
     * 固定最长的一条边，运用双指针扫描
     *  1、如果 nums[left] + nums[right] > nums[i],同时说明 nums[left + 1] + nums[right] > nums[i], ...
     *  nums[right - 1] + nums[right] > nums[i],满足条件的有 right - left种，right左移进入下一轮。
     *  2、如果 nums[left] + nums[right] <= nums[i], left 右移进入下一轮
     */
    public int triangleNumber_v2(int[] nums) {
        Arrays.sort(nums);
        int res = 0;
        for (int i = nums.length - 1; i >= 2; i--) {
            int left = 0, right = i - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[i]) {
                    res += right - left;
                    right--;
                } else {
                    left++;
                }
            }
        }
        return res;
    }

    // -------有效三角形的个数 << end --------

    // -------任务调度器 start >>--------

    /**
     * 给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1
     * 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。
     * 然而，两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。
     * 你需要计算完成所有任务所需要的 最短时间 。
     *
     * 对应 leetcode 中第 621 题。
     */
    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> map = new HashMap<>();
        int maxCount = 0, maxTaskCount = 0;
        for (char task : tasks) {
            map.put(task, map.getOrDefault(task, 0) + 1);
            maxCount = Math.max(maxCount, map.get(task));
        }
        for (Integer value : map.values()) {
            if (maxCount == value) {
                maxTaskCount++;
            }
        }
        return Math.max(tasks.length, (maxCount - 1) * (n + 1) + maxTaskCount);
    }

    // -------任务调度器 << end --------

    // -------分割数组为连续子序列 start >>--------

    /**
     * 给你一个按升序排序的整数数组 num（可能包含重复数字），请你将它们分割成一个或多个长度至少为 3 的子序列，其中每个子序列都由连续整数组成。
     * 如果可以完成上述分割，则返回 true ；否则，返回 false 。
     *
     * 对应 leetcode 中第 659 题。
     */
    public boolean isPossible(int[] nums) {
        Map<Integer, Integer> countMap = new HashMap<>();
        for (int num : nums) {
            countMap.put(num, countMap.getOrDefault(num, 0) + 1);
        }
        // 定义一个哈希表记录最长的子序列
        Map<Integer, Integer> tail = new HashMap<>();
        for (int num : nums) {
            Integer count = countMap.getOrDefault(num, 0);
            if (count <= 0) {
                // 当前元素已经用完，直接跳过
                continue;
            } else if (tail.getOrDefault(num - 1, 0) > 0) {
                countMap.put(num, count - 1);
                tail.put(num - 1, tail.get(num - 1) - 1); // 覆盖当前最长的子序列
                tail.put(num, tail.getOrDefault(num, 0) + 1);  // 当前以num结尾的子序列 加1.
            } else if (countMap.getOrDefault(num + 1, 0) > 0 && countMap.getOrDefault(num + 2, 0) > 0) {
                countMap.put(num, count - 1);
                countMap.put(num + 1, countMap.get(num + 1) - 1);
                countMap.put(num + 2, countMap.get(num + 2) - 1);
                tail.put(num + 2, countMap.getOrDefault(num + 2, 0) + 1);
            } else {
                return false;
            }
        }
        return true;
    }

    // -------分割数组为连续子序列 << end --------

    // -------矩阵中的最长递增路径 start >>--------

    /**
     * 给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。
     * 对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。
     *
     * 使用带记忆化的深度优先搜索的方法进行解答。
     *
     * 对应 leetcode 中第 329 题。
     */
    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return 0;
        int row = matrix.length, col = matrix[0].length;
        int[][] memo = new int[row][col];
        int res = 1;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                res = Math.max(res, longestIncreasingPath_backTrack(matrix, i, j, memo));
            }
        }
        return res;
    }

    private int longestIncreasingPath_backTrack(int[][] matrix, int i, int j, int[][] memo) {
        if (memo[i][j] != 0) {
            return memo[i][j];
        }
        ++memo[i][j];
        int row = matrix.length, col = matrix[0].length;
        int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int[] dir : dirs) {
            int newRow = i + dir[0], newCol = j + dir[1];
            if (newRow >= 0 && newRow < row &&
                    newCol >= 0 && newCol < col &&
                    matrix[newRow][newCol] > matrix[i][j]) {
                memo[i][j] = Math.max(memo[i][j], longestIncreasingPath_backTrack(matrix, newRow, newCol, memo) + 1);
            }
        }
        return memo[i][j];
    }


    // -------矩阵中的最长递增路径 << end --------

    // -------求众数 start >>--------

    /**
     * 给定一个大小为 n 的整数数组，找出其中所有出现超过 【 n/3 】 次的元素。
     *
     * 使用 摩尔投票法 进行解答。
     * 归纳如下：
     * 如果至多选一个代表，那他的票数至少要超过一半 【1/2】的数
     * 如果至多选两个代表，那它的票数至少要超过 【1/3】 的数
     * 如果至多选 m 个代表，那他的票数至少要超过 【1/(m+1)】 的票数
     *
     * 对应 leetcode 中第 229 题。
     */
    public List<Integer> majorityElement(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (null == nums || nums.length == 0) return res;
        int can1 = nums[0], count1 = 0;
        int can2 = nums[0], count2 = 0;
        // 摩尔投票法，分为两个阶段，配对阶段和计数阶段
        for (int num : nums) {
            if (can1 == num) {
                count1++;
                continue;
            }
            if (can2 == num) {
                count2++;
                continue;
            }
            if (count1 == 0) {
                can1 = num;
                count1++;
                continue;
            }
            if (count2 == 0) {
                can2 = num;
                count2++;
                continue;
            }
            count1--;
            count2--;
        }

        // 计数阶段
        count1 = 0;
        count2 = 0;
        for (int num : nums) {
            if (can1 == num) count1++;
            else if (can2 == num) count2++;
        }
        if (count1 > nums.length / 3) res.add(can1);
        if (count2 > nums.length / 3) res.add(can2);
        return res;
    }

    // -------求众数 << end --------

    // -------最接近的三数之和 start >>--------

    /**
     * 给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。
     * 返回这三个数的和。
     * 假定每组输入只存在恰好一个解。
     *
     * 对应 leetcode 中第 16 题。
     */
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length - 2; i++) {
            int start = i + 1, end = nums.length - 1;
            while (start < end) {
                int num = nums[i] + nums[start] + nums[end];
                if (Math.abs(num - target) < Math.abs(target - res)) {
                    res = num;
                }
                if (num > target) {
                    end--;
                } else if (num < target) {
                    start++;
                } else return num;
            }
        }
        return res;
    }

    // -------最接近的三数之和 << end --------

    // -------数据流中的第K大元素 start >>--------

    /**
     * 设计一个找到数据流中第 k 大元素的类（class）。注意是排序后的第 k 大元素，不是第 k 个不同的元素。
     */
    static final class KthLargest {
        private final int k;
        private final PriorityQueue<Integer> pq = new PriorityQueue<>();

        public KthLargest(int k, int[] nums) {
            this.k = k;
            for (int i = 0; i < k; i++) {
                pq.offer(nums[i]);
            }
            for (int i = k; i < nums.length; i++) {
                if (nums[i] > pq.peek()) {
                    pq.poll();
                    pq.offer(nums[i]);
                }
            }
        }

        public int add(int val) {
            // 维护小顶堆只保留前 k 大的元素
            pq.offer(val);
            if (pq.size() > k) {
                pq.poll();
            }
            return pq.peek();
        }
    }

    // -------数据流中的第K大元素 << end --------

    // -------验证栈序列 start >>--------

    /**
     * 给定 pushed 和 popped 两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push
     * 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false 。
     *
     * 采用模拟的方法进行解答。
     *
     * 对应 leetcode 中第 946 题。
     */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int i = 0, j = 0, len = pushed.length;
        while (j < len) {
            while (i < len && pushed[i] != popped[j]) {
                stack.push(pushed[i]);
                i++;
            }
            if (i < len) {
                stack.push(pushed[i]);
                i++;
            }
            while (!stack.isEmpty() && popped[j] == stack.peek()) {
                stack.pop();
                j++;
            }
            if (i == len) break;
        }
        return j == len;
    }

    public boolean validateStackSequences_v2(int[] pushed, int[] popped) {
        Deque<Integer> stack = new ArrayDeque<>();
        int len = pushed.length;
        for (int i = 0, j = 0; i < len; i++) {
            stack.push(pushed[i]);
            while (!stack.isEmpty() && stack.peek() == popped[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }

    // -------验证栈序列 << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------
}
