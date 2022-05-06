package com.study.leetcode;

import java.util.*;

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
     * 根据题意，对于任意一个位置 i （假设其值为 u），我们其实是希望在下表范围为 [max(0,i-k), i)内找到值范围在 [u-t,u+t]的数。
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
            if (i > k) {
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

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------
}
