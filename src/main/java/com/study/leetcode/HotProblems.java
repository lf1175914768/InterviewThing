package com.study.leetcode;

import java.util.*;

/**
 * 热题
 */
public class HotProblems {

    // -------二分搜索 start >>--------

    /**
     * 普通的二分搜索
     */
    public int binarySearch(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {   // 注意这里的 小于等于
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    public int binarySearch_v2(int[] nums, int target) {
        int left = 0, right = nums.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (left >= nums.length) {
            return -1;
        }
        return nums[left] == target ? left : -1;
    }

    /**
     * 搜索最左侧的下标
     */
    public int binaryLeftSearch(int[] nums, int target) {
        int left = 0, right = nums.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                // 包括 nums[mid] == target 和 nums[mid] > target
                right = mid;
            }
        }
        if (left >= nums.length) {
            return -1;
        }
        return nums[left] == target ? left : -1;
    }

    public int binaryLeftSearch_v2(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                // 包括 nums[mid] == target 和 nums[mid] > target
                right = mid - 1;
            }
        }
        if (left >= nums.length) {
            return -1;
        }
        return nums[left] == target ? left : -1;
    }

    /**
     * 搜索最右侧的下标
     */
    public int binaryRightSearch(int[] nums, int target) {
        int left = 0, right = nums.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (left - 1 <= 0)
            return -1;
        return nums[left - 1] == target ? left - 1 : -1;
    }

    public int binaryRightSearch_v2(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (left - 1 <= 0)
            return -1;
        return nums[left - 1] == target ? left - 1 : -1;
    }

    // -------二分搜索 << end --------

    // -- 最长回文子串 start >> --
    public String longestPalindrome(String s) {
        if (null == s || s.length() == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder(s);
        char[] reversedS = sb.reverse().toString().toCharArray();
        char[] charS = s.toCharArray();
        int length = s.length(), maxLength = 0, maxEnd = 0;
        int[][] arr = new int[length][length];
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < length; j++) {
                if (charS[i] == reversedS[j]) {
                    if (i == 0 || j == 0) {
                        arr[i][j] = 1;
                    } else {
                        arr[i][j] = 1 + arr[i - 1][j - 1];
                    }
                }
                // 求出最长公共子串后，并不一定是回文串， 还需要判断该字符串倒置前的下标和当前的字符串下标是否匹配
                /*
                 * 比如 origin = “caba”， reverse = “abac”
                 * 首先i，j始终指向子串的末尾字符，所以j指向的红色的 a 倒置前的下标是 beforeRev = length - 1 - j = 4 - 1 - 2 = 1,
                 * 对应的是字符串首位的下标，我们需要加上 maxLength 才是末尾字符的下标, 也就是 beforeRev + arr[i][j] - 1 = 1 + 3 - 1 = 3，
                 * 因为 arr[i][j] 保存的就是当前子串的长度。 此时再让它与 i 进行比较，如果相等，则说明他是我们要找的回文串
                 */
                if (arr[i][j] > maxLength) {
                    int beforeRev = length - 1 - j;
                    if (beforeRev + arr[i][j] - 1 == i) {
                        // 判断下标是否相等
                        maxEnd = i;
                        maxLength = arr[i][j];
                    }
                }
            }
        }
        return s.substring(maxEnd - maxLength + 1, maxEnd + 1);
    }

    public String longestPalindrome_v2(String s) {
        if (null == s || s.length() == 0) {
            return "";
        }
        String reverse = new StringBuilder(s).reverse().toString();
        int length = s.length();
        int[] arr = new int[length];
        int maxLength = 0, maxEnd = 0;
        for (int i = 0; i < length; i++) {
            for (int j = length - 1; j >= 0; j--) {
                if (s.charAt(i) == reverse.charAt(j)) {
                    if (i == 0 || j == 0) {
                        arr[j] = 1;
                    } else {
                        arr[j] = arr[j - 1] + 1;
                    }
                } else {
                    // 之前是二维数组，每次用的是不同的列， 所以不用置0
                    arr[j] = 0;
                }
                if (arr[j] > maxLength) {
                    int beforeRev = length - 1 - j;
                    if (beforeRev + arr[j] - 1 == i) {
                        maxLength = arr[j];
                        maxEnd = i;
                    }
                }
            }
        }
        return s.substring(maxEnd - maxLength + 1, maxEnd + 1);
    }

    public String longestPalindrome_v3(String s) {
        if (null == s || s.length() == 0)
            return "";
        int start = 0,end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = longestPalindromeExpandAroundCenter(s, i, i);
            int len2 = longestPalindromeExpandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int longestPalindromeExpandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }

    /**
     * Manacher's Algorithm 马拉车算法
     * <p>
     * 马拉车算法 是用来查找一个字符串的最长回文子串的线性方法， 由一个叫 manacher 的人再1975年发明的， 这个方法的最大贡献在于将时间复杂度提升到了线性。
     * <p>
     * 首先我们解决下奇数和偶数的问题，在每个字符间插入 “#”， 并且为了是的扩展的过程中，到边界后自动结束，在两端分别插入 “^” 和 “$”，两个不可能
     * 在字符串中出现的字符，这样中心扩展的时候，判断两端字符是否相等的时候，如果到了边界就一定会不相等，从而出了循环。经过处理，字符串的长度永远都是奇数。
     * <p>
     * ^ a # b # c # b # a # a # d # e $
     * <p>
     * 首先我们用一个数组 P 保存从中心扩展的最大个数，而它刚好也是去掉 "#"的原字符串的总长度。例如下图中下标是6的地方，可以看到 P[6] 等于5， 所以它是从左边扩展
     * 5个字符，相应的右边也是扩展 5个字符，也就是 “#c#b#c#b#c#”。 而去掉#恢复到原来的字符串，变成 “cbcbc”， 它的长度刚好也就是5。
     * 其中，T: 处理后的数组；    P: 从中心扩展的长度
     * <p>
     * 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
     * T = ^ # c # b # c # b # c  #  c  #  d  #  e  #  $
     * P = 0 0 1 0 3 0 5 0 3 0 1  2  1  0  1  0  1  0  0
     *
     * <p> 求原字符串下标 </p>
     * 用 P 的下标 i 减去 P[i]， 在除以2，就是原字符串的开头下标了。
     * 例如我们找到 P[i] 的最大值为5，也就是回文串的最大长度是5，对应的下标是6，所以原字符串的开头下标是 (6 - 5) / 2 = 0.所以我们就只需要返回原字符串的
     * 第 0 到 第 (5-1) 位就可以了。
     *
     * @param s
     * @return
     */
//    public String longestPalindrome_v3(String s) {
//        String T = preProcessForPalindrome(s);
//        int n = T.length();
//        int[] P = new int[n];
//        int C = 0, R = 0;
//        for (int i = 1; i < n - 1; i++) {
//            int i_mirror = 2 * C - i;
//
//        }
//    }
    private String preProcessForPalindrome(String s) {
        int n = s.length();
        if (n == 0) {
            return "^$";
        }
        StringBuilder ret = new StringBuilder("^");
        for (int i = 0; i < n; i++) {
            ret.append("#").append(s.charAt(i));
        }
        ret.append("#$");
        return ret.toString();
    }

    // -- 最长回文子串 << end --

    // -------正则表达式匹配 start >>--------

    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;

        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    f[i][j] = f[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    private boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    // -------正则表达式匹配 << end --------

    // -------盛最多水的容器 start >>--------

    /**
     * 双指针法
     *
     * @param height arr of height
     * @return result
     */
    public int maxArea(int[] height) {
        if (null == height || height.length == 0) {
            return 0;
        }
        int length = height.length, i = 0, j = length - 1;
        int result = 0;
        while (i < j) {
            int min = Math.min(height[i], height[j]);
            result = Math.max(result, (j - i) * min);
            if (height[i] < height[j]) {
                i++;
            } else {
                j--;
            }
        }
        return result;
    }

    // -------盛最多水的容器 << end --------

    // -------三数之和 start >>--------

    /**
     * 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
     * 注意：答案中不可以包含重复的三元组。
     *
     * 本题的难点在于如何去除重复解。
     * 1、特判，对于数组长度 n，如果数组为null 或者数组长度小于3，返回 [].
     * 2、对数组进行排序
     * 3、遍历排序后数组：
     *     若 nums[i] > 0: 因为已经排好序，所以后面不可能有三个数加和等于0，直接返回结果。
     *     对于重复元素：跳过，避免出现重复解
     *     令左指针 left = i + 1， 右指针 right = n - 1， 当 left < right 时，执行循环：
     *        当 nums[i] + nums[left] + nums[right] == 0, 执行循环，判断左界和右界是否和下一位置重复，去除重复解。
     *        并同时将 left，right 移到下一位置，寻找新得解
     *        若和大于0，说明 nums[right] 太大，right 左移
     *        若和小于0，说明 nums[left] 太小，left 右移
     *
     * 对应 leetcode 中第 15 题
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (null == nums || nums.length < 3) {
            return result;
        }

        Arrays.sort(nums);
        if (nums[0] > 0 || nums[nums.length - 1] < 0) {
            return result;
        }
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (nums[i] > 0) {
                return result;
            }

            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int curr = nums[i];
            int left = i + 1, right = len - 1;
            while (left < right) {
                int temp = curr + nums[left] + nums[right];
                if (temp == 0) {
                    List<Integer> list = new ArrayList<>();
                    list.add(curr);
                    list.add(nums[left]);
                    list.add(nums[right]);
                    result.add(list);
                    while (left < right && nums[left + 1] == nums[left]) left++;
                    while (left < right && nums[right - 1] == nums[right]) right--;
                    left++;
                    right--;
                } else if (temp < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;
    }

    // -------三数之和 << end --------

    // -------电话号码的字母组合 start >>--------

    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<>();
        if (digits == null || digits.length() == 0) {
            return combinations;
        }

        HashMap<Character, String[]> map = new HashMap<Character, String[]>() {{
            put('2', new String[]{"a", "b", "c"});
            put('3', new String[]{"d", "e", "f"});
            put('4', new String[]{"g", "h", "i"});
            put('5', new String[]{"j", "k", "l"});
            put('6', new String[]{"m", "n", "o"});
            put('7', new String[]{"p", "q", "r", "s"});
            put('8', new String[]{"t", "u", "v"});
            put('9', new String[]{"w", "x", "y", "z"});
        }};

        Queue<String> queue = new LinkedList<>();
        for (int i = 0; i < digits.length(); i++) {
            queueLetterCombinations(queue, map.get(digits.charAt(i)));
        }

        combinations.addAll(queue);
        return combinations;
    }

    private void queueLetterCombinations(Queue<String> queue, String[] letters) {
        if (queue.size() == 0) {
            queue.addAll(Arrays.asList(letters));
        } else {
            // 记录本次需要进行出队列组合的元素数量
            int queueLength = queue.size();
            for (int i = 0; i < queueLength; i++) {
                String s = queue.poll();
                for (String letter : letters) {
                    queue.add(s + letter);
                }
            }
        }
    }

    // -------电话号码的字母组合 << end --------

    // -------有效的括号 start >>--------

    /**
     * 使用栈来实现，
     *
     * 对应 leetcode 中第 20 题。
     */
    public boolean isValid(String s) {
        if (null == s || s.length() == 0) {
            return true;
        }
        Stack<Character> stack = new Stack<>();
        int i = 0, len = s.length();
        Map<Character, Character> map = new HashMap<>();
        map.put('(', ')');
        map.put('{', '}');
        map.put('[', ']');
        while (i < len) {
            Character c = s.charAt(i);
            i++;
            if (!stack.isEmpty() && !map.containsKey(c)) {
                Character pop = stack.pop();
                if (!c.equals(map.get(pop))) {
                    return false;
                }
                continue;
            }
            stack.add(c);
        }
        return stack.isEmpty();
    }

    // -------有效的括号 << end --------

    // -------括号生成 start >>--------

    public List<String> generateParenthesis(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<List<String>> list = new ArrayList<>();
        List<String> list0 = new ArrayList<>();
        list0.add("");
        list.add(list0);
        List<String> list1 = new ArrayList<>();
        list1.add("()");
        list.add(list1);

        for (int i = 2; i <= n; i++) {
            List<String> temp = new ArrayList<>();
            for (int j = 0; j < i; j++) {
                List<String> str1 = list.get(j);
                List<String> str2 = list.get(i - 1 - j);
                for (String s1 : str1) {
                    for (String s2 : str2) {
                        String candidate = "(" + s1 + ")" + s2;
                        temp.add(candidate);
                    }
                }
            }
            list.add(temp);
        }
        return list.get(n);
    }

    public List<String> generateParenthesis_v2(int n) {
        List<String> ans = new ArrayList<>();
        if (n == 0) {
            return ans;
        }

        StringBuilder path = new StringBuilder();
        dfs(path, n, n, ans);
        return ans;
    }

    private void dfs(StringBuilder path, int left, int right, List<String> res) {
        if (left == 0 && right == 0) {
            // path.toString() 生成了一个新的字符串，相当于做了一次拷贝
            res.add(path.toString());
            return;
        }

        // 剪枝
        if (left > right) {
            return;
        }

        if (left > 0) {
            path.append("(");
            dfs(path, left - 1, right, res);
            path.deleteCharAt(path.length() - 1);
        }

        if (right > 0) {
            path.append(")");
            dfs(path, left, right - 1, res);
            path.deleteCharAt(path.length() - 1);
        }
    }

    // -------括号生成 << end --------

    // -------合并K个升序链表 start >>--------

    /**
     * 使用优先级队列进行实现
     *
     * @param lists list of node
     * @return node of head
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (null == lists || lists.length == 0) {
            return null;
        }
        Queue<ListNode> queue = new PriorityQueue<>(lists.length, Comparator.comparingInt(o -> o.val));
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        for (ListNode node : lists) {
            if (node != null) {
                queue.add(node);
            }
        }
        while (!queue.isEmpty()) {
            p.next = queue.poll();
            p = p.next;
            if (p.next != null) {
                queue.add(p.next);
            }
        }
        return dummy.next;
    }

    /**
     * 使用分而治之对的思想解决
     *
     * @param lists list of list node
     * @return head of list
     */
    public ListNode mergeKLists_v2(ListNode[] lists) {
        if (null == lists || lists.length == 0) return null;
        return merge(lists, 0, lists.length - 1);
    }

    private ListNode merge(ListNode[] lists, int left, int right) {
        if (left >= right) {
            return lists[left];
        }
        int middle = left + (right - left) / 2;
        ListNode l1 = merge(lists, left, middle);
        ListNode l2 = merge(lists, middle + 1, right);

        return mergeTwoLists(l1, l2);
    }

    private ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                tail.next = l1;
                l1 = l1.next;
            } else {
                tail.next = l2;
                l2 = l2.next;
            }
            tail = tail.next;
        }

        tail.next = l1 == null ? l2 : l1;
        return dummy.next;
    }


    // -------合并K个升序链表 << end --------

    // -------下一个排列 start >>--------

    /**
     * 整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。
     * 整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，
     * 那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
     *
     * 例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
     * 类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
     * 而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
     *
     *
     * 解题思路：
     * 1、从后向前查找第一个相邻的元素对 (i, i + 1), 满足 A[i] < A[i + 1]。此时[i+1, end) 必然是降序
     * 2、在 [i + 1, end)中从后往前查找第一个满足 A[i] < A[k] 的k。 A[i], A[k] 分别就是【小数】和【大数】。
     * 3、将 A[i] 与 A[k] 交换
     * 4、可以断定这时 [i + 1, end) 必然是降序，重新排序 [i + 1, end) 使其升序。
     * 5、如果在步骤1中找不到符合的相邻元素时，说明当前 [begin, end) 为一个降序顺序，直接跳到 步骤4
     *
     * 对应 leetcode 中第 31 题。
     *
     * @param nums list of nums
     */
    public void nextPermutation(int[] nums) {
        if (null == nums || nums.length == 0) {
            return;
        }
        int len = nums.length;
        for (int i = len - 1; i > 0; i--) {
            if (nums[i] > nums[i - 1]) {
                for (int j = len - 1; j >= i; j--) {
                    if (nums[j] > nums[i - 1]) {
                        int temp = nums[i - 1];
                        nums[i - 1] = nums[j];
                        nums[j] = temp;
                        Arrays.sort(nums, i, len);
                        return;
                    }
                }
            }
        }
        Arrays.sort(nums);
    }

    // -------下一个排列 << end --------

    // -------最长有效括号 start >>--------

    /**
     * 定义 dp[i] 表示下标i字符结尾的最长有效括号的长度。我们将 dp 数组全部初始化为0。显然有效的字符串一定以 ‘)’ 结尾，
     * 因此哦们可以知道以 '('结尾的子串对应的dp值必定为0，我们只需要求解 '(' 在dp数组中对应位置的值。
     * <p>
     * 从前往后遍历字符串求解dp值，我们每两个字符检查一次：
     * 1、s[i] = ')'且 s[i - 1] = '('， 也就是字符串形如 ‘......()’, 所以有
     * dp[i] = dp[i - 2] + 2;
     * 2、s[i] = ')'且 s[i - 1] = ')', 也就是字符串形如 '......))'， 我们可以推出：
     * 如果 s[i - dp[i - 1] - 1] = '(', 那么
     * dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2
     *
     * @param s candidate of string
     * @return validated count character
     */
    public int longestValidParentheses(String s) {
        int[] dp = new int[s.length()];
        int result = 0;
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    int before = i - dp[i - 1] >= 2 ? dp[i - dp[i - 1] - 2] : 0;
                    dp[i] = before + 2 + dp[i - 1];
                }
                result = Math.max(result, dp[i]);
            }
        }
        return result;
    }

    /**
     * 使用栈来实现。
     * <p>
     * 我们始终保持栈底元素为当前已经遍历过的元素中，【最后一个没有被匹配的右括号的下标】，这样的做法主要是考虑了边界条件的处理。
     * 栈里其他元素维护左括号的下标：
     * 1、对于遇到的每个 '('， 我们将他的下标放入栈中。
     * 2、对于遇到的每个 ')'，我们先弹出栈顶元素表示匹配了当前右括号：
     * 如果栈为空，说明当前的右括号为没有配匹配的右括号，我们将其下标放入栈中来更新我们之前提到的 【最后一个没有被匹配的右括号的下标】
     * 如果栈不为空，当前右括号的下标减去栈顶元素即为 【以该右括号为结尾的最长有效括号的长度】
     * <p>
     * 我们从前往后遍历字符串并更新答案即可。
     * 需要注意的是： 如果栈一开始为空，第一个字符为左括号的时候我们会将其放入栈中， 这样就不满足提及的 【最后一个没有被匹配的右括号的下标】，
     * 为了保持统一，我们在一开始的时候放入一个值为 -1 的元素。
     *
     * @param s candidate of string
     * @return validated character count
     */
    public int longestValidParentheses_v2(String s) {
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        stack.add(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.add(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    result = Math.max(result, i - stack.peek());
                }
            }
        }
        return result;
    }

    /**
     * 在此方法中，我们利用两个计数器 left 和 right。首先，我们从左到右遍历字符串，对于遇到的每个 '(', 我们增加 left 计数器，
     * 对于遇到的每个 right 计数器。每当left 计数器 与 right 计数器相等时， 我们计算当前有效字符串的长度，并且记录目前为止找到的 最长字符串。
     * 当 right 计数器 比 left 计数器大时，我们将left 和 right 计数器同时变为 0。
     * <p>
     * 这样的做法贪心地考虑了以当前字符下标结尾的有效括号长度，每次当右括号数量多于左括号数量的时候之前的字符我们都扔掉不再考虑，重新从下一个字符开始计算，但这样会漏掉一种情况，就是遍历的时候左括号的数量始终大于右括号的数量，即 (() ，这种时候最长有效括号是求不出来的。
     * 解决的方法也很简单，我们只需要从右往左遍历用类似的方法计算即可，只是这个时候判断条件反了过来：
     * <p>
     * 当 left 计数器比 right 计数器大时，我们将 left 和 right 计数器同时变回 0
     * 当 left 计数器与 right 计数器相等时，我们计算当前有效字符串的长度，并且记录目前为止找到的最长子字符串
     *
     * @param s candidate of string
     * @return validated count character
     */
    public int longestValidParentheses_v3(String s) {
        int left = 0, right = 0, result = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                result = Math.max(result, 2 * right);
            } else if (right > left) {
                left = right = 0;
            }
        }
        left = right = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                result = Math.max(result, 2 * left);
            } else if (right < left) {
                left = right = 0;
            }
        }
        return result;
    }

    // -------最长有效括号 << end --------

    // -------在排序数组中查找元素的第一个和最后一个位置 start >>--------

    public int[] searchRange(int[] nums, int target) {
        int[] result = {-1, -1};
        if (null == nums || nums.length == 0) {
            return result;
        }

        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int middle = (right - left) / 2 + left;
            if (nums[middle] == target) {
                left = right = middle;
                while (left >= 0 && nums[left] == target) left--;
                result[0] = left + 1;
                while (right < nums.length && nums[right] == target) right++;
                result[1] = right - 1;
                return result;
            }
            if (nums[middle] < target) {
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }
        return result;
    }

    // -------在排序数组中查找元素的第一个和最后一个位置 << end --------

    // -------组合总和 start >>--------

    /**
     * 不加剪枝的情况下，使用idx 表示按照一定的顺序迭代进行递归，保证没有重复性。
     *
     * @param candidates arrays of candidates
     * @param target     target
     * @return list of result list
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();

        combinationSumRecursive(candidates, target, result, new ArrayList<>(), 0);
        return result;
    }

    private void combinationSumRecursive(int[] candidates, int leftTarget, List<List<Integer>> result, List<Integer> temp, int idx) {
        if (idx == candidates.length) {
            return;
        }
        if (leftTarget == 0) {
            result.add(new ArrayList<>(temp));
            return;
        }

        if (leftTarget - candidates[idx] >= 0) {
            temp.add(candidates[idx]);
            combinationSumRecursive(candidates, leftTarget - candidates[idx], result, temp, idx);
            temp.remove(temp.size() - 1);
        }
        combinationSumRecursive(candidates, leftTarget, result, temp, idx + 1);
    }

    /**
     * 如果 target 减去一个数得到负数，那么减去一个更大的数依然是负数， 同样搜索不到结果。所以我们可以对输入数组进行排序，添加相关逻辑达到进一步剪枝的目的。
     * <p>
     * 排序是为了提高搜索速度，对于解决这个问题来说非必要。但是搜索问题一般复杂度较高，能剪枝就尽量剪枝。
     *
     * @param candidates arrays of candidates
     * @param target     target
     * @return list of result list
     */
    public List<List<Integer>> combinationSum_v2(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        // 排序是剪枝的前提
        Arrays.sort(candidates);
        Deque<Integer> path = new ArrayDeque<>();
        combinationSumRecursive_v2(candidates, 0, len, target, path, res);
        return res;
    }

    private void combinationSumRecursive_v2(int[] candidates, int begin, int len, int target, Deque<Integer> path, List<List<Integer>> res) {
        // 由于进入更深层的时候，小于0的部分被剪枝，因此递归终止条件只判断等于0的情况
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = begin; i < len; i++) {
            // 重点理解这里的剪枝， 前提是候选数组已经有序
            if (target - candidates[i] < 0) {
                break;
            }
            path.addLast(candidates[i]);
            combinationSumRecursive_v2(candidates, i, len, target - candidates[i], path, res);
            path.removeLast();
        }
    }

    // -------组合总和 << end --------

    // -------计算不同的岛屿数量 start >>--------

    /**
     * 计算不同岛屿的数量
     * 采取的是遍历岛屿的方向的序列化，然后通过一个set进行去重处理
     *
     * @param grid matrix of grid
     * @return num of distinct islands
     */
    public int numDistinctIslands(int[][] grid) {
        int row = grid.length, col = grid[0].length;
        Set<String> isLands = new HashSet<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    StringBuilder sb = new StringBuilder();
                    // 初始的方向可以随便写，不影响正确性
                    numDistinctIslandsDfs(grid, i, j, sb, 666);
                    isLands.add(sb.toString());
                }
            }
        }
        return isLands.size();
    }

    private void numDistinctIslandsDfs(int[][] grid, int i, int j, StringBuilder sb, int dir) {
        int row = grid.length, col = grid[0].length;
        if (i < 0 || j < 0 || i >= row || j >= col || grid[i][j] == 0) {
            return;
        }
        // 前序遍历位置， 进入 （i， j）
        grid[i][j] = 0;
        sb.append(dir).append(',');
        numDistinctIslandsDfs(grid, i - 1, j, sb, 1); // 上
        numDistinctIslandsDfs(grid, i + 1, j, sb, 2); // 下
        numDistinctIslandsDfs(grid, i, j - 1, sb, 3); // 左
        numDistinctIslandsDfs(grid, i, j + 1, sb, 4); // 右
        // 后序遍历位置， 离开 （i， j）
        sb.append(-dir).append(',');
    }

    // -------计算不同的岛屿数量 << end --------

    // -------统计子岛屿 start >>--------

    public int countSubIslands(int[][] grid1, int[][] grid2) {
        int row = grid1.length, col = grid1[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                // 如果grid2的一部分是陆地，但是grid1对应的部分是海洋，
                // 那么这个肯定不是子岛屿，直接淹没
                if (grid2[i][j] == 1 && grid1[i][j] == 0) {
                    countSubIslandsDfs(grid2, i, j);
                }
            }
        }

        int res = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid2[i][j] == 1) {
                    res++;
                    countSubIslandsDfs(grid2, i, j);
                }
            }
        }
        return res;
    }

    private void countSubIslandsDfs(int[][] grid, int i, int j) {
        int row = grid.length, col = grid[0].length;
        if (i < 0 || j < 0 || i >= row || j >= col) {
            return;
        }
        if (grid[i][j] == 0) {
            return;
        }
        grid[i][j] = 0;
        countSubIslandsDfs(grid, i - 1, j);
        countSubIslandsDfs(grid, i + 1, j);
        countSubIslandsDfs(grid, i, j - 1);
        countSubIslandsDfs(grid, i, j + 1);
    }

    // -------统计子岛屿 << end --------

    // -------岛屿的最大面积 start >>--------

    /**
     * 给你一个大小为 m x n 的二进制矩阵 grid 。
     * 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
     * 岛屿的面积是岛上值为 1 的单元格的数目。
     * 计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
     *
     * @param grid matrix of grid
     * @return max area of island
     */
    public int maxAreaOfIsland(int[][] grid) {
        int row = grid.length, col = grid[0].length;
        int res = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    res = Math.max(res, maxAreaOfIsLandDfs(grid, i, j));
                }
            }
        }
        return res;
    }

    private int maxAreaOfIsLandDfs(int[][] grid, int i, int j) {
        int row = grid.length, col = grid[0].length;
        if (i < 0 || j < 0 || i >= row || j >= col) {
            return 0;
        }
        if (grid[i][j] == 0) {
            return 0;
        }
        grid[i][j] = 0;
        return maxAreaOfIsLandDfs(grid, i - 1, j)
                + maxAreaOfIsLandDfs(grid, i + 1, j)
                + maxAreaOfIsLandDfs(grid, i, j - 1)
                + maxAreaOfIsLandDfs(grid, i, j + 1) + 1;
    }

    // -------岛屿的最大面积 << end --------

    // -------统计封闭岛屿的数量 start >>--------

    /**
     * 有一个二维矩阵 grid ，每个位置要么是陆地（记号为 0 ）要么是水域（记号为 1 ）。
     * 我们从一块陆地出发，每次可以往上下左右 4 个方向相邻区域走，能走到的所有陆地区域，我们将其称为一座「岛屿」。
     * 如果一座岛屿 完全 由水域包围，即陆地边缘上下左右所有相邻区域都是水域，那么我们将其称为 「封闭岛屿」。
     * 请返回封闭岛屿的数目。
     *
     * @param grid matrix of grid
     * @return count of island
     */
    public int closedIsland(int[][] grid) {
        int row = grid.length, col = grid[0].length;
        for (int j = 0; j < col; j++) {
            // 将靠上边的岛屿淹没
            closedIsLandDfs(grid, 0, j);
            // 将靠下边的岛屿淹没
            closedIsLandDfs(grid, row - 1, j);
        }

        for (int i = 1; i < row - 1; i++) {
            closedIsLandDfs(grid, i, 0);
            ;
            closedIsLandDfs(grid, i, col - 1);
            ;
        }
        int res = 0;
        for (int i = 1; i < row - 1; i++) {
            for (int j = 1; j < col - 1; j++) {
                if (grid[i][j] == 0) {
                    res++;
                    closedIsLandDfs(grid, i, j);
                }
            }
        }
        return res;
    }

    private void closedIsLandDfs(int[][] grid, int i, int j) {
        int row = grid.length, col = grid[0].length;
        if (i < 0 || j < 0 || i >= row || j >= col) {
            return;
        }
        if (grid[i][j] == 1) {
            return;
        }
        grid[i][j] = 1;
        closedIsLandDfs(grid, i - 1, j);
        closedIsLandDfs(grid, i + 1, j);
        closedIsLandDfs(grid, i, j - 1);
        closedIsLandDfs(grid, i, j + 1);
    }

    // -------统计封闭岛屿的数量 << end --------

    // -------岛屿数量 start >>--------

    /**
     * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * 此外，你可以假设该网格的四条边均被水包围。
     *
     * @param grid grid
     * @return island number
     */
    public int numIslands(char[][] grid) {
        int res = 0;
        int lines = grid.length, col = grid[0].length;
        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '1') {
                    // 每发现一个岛屿， 将岛屿数量加1
                    res++;
                    // 然后使用dfs 将岛屿淹了
                    numIsLandsDfs(grid, i, j);
                }
            }
        }
        return res;
    }

    private void numIsLandsDfs(char[][] grid, int i, int j) {
        int rows = grid.length, cols = grid[0].length;
        if (i < 0 || j < 0 || i >= rows || j >= cols) {
            return;
        }
        if (grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';
        numIsLandsDfs(grid, i - 1, j);
        numIsLandsDfs(grid, i + 1, j);
        numIsLandsDfs(grid, i, j - 1);
        numIsLandsDfs(grid, i, j + 1);
    }

    // -------岛屿数量 << end --------

    // -------最小覆盖子串 start >>--------

    /**
     * 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
     * <p>
     * 注意：
     * 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
     * 如果 s 中存在这样的子串，我们保证它是唯一的答案。
     *
     * @param s original string
     * @param t sub string
     * @return minimum substring which contains the specified characters
     */
    public String minWindow(String s, String t) {
        if (s.length() < t.length()) {
            return "";
        }
        Map<Character, Integer> windowCharsMap = new HashMap<>();
        Map<Character, Integer> subCharsMap = new HashMap<>();
        for (char c : t.toCharArray()) {
            Integer count = subCharsMap.getOrDefault(c, 0);
            subCharsMap.put(c, ++count);
        }

        int left = 0, right = 0, valid = 0;
        int start = left, len = Integer.MAX_VALUE;
        while (right < s.length()) {
            char c = s.charAt(right);
            right++;
            if (subCharsMap.containsKey(c)) {
                Integer windowCount = windowCharsMap.getOrDefault(c, 0);
                windowCharsMap.put(c, ++windowCount);
                if (subCharsMap.get(c).equals(windowCount)) {
                    valid++;
                }
            }
            while (valid == subCharsMap.size()) {
                if (right - left < len) {
                    start = left;
                    len = right - left;
                }
                char leftC = s.charAt(left);
                left++;
                if (subCharsMap.containsKey(leftC)) {
                    if (subCharsMap.get(leftC).equals(windowCharsMap.get(leftC))) {
                        valid--;
                    }
                    Integer allAlreadyCount = windowCharsMap.get(leftC);
                    windowCharsMap.put(leftC, --allAlreadyCount);
                }
            }
        }
        return len == Integer.MAX_VALUE ? "" : s.substring(start, start + len);
    }

    // -------最小覆盖子串 << end --------

    // -------字符串的排列 start >>--------

    /**
     * 给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。
     * <p>
     * 换句话说，s1 的排列之一是 s2 的 子串 。
     */
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) {
            return false;
        }
        Map<Character, Integer> subMap = new HashMap<>();
        Map<Character, Integer> windowMap = new HashMap<>();
        for (char c : s1.toCharArray()) {
            Integer count = subMap.getOrDefault(c, 0);
            subMap.put(c, ++count);
        }
        int left = 0, right = 0, valid = 0;
        while (right < s2.length()) {
            char c = s2.charAt(right);
            right++;
            if (subMap.containsKey(c)) {
                Integer count = windowMap.getOrDefault(c, 0);
                windowMap.put(c, ++count);
                if (windowMap.get(c).equals(subMap.get(c))) {
                    valid++;
                }
            }
            // 判断左侧窗口是否要收缩
            while (right - left >= s1.length()) {
                if (valid == subMap.size()) {
                    return true;
                }
                char leftC = s2.charAt(left);
                left++;
                if (subMap.containsKey(leftC)) {
                    Integer count = windowMap.get(leftC);
                    if (count.equals(subMap.get(leftC))) {
                        valid--;
                    }
                    windowMap.put(leftC, --count);

                }
            }
        }
        return false;
    }

    // -------字符串的排列 << end --------

    // -------无重复字符的最长子串 start >>--------

    /**
     * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
     *
     * 对于滑动窗口问题，窗口的定义 是 左闭右开的，也就是 [left, right)
     *
     * 对应 leetcode 中第 3 题。
     */
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> window = new HashMap<>();
        int right = 0, left = 0, result = 0;
        while (right < s.length()) {
            char c = s.charAt(right);
            right++;
            Integer count = window.getOrDefault(c, 0);
            window.put(c, ++count);
            while (window.get(c) > 1) {
                char leftC = s.charAt(left);
                Integer alreadyCount = window.get(leftC);
                window.put(leftC, --alreadyCount);
                left++;
            }
            result = Math.max(result, right - left);
        }
        return result;
    }

    // -------无重复字符的最长子串 << end --------

    // -------网络延迟时间 start >>--------

    /**
     * 有 n 个网络节点，标记为 1 到 n。
     * 给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点，
     * wi 是一个信号从源节点传递到目标节点的时间。
     * 现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。
     */
    public int networkDelayTime(int[][] times, int n, int k) {
        // 节点编号是从1开始的，所以要一个大小为 n + 1 的邻接表
        List<int[]>[] graph = new LinkedList[n + 1];
        for (int i = 1; i <= n; i++) {
            graph[i] = new LinkedList<>();
        }
        for (int[] edge : times) {
            int from = edge[0];
            int to = edge[1];
            int weight = edge[2];
            // from -> List(<to, weight>)
            // 邻接表存储图结构，同时存储权重信息
            graph[from].add(new int[] {to, weight});
        }
        // 启动 dijkstra 算法计算以节点k为起点到其他节点的最短路径
        int[] distTo = networkDelayTimeDijkstra(k , graph);

        // 找到最长的那一条最短路径
        int res = 0;
        for (int i = 1; i < distTo.length; i++) {
            if (distTo[i] == Integer.MAX_VALUE) {
                // 右节点不可达，返回 -1
                return -1;
            }
            res = Math.max(res, distTo[i]);
        }
        return res;
    }

    /**
     * 输入一个起点 start，计算从 start 到其他节点的最短距离
     */
    private int[] networkDelayTimeDijkstra(int start, List<int[]>[] graph) {
        // 定义： distTo[i] 的值就是起点 start 到达节点 i 的最短路径权重
        int[] distTo = new int[graph.length];
        Arrays.fill(distTo, Integer.MAX_VALUE);
        // base case, start 到 start 的距离是0
        distTo[start] = 0;

        // 优先级队列， distFromStart 较小的排在前面
        Queue<State> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.distFromStart));
        pq.offer(new State(start, 0));
        while (!pq.isEmpty()) {
            State curState = pq.poll();
            int curNodeId = curState.id;
            int curDistFromStart = curState.distFromStart;
            if (distTo[curNodeId] < curDistFromStart) {
                continue;
            }

            // 将curNode 的相邻节点装入队列
            for (int[] neighbor : graph[curNodeId]) {
                int nextNodeId = neighbor[0];
                int distToNextNode = curDistFromStart + neighbor[1];

                // 更新dp table
                if (distTo[nextNodeId] > distToNextNode) {
                    distTo[nextNodeId] = distToNextNode;
                    pq.offer(new State(nextNodeId, distToNextNode));
                }
            }
        }
        return distTo;
    }

    // -------网络延迟时间 << end --------

    /*
     * 动态规划算法本质上就是穷举“状态”，然后在“选择”中选取最优解
     * 对于股票买卖问题，我们具体到每一天，看看总共有几种可能的“状态”，在找出每个“状态”对应的“选择”。我们要穷举所有的“状态”，穷举的目的是根据对应的
     * “选择”更新状态。
     * 每天都有三种选择： 买入、卖出、无操作，我们用 buy、sell、rest表示这三种选择。
     * 但问题是，并不是每天都可以任意选择这三种选择的，因为sell必须在 buy 之后，buy 必须在 sell 之后。那么 rest 操作还应该分两种状态， 一种是
     * buy 之后的 rest （持有了股票），一种是 sell 之后的 rest （没有持有股票）。而且别忘了，我们还有交易次数 K 的限制，就是说你buy还只能在
     * K > 0 的前提下操作。
     *
     * 这个问题的状态有三个，第一个是天数，第二个是允许交易的最大次数，第三个是当前的持有状态（即之前说的 rest 状态，我们不妨用1表示持有，0表示没有持有）
     * 然后我们用一个三维数组就可以装下这几种状态的全部组合：
     * dp[i][k][0 or 1]
     * 0 <= i <= n - 1, 1 <= k <= K
     * n 为天数，大K为交易数的上限，0 和 1 代表是否持有股票
     * 此问题共 n * K * 2 中状态，全部穷举就能搞定。
     * 于是状态转移方程可以写出来:
     *
     * dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
     *               max(今天选择 rest，    今天选择sell)
     * 解释： 今天我没有持有股票，有两种可能，我从这两种可能中求最大利润：
     * 1、我昨天j就没有持有，且截至昨天最大交易次数限制为 k， 然后今天我选择rest， 所以我今天还是没有持有，最大交易次数限制为 k。
     * 2、我昨天持有股票，且截至昨天最大交易次数限制为 k，但是今天我 sell 了， 所以我今天没有持有股票了，最大交易次数限制依然为 k。
     *
     * dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
     *               max( 今天选择rest，    今天选择 buy)
     * 解释：今天我持有着股票，最大交易次数限制为k，那么对于昨天来说，有两种可能，我从这两种可能中求最大利润：
     * 1、我昨天就持有着股票，且截至昨天最大交易次数限制为 k，然后今天选择rest，所以我还是持有者股票，最大交易次数限制依然为 k。
     * 2、我昨天没有持有，且截至昨天最大交易次数限制为 k - 1， 但今天我选择 buy， 所以我今天持有了，最大交易次数限制为 k。
     *
     * base case 的定义：
     * dp[-1][...][0] = 0
     * 解释： 因为i是从0开始的， 所以i = -1意味着还没有开始，这时候的利润当然是0。
     *
     * dp[-1][...][1] = -infinity
     * 解释： 还没开始的时候，是不可能持有股票的。因为我们的算法要求一个最大值，所以初始值设为一个最小值，方便取最大值。
     *
     * dp[...][0][0] = 0
     * 解释： 因为k是从1开始的，所以k = 0意味着根本不允许交易，这时候的利润当然是0.
     *
     * dp[...][0][1] = -infinity
     * 解释：不允许交易的情况下，是不可能持有股票的。因为我们的算法要求一个最大值，所以初始值设为一个最小值，方便取最大值。
     */
    // -------买卖股票的最佳时机 start >>--------

    /**
     * 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
     * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
     * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
     *
     * 这里的 k = 1.
     * 直接套状态转移方程，根据base case，可以做一些化简：
     * dp[i][1][0] = max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i])
     * dp[i][1][1] = max(dp[i - 1][1][1], dp[i - 1][0][0] - prices[i])
     *             = max(dp[i - 1][1][1], -prices[i])
     *
     * 现在发现 k 都是1，不会改变，即 k 对状态转移方程已经没有影响了。可以进一步化简去掉所有的 k。
     * dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
     * dp[i][1] = max(dp[i - 1][1], -prices[i])
     *
     */
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        for (int i = 0; i < n; i++) {
            if (i - 1 == -1) {
                // base case
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
        }
        return dp[n - 1][0];
    }

    // 空间优化版本
    public int maxProfit_v1(int[] prices) {
        int dp_i0 = 0, dp_i1 = Integer.MIN_VALUE;
        for (int price : prices) {
            dp_i0 = Math.max(dp_i0, dp_i1 + price);
            dp_i1 = Math.max(dp_i1, -price);
        }
        return dp_i0;
    }

    // -------买卖股票的最佳时机 << end --------

    // -------买卖股票的最佳时机II start >>--------

    /**
     * 可以认为这一题的 k 为 infinity， 如果 k 为正无穷，那么就可以认为 k 和 k - 1 是一样的，
     * 可以这样改写框架
     *
     * dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
     * dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
     *             = max(dp[i-1][k][1], dp[i-1][k][0] - prices[i])
     *
     * 我们发现数组中的 k 已经不会在改变了，也就是不需要记录 k 这个状态了。
     * dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
     * dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
     *
     */
    public int maxProfit_infinity(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        for (int i = 0; i < n; i++) {
            if (i - 1 == -1) {
                // base case
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    public int maxProfit_infinity_v2(int[] prices) {
        int n = prices.length;
        int dp_i0 = 0, dp_i1 = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int temp = dp_i0;
            dp_i0 = Math.max(dp_i0, dp_i1 + prices[i]);
            dp_i1 = Math.max(dp_i1, temp - prices[i]);
        }
        return dp_i0;
    }

    // -------买卖股票的最佳时机II << end --------

    // -------买卖股票的最佳时机含冷冻期 start >>--------

    /**
     * 每次sell 之后要等一天才能继续交易，只要将这个特点融入上一题即可
     *
     * dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
     * dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])
     * 解释：第 i 天选择 buy 的时候，要从 i-2 的状态转移，而不是 i-1 。
     */
    public int maxProfit_withCool(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        for (int i = 0; i < n; i++) {
            if (i - 1 == -1) {
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            if (i - 2 == -1) {
                dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
                dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
                continue;
            }
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 2][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    public int maxProfit_withCool_v2(int[] prices) {
        int n = prices.length;
        int dp_i0 = 0, dp_i1 = Integer.MIN_VALUE, dp_pre0 = 0; // 代表dp[i - 2][0]
        for (int i = 0; i < n; i++) {
            int temp = dp_i0;
            dp_i0 = Math.max(dp_i0, dp_i1 + prices[i]);
            dp_i1 = Math.max(dp_i1, dp_pre0 - prices[i]);
            dp_pre0 = temp;
        }
        return dp_i0;
    }

    // -------买卖股票的最佳时机含冷冻期 << end --------

    // -------买卖股票的最佳时机III start >>--------

    public int maxProfit_k_2(int[] prices) {
        int max_k = 2, n = prices.length;
        int[][][] dp = new int[n][max_k + 1][2];
        for (int i = 0; i < n; i++) {
            for (int k = max_k; k >= 1; k--) {
                if (i - 1 == -1) {
                    // 处理 base case
                    dp[i][k][0] = 0;
                    dp[i][k][1] = -prices[i];
                    continue;
                }
                dp[i][k][0] = Math.max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
                dp[i][k][1] = Math.max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
            }
        }
        // 穷举了 n × max_k × 2 个状态，正确。
        return dp[n - 1][max_k][0];
    }

    // -------买卖股票的最佳时机III << end --------

    // -------整数反转 start >>--------

    /**
     * 给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
     * 如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。
     * 假设环境不允许存储 64 位整数（有符号或无符号）。
     *
     * 对应 leetcode 中第 7 题
     */
    public int reverse(int x) {
        int result = 0;
        while (x != 0) {
            if (result < Integer.MIN_VALUE / 10 || result > Integer.MAX_VALUE / 10)
                return 0;
            int digit = x % 10;
            result = result * 10 + digit;
            x /= 10;
        }
        return result;
    }

    // -------整数反转 << end --------

    // -------全排列 start >>--------

    /**
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     *
     * 使用 回溯的解法进行求解
     *
     * 对应 leetcode 中第 46 题
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        boolean[] used = new boolean[nums.length];
        permuteTraverse(nums, used, new LinkedList<>(), result);
        return result;
    }

    private void permuteTraverse(int[] nums, boolean[] used, Deque<Integer> list, List<List<Integer>> result) {
        if (list.size() == nums.length) {
            result.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!used[i]) {
                used[i] = true;
                list.offerLast(nums[i]);
                permuteTraverse(nums, used, list, result);
                list.removeLast();
                used[i] = false;
            }
        }
    }

    // -------全排列 << end --------

    // -------旋转图像 start >>--------

    /**
     * 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
     * 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
     *
     * 解题思路：
     * 经证明，发现可以先通过水平轴翻转，然后在根据主对角线翻转 得到
     *
     * 对应 leetcode 中第 48 题
     */
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        // 水平上下旋转
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 -i][j];
                matrix[n - 1 - i][j] = temp;
            }
        }
        // 主对角线旋转
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

    // -------旋转图像 << end --------

    // -------跳跃游戏 start >>--------

    /**
     * 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * 判断你是否能够到达最后一个下标。
     *
     * 对应 leetcode 中第 55 题
     */
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int i = 0;
        boolean[] jump = new boolean[n];
        jump[0] = true;
        while (jump[i] && i < n - 1) {
            for (int j = i + 1; j < n && j - i <= nums[i]; j++) {
                jump[j] = true;
            }
            i++;
        }
        return jump[n - 1];
    }

    /**
     * 如果某一个作为起跳点的格子可以跳跃的距离是 3，那么表示后面 3个格子都可以作为起跳点。
     * 可以对每一个能作为起跳点 的格子都尝试跳一次，把能跳到最远的距离不断更新。
     * 如果可以一直跳到最后，就成功了。
     */
    public boolean canJump_v2(int[] nums) {
        int k = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > k) return false;
            k = Math.max(k, i + nums[i]);
        }
        return true;
    }

    // -------跳跃游戏 << end --------

    // -------合并区间 start >>--------

    /**
     * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，
     * 并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。
     *
     * 对应 leetcode 中第 56 题
     */
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        int minValue = intervals[0][0], maxValue = intervals[0][1], index = 0;
        int[][] res = new int[intervals.length][2];
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] > maxValue) {
                res[index][0] = minValue;
                res[index][1] = maxValue;
                index++;
                minValue = intervals[i][0];
                maxValue = intervals[i][1];
            } else {
                maxValue = Math.max(maxValue, intervals[i][1]);
            }
        }
        res[index][0] = minValue;
        res[index][1] = maxValue;
        return Arrays.copyOfRange(res, 0, index + 1);
    }

    // -------合并区间 << end --------

    // -------不同路径 start >>--------

    /**
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     * 问总共有多少条不同的路径？
     *
     * 解题思路:
     * dp 数组的定义: dp[i][j] 表示从开始的 [0,0] 到达 [i,j] 共有 dp[i][j] 条路径.
     * 那么对于 dp[i][j] 来说,由于只能 向右或者向下走,所以 [i,j] 的来源只能是 [i-1,j] 或者 [i,j-1] ,
     * 对于 dp 数组的定义,分别对应 dp[i-1][j] 和 dp[i][j-1] ,所以有递推公式, dp[i][j] = dp[i-1][j] + dp[i][j-1]
     * 同时考虑 base case, dp[..][0] = 1 , dp[0][..] = 1, 显然,只能通过直线到达,所以值都为 1.
     *
     * 对应 leetcode 中第 62 题
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }

    /**
     * 上面解法的 dp 数组压缩版
     */
    public int uniquePaths_v2(int m, int n) {
        int min = Math.min(m, n), max = Math.max(m, n);
        int[] dp = new int[min];
        Arrays.fill(dp, 1);
        for (int i = 1; i < max; i++) {
            for (int j = 1; j < min; j++) {
                dp[j] += dp[j - 1];
            }
        }
        return dp[min - 1];
    }

    // -------不同路径 << end --------

    // -------最小路径和 start >>--------

    /**
     * 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     * 说明：每次只能向下或者向右移动一步。
     *
     * 对应 leetcode 中第 64 题
     */
    public int minPathSum(int[][] grid) {
        for (int i = 1; i < grid.length; i++) {
            grid[i][0] += grid[i - 1][0];
        }
        for (int j = 1; j < grid[0].length; j++) {
            grid[0][j] += grid[0][j - 1];
        }
        for (int i = 1; i < grid.length; i++) {
            for (int j = 1; j < grid[0].length; j++) {
                grid[i][j] += Math.min(grid[i-1][j], grid[i][j - 1]);
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }

    // -------最小路径和 << end --------

    // -------爬楼梯 start >>--------

    /**
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     * 注意：给定 n 是一个正整数。
     *
     * 对应 leetcode 中第 70题
     */
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    // -------爬楼梯 << end --------

    // -------颜色分类 start >>--------

    /**
     * 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
     * 此题中，我们使用整数 0、1 和 2 分别表示红色、白色和蓝色。
     *
     * 对应 leetcode 中第 75 题
     */
    public void sortColors(int[] nums) {
        int len = nums.length;
        if (len < 2) {
            return;
        }
        // all in [0, zero) = 0
        // all in [zero, i) = 1
        // all in [two, len) = 2

        int zero = 0;
        int two = len;
        int i = 0;
        while (i < two) {
            if (nums[i] == 0) {
                int temp = nums[i];
                nums[i] = nums[zero];
                nums[zero] = temp;

                zero++;
                i++;
            } else if (nums[i] == 1) {
                i++;
            } else {
                two--;

                int tmp = nums[i];
                nums[i] = nums[two];
                nums[two] = tmp;
            }
        }
    }

    // -------颜色分类 << end --------

    // -------子集 start >>--------

    /**
     * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
     * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
     *
     * 对应 leetcode 中第 78 题
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        subsetsBackTrack(nums, 0, new LinkedList<>(), result);
        return result;
    }

    private void subsetsBackTrack(int[] nums, int start, Deque<Integer> list, List<List<Integer>> result) {
        result.add(new ArrayList<>(list));
        for (int i = start; i < nums.length; i++) {
            list.offerLast(nums[i]);
            subsetsBackTrack(nums, i + 1, list, result);
            list.removeLast();
        }
    }

    // -------子集 << end --------

    // -------子集II start >>--------

    /**
     * 给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
     * 解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
     *
     * 对应 leetcode 中第 90 题
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        subsetsWithDupBacktrack(nums, 0, new LinkedList<>(), result);
        return result;
    }

    private void subsetsWithDupBacktrack(int[] nums, int start, Deque<Integer> list, List<List<Integer>> result) {
        result.add(new ArrayList<>(list));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1])
                continue;
            list.offerLast(nums[i]);
            subsetsWithDupBacktrack(nums, i + 1, list, result);
            list.removeLast();
        }
    }

    // -------子集II << end --------

    // -------单词搜索 start >>--------

    /**
     * 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     *
     * 对应 leetcode 中第 79 题
     */
    public boolean exist(char[][] board, String word) {
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (existWords(board, i, j, word, 0, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean existWords(char[][] board, int row, int col, String word, int curChar, boolean[][] visited) {
        if (curChar == word.length())
            return true;
        if (row < 0 || col < 0 || row >= board.length || col >= board[0].length || visited[row][col]) {
            return false;
        }
        if (board[row][col] != word.charAt(curChar))
            return false;
        visited[row][col] = true;
        boolean result =  existWords(board, row + 1, col, word, curChar + 1, visited)
                || existWords(board, row - 1, col, word, curChar + 1, visited)
                || existWords(board, row, col + 1, word, curChar + 1, visited)
                || existWords(board, row, col - 1, word, curChar + 1, visited);
        visited[row][col] = false;
        return result;
    }

    // -------单词搜索 << end --------

    // -------柱状图中最大的矩形 start >>--------

    /**
     * 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
     * 求在该柱状图中，能够勾勒出来的矩形的最大面积。
     *
     * 使用单调栈的方法进行解答
     *
     * 对应 leetcode 中第 84 题
     */
    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        int res = 0;
        Deque<Integer> stack = new ArrayDeque<>(len);
        for (int i = 0; i < len; i++) {
            // 这个while 很关键，因为有可能不止一个柱形的最大宽度可以被计算出来
            while (!stack.isEmpty() && heights[i] < heights[stack.peekLast()]) {
                int curHeight = heights[stack.pollLast()];
                while (!stack.isEmpty() && heights[stack.peekLast()] == curHeight) {
                    stack.pollLast();
                }
                int curWidth;
                if (stack.isEmpty()) {
                    curWidth = i;
                } else {
                    curWidth = i - stack.peekLast() - 1;
                }
                res = Math.max(res, curHeight * curWidth);
            }
            stack.offerLast(i);
        }

        while (!stack.isEmpty()) {
            int curHeight = heights[stack.pollLast()];
            while (!stack.isEmpty() && heights[stack.peekLast()] == curHeight) {
                stack.pollLast();
            }
            int curWidth;
            if (stack.isEmpty()) {
                curWidth = len;
            } else {
                curWidth = len - stack.peekLast() - 1;
            }
            res = Math.max(res, curHeight * curWidth);
        }
        return res;
    }

    /**
     * 上面方法的简化版，在开头和末尾分别添加 0
     * 因为上面的代码需要考虑两种特殊的情况，
     * 1、弹栈的时候，栈为空
     * 2、遍历完成以后，栈中还有元素。
     * 然后在开头添加 0 的时候，由于它一定比输入数组里任何一个元素小，它肯定不会出栈，因此栈一定不会为空。
     * 在末尾添加 0 的时候，也正是因为它一定比输入数组中的任何一个元素小，他会让所有输入数组里的元素出栈。
     */
    public int largestRectangleArea_v2(int[] heights) {
        int[] newHeights = new int[heights.length + 2];
        System.arraycopy(heights, 0, newHeights, 1, heights.length);
        Deque<Integer> stack = new ArrayDeque<>();
        int res = 0;
        for (int i = 0; i < newHeights.length; i++) {
            while (!stack.isEmpty() && newHeights[stack.peekLast()] > newHeights[i]) {
                int height = newHeights[stack.pollLast()];
                int width = i - stack.peekLast() - 1;
                res = Math.max(res, height * width);
            }
            stack.offerLast(i);
        }
        return res;
    }

    // -------柱状图中最大的矩形 << end --------

    // -------最大矩形 start >>--------

    /**
     * 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
     *
     * 解题思路：
     * 遍历每个点，求以这个点为矩阵的右下角的所有矩阵面积。那么如何找出这样的矩阵呢？
     * 如果我们知道了以这个点结尾的连续 1 的个数的话，问题就变得简单了。
     * 1、首先求出高度是 1 的矩形面积，也就是它自身的数，
     * 2、然后向上扩展一行，高度增加 1，选出当前列最小的数字，作为矩阵的宽，求出面积。
     * 3、然后继续向上扩展，重复步骤 2 .
     *
     * 对应 leetcode 中第 85 题
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0) {
            return 0;
        }
        // 保存以当前数字结尾的连续 1 的个数
        int[][] width = new int[matrix.length][matrix[0].length];
        int maxArea = 0;
        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[0].length; col++) {
                // 更新width
                if (matrix[row][col] == '1') {
                    if (col == 0) {
                        width[row][col] = 1;
                    } else {
                        width[row][col] = width[row][col - 1] + 1;
                    }
                }
                // 记录所有行中最小的数
                int minWidth = width[row][col];
                // 向上扩展行
                for (int upRow = row; upRow >= 0 && width[upRow][col] > 0; upRow--) {
                    int height = row - upRow + 1;
                    // 找最小的数作为矩阵的宽
                    minWidth = Math.min(minWidth, width[upRow][col]);
                    maxArea = Math.max(maxArea, height * minWidth);
                }
            }
        }
        return maxArea;
    }

    // -------最大矩形 << end --------

    // -------最长连续序列 start >>--------

    /**
     * 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
     * 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
     *
     * 解题思路：
     *  使用哈希表的方式进行解答，对于每一个 nums 中的元素，只判断满足当前元素没有左侧边界的情况进入内层循环，
     *  而对于所有的元素来说，每一个元素最终至多只会走一遍内层循环。大大优化了算法的效率。
     *
     * 对应 leetcode 中第 128 题
     */
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>(nums.length);
        for (int num : nums) {
            set.add(num);
        }
        int result = 0;
        for (int num : nums) {
            if (!set.contains(num - 1)) {
                int currentNum = num + 1;
                int curCount = 1;
                while (set.contains(currentNum)) {
                    currentNum += 1;
                    curCount += 1;
                }
                result = Math.max(result, curCount);
            }
        }
        return result;
    }

    /**
     * 解题思路：
     * 用 hashmap 存储每个端点值对应的连续区间的长度。
     * 若数已经在 hashmap 中，跳过不做处理
     * 若是新的数字加入：
     *  取出其左右相邻数已有的连续区间长度 left 和 right
     *  计算当前数的区间长度为 cur_length = left + right + 1;
     *  根据 cur_length 更新最大长度 max_length 的值
     *  更新区间两端点的值
     */
    public int longestConsecutive_v2(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int result = 0;
        for (int num : nums) {
            // 当map中不包含num，也就是说num 第一次出现
            if (!map.containsKey(num)) {
                // 左连续区间的长度
                int left = map.getOrDefault(num - 1, 0);
                // 右连续区间的长度
                int right = map.getOrDefault(num + 1, 0);
                // 当前连续区间的总长度
                int curLen = left + right + 1;
                result = Math.max(result, curLen);
                map.put(num, curLen);
                map.put(num - left, curLen);
                map.put(num + right, curLen);
            }
        }
        return result;
    }

    // -------最长连续序列 << end --------

    // -------单词拆分 start >>--------

    /**
     * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
     * 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
     *
     * 解题思路：
     * DFS
     *
     * 对应 leetcode 中第 139 题
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        Map<Character, List<String>> map = new HashMap<>();
        for (String word : wordDict) {
            map.computeIfAbsent(word.charAt(0), k -> new ArrayList<>()).add(word);
        }
        // 1: 访问且为 true， -1： 访问且为false
        int[] visited = new int[s.length()];
        return wordBreakTraverse(s, 0, map, visited);
    }

    private boolean wordBreakTraverse(String s, int start, Map<Character, List<String>> map, int[] visited) {
        if (start == s.length()) {
            return true;
        }
        if (start > s.length()) {
            return false;
        }
        // 剪枝防止重复计算
        if (visited[start] == 1) {
            return true;
        } else if (visited[start] == -1) {
            return false;
        }

        char c = s.charAt(start);
        if (!map.containsKey(c)) {
            return false;
        }
        for (String word : map.get(c)) {
            if (s.startsWith(word, start)
                    && wordBreakTraverse(s, start + word.length(), map, visited)) {
                visited[start] = 1;
                return true;
            }
        }
        visited[start] = -1;
        return false;
    }

    /**
     * 使用动态规划的方法进行解答：
     * s 喘能否分解为单词表的单词
     * 将大问题分解为规模小一点的子问题：san
     *  1、前 i 个字符的子串，能否分解盛单词
     *  2、剩余子串，是否为单个单词。
     *
     * 定义 dp[i] 长度为 i 的 s[0:i - 1]子串是否能拆分盛单词。题目求 dp[s.length]
     *
     * 状态转移方程：
     * 类似的，我们用指针 j 去划分 s[0:i] 子串，
     * s[0:i] 子串对应 dp[i + 1],它是否为 true(s[0:i] 能否break)，取决于两点：
     *  它的前缀子串 s[0:j-1] 的 dp[j]，是否为 true
     *  剩余子串 s[j:i], 是否是单词表的单词
     *
     * base case：
     *  dp[0] = true。即，长度为 0 的 s[0:-1] 能拆分盛单词表单词， （这看似荒谬，但这只是为了让边界情况也能套用状态转移方程而已）
     *  当 j = 0时， s[0:i] 的 dp[i + 1],取决于 s[0:-1] 的 dp[0], 和，剩余子串 s[0:i] 是否时单个单词。
     *  只有让 dp[0] 为真，dp[i + 1]才会只取决于 s[0:i] 是否为单个单词，才能用上这个状态转移方程。
     */
    public boolean wordBreak_v2(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    // -------单词拆分 << end --------

    // -------环形链表II start >>--------

    /**
     *`给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，
     * 评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
     * 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
     * 不允许修改 链表。
     *
     * 解题思路：使用双指针法
     * 设两指针第一次相遇， fast，slow都指向链表头部 head，fast每轮走 2 步，slow 每轮走 1 步。
     * 第一种结果：
     *    fast 指针走过链表末端，说明链表无环，直接返回 null
     * 第二种结果：
     *    当 fast == slow 时，两指针在环中第一次相遇。 假设链表共有 a + b 个节点，其中链表头部到链表入口右 a 个节点。链表环有 b 个节点
     *    设两指针分别走了 f, s 步，则有：
     *    1、fast 走的步数是 slow 的两倍，即 f = 2s;
     *    2、fast 比 slow 多走了n个环的长度，即 f = s + nb;
     *    以上两式相减得：f = 2nb, s = nb,即 fast 和 slow指针分别走了 2n 个，n 个环的周长。
     *
     * 目前情况分析：
     *    如果让指针从链表头部一直向前走并统计步数 k， 那么所有 走到链表入口节点时的步数是 k = a + nb
     *    而目前， slow 指针走过的步数是 nb 步。因此我们只要想办法让 slow 再走 a 步停下来，就可以到环的入口。
     *    但是我们不知道 a 的值，但是我们可以使用双指针法。我们构建一个指针，此指针需要有如下性质：
     *      此指针 和 slow 一起向前走 a 步后，两者在入口节点重合。那么从哪里走到入口节点需要 a 步呢？答案就是链表头部 head
     *
     * 对应 leetcode 中第 142 题
     */
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head, slow = head;
        do {
            if (fast == null || fast.next == null)
                return null;
            fast = fast.next.next;
            slow = slow.next;
        } while (fast != slow);
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    // -------环形链表II << end --------

    // -------乘积最大子数组 start >>--------

    /**
     * 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
     *
     * 解题思路：
     * 使用动态规划进行解题；我们可以根据正负性进行分类讨论。
     * 考虑当前位置如果是一个负数的话，那么我们希望以他前一个位置结尾的某个段的积也是一个负数，这样就可以负负得正，并且我们希望这个积尽可能的小。
     * 如果当前位置是一个正数的话，我们希望以他前一个位置结尾的积也是一个正数，并且希望他尽可能的大。所以我们维护两个dp 数组。转移方程如下：
     * dpMax[i] = Math.max(dpMax[i - 1] * nums[i], dpMin[i - 1] * nums[i], nums[i])
     * dpMin[i] = Math.min(dpMax[i - 1] * nums[i], dpMax[i - 1] * nums[i], nums[i])
     * 它代表第 i 个元素结尾的最大子数组的乘积 dpMax[i]。 可以考虑把 nums[i] 加入第 i - 1个元素结尾的乘积最大或最小的子数组中，二者加上
     * nums[i]， 三者取最大，就是第 i 个元素结尾的乘积中最大子数组的乘积， 同样的，最小子数组的乘积同理。
     *
     * 对应 leetcode 中第 152 题。
     */
    public int maxProduct(int[] nums) {
        int[] dpMax = new int[nums.length];
        int[] dpMin = new int[nums.length];
        System.arraycopy(nums, 0, dpMax, 0, nums.length);
        System.arraycopy(nums, 0, dpMin, 0, nums.length);
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dpMax[i] = Math.max(dpMax[i - 1] * nums[i], Math.max(dpMin[i - 1] * nums[i], nums[i]));
            dpMin[i] = Math.min(dpMin[i - 1] * nums[i], Math.min(dpMax[i - 1] * nums[i], nums[i]));
            res = Math.max(res, dpMax[i]);
        }
        return res;
    }

    /**
     * 上面解法的 dp 数组压缩版
     */
    public int maxProduct_v2(int[] nums) {
        int dpMax = nums[0], dpMin = nums[0], res = nums[0], pre = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dpMax = Math.max(dpMax * nums[i], Math.max(dpMin * nums[i], nums[i]));
            dpMin = Math.min(dpMin * nums[i], Math.min(nums[i], pre * nums[i]));
            pre = dpMax;
            res = Math.max(res, dpMax);
        }
        return res;
    }

    // -------乘积最大子数组 << end --------

    // -------多数元素 start >>--------

    /**
     * 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于 n/2的元素。
     * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     *
     * 对应 leetcode 中第 169 题。
     */
    public int majorityElement(int[] nums) {
        int count = 0;
        Integer candidate = null;
        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            count += (candidate == num) ? 1 : -1;
        }
        return candidate;
    }

    // -------多数元素 << end --------

    // -------反转链表 start >>--------

    /**
     * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
     *
     * 对应 leetcode 中第 206 题。
     *
     * @param head head node
     * @return head node
     */
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }

    /**
     * 递归版本
     */
    public ListNode reverseList_v2(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverseList_v2(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    // -------反转链表 << end --------

    // -------数组中的第K个最大元素 start >>--------

    /**
     * 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
     * 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
     *
     * 对应 leetcode 中第 215 题
     *
     */
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>(k, Comparator.comparingInt(a -> a));
        for (int i = 0; i < k; i++) {
            queue.offer(nums[i]);
        }
        for (int i = k; i < nums.length; i++) {
            Integer topElement = queue.peek();
            if (nums[i] > topElement) {
                queue.poll();
                queue.offer(nums[i]);
            }
        }
        return queue.peek();
    }

    /**
     * 自己实现 堆
     */
    public int findKthLargestManual(int[] nums, int k) {
        int[] queue = new int[k];
        for (int i = 0; i < k; i++) {
            queue[i] = nums[i];
            siftUp(queue, i);
        }
        for (int i = k; i < nums.length; i++) {
            if (nums[i] > queue[0]) {
                queue[0] = nums[i];
                siftDown(queue, 0);
            }
        }
        return queue[0];
    }

    private void siftDown(int[] queue, int k) {
        int cur = queue[k];
        int half = queue.length >>> 1;
        while (k < half) {
            int child = (k << 1) + 1;
            int right = child + 1;
            if (right < queue.length && queue[child] > queue[right]) {
                child = right;
            }
            if (cur <= queue[child]) {
                break;
            }
            queue[k] = queue[child];
            k = child;
        }
        queue[k] = cur;
    }

    private void siftUp(int[] queue, int k) {
        int cur = queue[k];
        while (k > 0) {
            int parent = (k - 1) >>> 1;
            if (cur >= queue[parent]) {
                break;
            }
            queue[k] = queue[parent];
            k = parent;
        }
        queue[k] = cur;
    }

    /**
     * 使用 快排 + 迭代的方式进行解答
     */
    public int findKthLargest_v2(int[] nums, int k) {
        int len = nums.length, target = len - k;
        int start = 0, end = len - 1;
        while (true) {
            int p = findKthLargestFind(nums, start, end);
            if (p == target) {
                return nums[p];
            } else if (p < target) {
                start = p + 1;
            } else {
                end = p - 1;
            }
        }
    }

    /**
     * 使用快排 + 递归的方式进行解答
     */
    public int findKthLargest_v3(int[] nums, int k) {
        int len = nums.length, targetPosition = len - k;
        findKthLargestPartSort(nums, 0, len - 1, targetPosition);
        return nums[targetPosition];
    }

    private boolean findKthLargestPartSort(int[] nums, int start, int end, int targetPosition) {
        if (start == targetPosition || end == targetPosition) {
            return true;
        }
        if (start >= end)
            return false;
        int p = findKthLargestFind(nums, start, end);
        return findKthLargestPartSort(nums, start, p - 1, targetPosition) ||
        findKthLargestPartSort(nums, p + 1, end, targetPosition);
    }

    private int findKthLargestFind(int[] nums, int start, int end) {
        int cur = nums[start];
        int i = start, j = end;
        while (i < j) {
            while (i < j && nums[j] >= cur) j--;
            nums[i] = nums[j];
            while (i < j && nums[i] <= cur) i++;
            nums[j] = nums[i];
        }
        nums[i] = cur;
        return i;
    }

    // -------数组中的第K个最大元素 << end --------

    // -------回文链表 start >>--------

    /**
     * 给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。
     *
     * 对应 leetcode 中第 234 题。
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        ListNode fast = head, slow = head;
        ListNode pre = null, prepre = null;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
            pre.next = prepre;
            prepre = pre;
        }
        if (fast != null) {
            slow = slow.next;
        }
        while (pre != null && slow != null) {
            if (pre.val != slow.val) {
                return false;
            }
            pre = pre.next;
            slow = slow.next;
        }
        return true;
    }

    // -------回文链表 << end --------

    // -------除自身以外数组的乘积 start >>--------

    /**
     * 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
     * 题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
     * 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
     *
     * 对应 leetcode 中第 238 题
     */
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        int p = 1, q = 1;
        for (int i = 0; i < nums.length; i++) {
            res[i] = p;
            p *= nums[i];
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            res[i] *= q;
            q *= nums[i];
        }
        return res;
    }

    // -------除自身以外数组的乘积 << end --------

    // -------完全平方数 start >>--------

    /**
     * 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
     * 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
     *
     * 使用动态规划进行求解
     * 首先初始化 长度为 n + 1 的数组 dp，每个位置都为 0
     * 如果 n 为 0，则结果为 0
     * 对数组进行遍历，下标为i，每次都将当前数字先更新为 最大的结果，即 dp[i] = i
     * 动态转移方程： dp[i] = MIN(dp[i], dp[i - j * j] + 1), i 表示当前数字， j * j 表示平方数
     *
     * 对应 leetcode 中第 279题。
     */
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = i;   // 最坏的情况就是 每次 +1
            for (int j = 1; i - j * j >= 0; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

    // -------完全平方数 << end --------

    // -------字符串解码 start >>--------

    /**
     * 给定一个经过编码的字符串，返回它解码后的字符串。
     * 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
     * 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
     * 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
     *
     * 对应 leetcode 中第 394 题
     */
    public String decodeString(String s) {
        StringBuilder res = new StringBuilder();
        int multi = 0;
        Deque<Integer> stack_multi = new LinkedList<>();
        Deque<String> stack_res = new LinkedList<>();
        for (Character c : s.toCharArray()) {
            if (c == '[') {
                stack_multi.addLast(multi);
                stack_res.addLast(res.toString());
                multi = 0;
                res.delete(0, res.length());
            } else if (c == ']') {
                StringBuilder tmp = new StringBuilder();
                int cur_multi = stack_multi.removeLast();
                for (int i = 0; i < cur_multi; i++) {
                    tmp.append(res);
                }
                res = new StringBuilder(stack_res.removeLast() + tmp);
            } else if (Character.isDigit(c)) {
                multi = multi * 10 + (c - '0');
            } else {
                res.append(c);
            }
        }
        return res.toString();
    }

    // -------字符串解码 << end --------

    // -------回文子串 start >>--------

    /**
     * 给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
     * 回文字符串 是正着读和倒过来读一样的字符串。
     * 子字符串 是字符串中的由连续字符组成的一个序列。
     * 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
     *
     * 使用动态规划实现
     * dp[i][j] 表示字符串 s 在[i, j] 区间的子串是否是一个回文串。
     * 状态转移方程：当 s[i] = s[j] && (j - i < 2 || dp[i + 1][j - 1]) 时， dp[i][j] = true, 否则为 false
     *
     * 对应 leetcode 中第 647 题
     */
    public int countSubstrings(String s) {
        boolean[][] dp = new boolean[s.length()][s.length()];
        int ans = 0;
        for (int j = 0; j < s.length(); j++) {
            for (int i = 0; i <= j; i++) {
                if (s.charAt(i) == s.charAt(j) && (j - i < 2 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                    ans++;
                }
            }
        }
        return ans;
    }

    // -------回文子串 << end --------

    // -------最短无序连续子数组 start >>--------

    /**
     * 给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
     * 请你找出符合题意的 最短 子数组，并输出它的长度。
     *
     * 假设把这个数组分成三段，左段和右段是标准的升序数组，中序数组虽是无序的，但满足最小值大于 左段的最大值，最大值小于右段的最小值。
     * 那么我们就可以找 中断的左右边界，我们分别定义为 begin 和 end；分两头开始遍历：
     * 从左到右维护一个最大值 max， 在进入右段之前，那么遍历到的 nums[i] 都是小于 max 的， 我们要求的 end 就是遍历中最后一个小于 max元素的位置。
     * 同理，从右到左维护一个最小值 min，在进入左段之前，那么遍历到的 nums[i] 也都是大于 min 的，要求的 begin 也就是最后一个大于 min 元素的位置。
     *
     * 对应 leetcode 中第 581 题。
     */
    public int findUnsortedSubArray(int[] nums) {
        int len = nums.length, min = nums[len - 1], max = nums[0];
        int begin = 0, end = -1;
        for (int i = 0; i < len; i++) {
            if (nums[i] < max) {
                // 从左到右维护最大值，寻找右边界
                end = i;
            } else {
                max = nums[i];
            }
            if (nums[len - 1 - i] > min) {
                // 从右到左维护最小值，寻找左边界
                begin = len - i -1;
            } else {
                min = nums[len - i - 1];
            }
        }
        return end - begin + 1;
    }

    // -------最短无序连续子数组 << end --------

    // -------找到字符串中所有字母异位词 start >>--------

    /**
     * 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
     * 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
     *
     * 使用 滑动窗口 方法实现
     *
     * 对应 leetcode 中第 438 题
     */
    public List<Integer> findAnagrams(String s, String p) {
        int n = s.length(), m = p.length();
        List<Integer> res = new ArrayList<>();
        if (n < m)
            return res;
        int[] pCnt = new int[26];
        int[] sCnt = new int[26];

        for (int i = 0; i < m; i++) {
            pCnt[p.charAt(i) - 'a']++;
        }
        int left = 0;
        for (int right = 0; right < n; right++) {
            int curRight = s.charAt(right) - 'a';
            sCnt[curRight]++;
            while (sCnt[curRight] > pCnt[curRight]) {
                int curLeft = s.charAt(left) - 'a';
                sCnt[curLeft]--;
                left++;
            }
            if (right - left + 1 == m) {
                res.add(left);
            }
        }
        return res;
    }

    // -------找到字符串中所有字母异位词 << end --------

    // -------每日温度 start >>--------

    /**
     * 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指在第 i 天之后，
     * 才会有更高的温度。如果气温在这之后都不会升高，请在该位置用 0 来代替。
     *
     * 使用 单调栈的方法进行实现。
     *
     * 对应 leetcode 中第 739题。
     */
    public int[] dailyTemperatures(int[] temperatures) {
        int[] res = new int[temperatures.length];
        if (temperatures.length == 0) {
            return res;
        }
        Deque<Integer> stack = new LinkedList<>();
        stack.offerLast(temperatures.length - 1);

        for (int i = temperatures.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && temperatures[stack.getLast()] <= temperatures[i]) {
                stack.removeLast();
            }
            if (stack.isEmpty()) {
                res[i] = 0;
            } else {
                res[i] = stack.getLast() - i;
            }
            stack.offerLast(i);
        }
        return res;
    }

    public int[] dailyTemperatures_v2(int[] temperatures) {
        int length = temperatures.length;
        int[] ans = new int[length];
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < length; i++) {
            int temperature = temperatures[i];
            while (!stack.isEmpty() && temperature > temperatures[stack.peek()]) {
                int preIndex = stack.pop();
                ans[preIndex] = i - preIndex;
            }
            stack.push(i);
        }
        return ans;
    }

    // -------每日温度 << end --------

    // -------寻找重复数 start >>--------

    /**
     * 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
     * 假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。
     * 你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
     *
     * 使用环形链表的方式解答此题：
     * 如果数组中没有重复的数，以数组 [1,3,4,2] 为例，我们将数组下标 n 和数 nums[n] 建立一个映射关系 f(n)，为：
     * 0 -> 1 , 1 -> 3 , 2 -> 4, 3 -> 2.
     * 我们从下标0 出发，根据 f(n) 计算出一个值，以这个值为新的下标，再用这个函数计算，可以产生一个类似链表的序列。
     * 0 -> 1 -> 3 -> 2 -> 4 -> null.
     * 如果数组中有重复的数，以数组 [1,3,4,2,2] 为例，我们将数组下标 n 和数 nums[n] 建立一个映射关系 f(n)，为：
     * 0 -> 1 , 1 -> 3 , 2 -> 4, 3 -> 2, 4 -> 2.
     * 同样的，可以产生一个类似链表一样的序列。
     * 0 -> 1 -> 3 -> 2 -> 4 -> 2 -> 4 -> 2 ...
     * 这里的 2 -> 4 是一个循环。
     *
     * 综上，可以得出两个结论：
     * 1. 数组中有一个重复的数 <==> 链表中存在环
     * 2. 找到数组中的重复数 <==> 找到链表的环入口
     *
     * 对应 leetcode 中第 287 题。
     */
    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        int pre = 0;
        while (pre != slow) {
            pre = nums[pre];
            slow = nums[slow];
        }
        return pre;
    }

    // -------寻找重复数 << end --------

    // -------两个大数进行相加 start >>--------

    public String sumOfTwoString(String s1, String s2) {
        int s1Length = s1.length(), s2Length = s2.length();
        String longStr = s1Length > s2Length ? s1 : s2;
        String shortStr = s1.equals(longStr) ? s2 : s1;
        int gap = longStr.length() - shortStr.length();

        StringBuilder sb = new StringBuilder();
        int prev = 0;
        for (int i = shortStr.length() - 1; i >= 0; i--) {
            int shortValue = shortStr.charAt(i) - '0';
            int longValue = longStr.charAt(i + gap) - '0';
            int temp = shortValue + longValue + prev;
            prev = temp / 10;
            sb.append(temp % 10);
        }
        for (int i = longStr.length() - shortStr.length() - 1; i >= 0; i--) {
            int temp = longStr.charAt(i) - '0' + prev;
            sb.append(temp % 10);
            prev = temp / 10;
        }
        if (prev != 0) {
            sb.append(prev);
        }
        return sb.reverse().toString();
    }

    // -------两个大数进行相加 << end --------


    ///////-------------helper class-------------------

    public static class State {
        // 图节点的id
        int id;

        // 从start节点到当前节点的距离
        int distFromStart;

        public State(int id, int distFromStart) {
            this.id = id;
            this.distFromStart = distFromStart;
        }
    }

}
