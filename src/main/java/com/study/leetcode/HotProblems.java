package com.study.leetcode;

import java.util.*;

/**
 * 热题
 */
public class HotProblems {

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
            int L = i + 1, R = len - 1;
            while (L < R) {
                int temp = curr + nums[L] + nums[R];
                if (temp == 0) {
                    List<Integer> list = new ArrayList<>();
                    list.add(curr);
                    list.add(nums[L]);
                    list.add(nums[R]);
                    result.add(list);
                    while (L < R && nums[L + 1] == nums[L]) L++;
                    while (L < R && nums[R - 1] == nums[R]) R--;
                    L++;
                    R--;
                } else if (temp < 0) {
                    L++;
                } else {
                    R--;
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

    // -------括号生成 start >>--------

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


    // -------括号生成 << end --------

    // -------下一个排列 start >>--------

    /**
     * 1、从后向前查找第一个相邻的元素对 (i, i + 1), 满足 A[i] < A[i + 1]。此时[i+1, end) 必然是降序
     * 2、在 [i + 1, end)中从后往前查找第一个满足 A[i] < A[k] 的k。 A[i], A[k] 分别就是【小数】和【大数】。
     * 3、将 A[i] 与 A[k] 交换
     * 4、可以断定这时 [i + 1, end) 必然是降序，重新排序 [i + 1, end) 使其升序。
     * 5、如果在步骤1中找不到符合的相邻元素时，说明当前 [begin, end) 为一个降序顺序，直接跳到 步骤4
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

    // -------组合总和 start >>--------

    public List<List<Integer>> permute(int[] nums) {
        return null;
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


    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------

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
