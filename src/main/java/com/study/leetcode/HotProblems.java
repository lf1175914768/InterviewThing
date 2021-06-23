package com.study.leetcode;

import com.sun.xml.internal.messaging.saaj.packaging.mime.util.LineInputStream;

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
                    if (beforeRev + arr[i][j] - 1 == i)  {
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

    /**
     * Manacher's Algorithm 马拉车算法
     *
     * 马拉车算法 是用来查找一个字符串的最长回文子串的线性方法， 由一个叫 manacher 的人再1975年发明的， 这个方法的最大贡献在于将时间复杂度提升到了线性。
     *
     * 首先我们解决下奇数和偶数的问题，在每个字符间插入 “#”， 并且为了是的扩展的过程中，到边界后自动结束，在两端分别插入 “^” 和 “$”，两个不可能
     * 在字符串中出现的字符，这样中心扩展的时候，判断两端字符是否相等的时候，如果到了边界就一定会不相等，从而出了循环。经过处理，字符串的长度永远都是奇数。
     *
     *                 ^ a # b # c # b # a # a # d # e $
     *
     * 首先我们用一个数组 P 保存从中心扩展的最大个数，而它刚好也是去掉 "#"的原字符串的总长度。例如下图中下标是6的地方，可以看到 P[6] 等于5， 所以它是从左边扩展
     * 5个字符，相应的右边也是扩展 5个字符，也就是 “#c#b#c#b#c#”。 而去掉#恢复到原来的字符串，变成 “cbcbc”， 它的长度刚好也就是5。
     * 其中，T: 处理后的数组；    P: 从中心扩展的长度
     *
     *                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
     *            T = ^ # c # b # c # b # c  #  c  #  d  #  e  #  $
     *            P = 0 0 1 0 3 0 5 0 3 0 1  2  1  0  1  0  1  0  0
     *
     * <p> 求原字符串下标 </p>
     * 用 P 的下标 i 减去 P[i]， 在除以2，就是原字符串的开头下标了。
     * 例如我们找到 P[i] 的最大值为5，也就是回文串的最大长度是5，对应的下标是6，所以原字符串的开头下标是 (6 - 5) / 2 = 0.所以我们就只需要返回原字符串的
     * 第 0 到 第 (5-1) 位就可以了。
     *
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
            put('2', new String[] {"a", "b", "c"});
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
        Queue<ListNode> queue = new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return Integer.compare(o1.val, o2.val);
            }
        });
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

    public int longestValidParentheses(String s) {
        return 0;
    }

    // -------最长有效括号 << end --------

    ///////-------------helper class-------------------

    public static class ListNode {
        int val;
        ListNode next;

        @Override
        public String toString() {
            return String.valueOf(val);
        }

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }
}
