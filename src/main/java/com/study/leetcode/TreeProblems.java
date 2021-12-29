package com.study.leetcode;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class TreeProblems {

    // -------反转二叉树 start >>--------

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        // 前序遍历
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    // -------反转二叉树 << end --------

    // -------填充每一个节点的下一个右侧节点指针 start >>--------

    public Node connect(Node root) {
        if (root == null)
            return null;
        connectTwoNode(root.left, root.right);
        return root;
    }

    private void connectTwoNode(Node left, Node right) {
        if (left == null || right == null)
            return;
        left.next = right;
        // 链接相同父节点的两个子节点
        connectTwoNode(left.left, left.right);
        connectTwoNode(right.left, right.right);
        // 连接跨越父节点的两个子节点
        connectTwoNode(left.right, right.left);
    }


    // -------填充每一个节点的下一个右侧节点指针 << end --------

    // -------二叉树展开为链表 start >>--------

    /**
     * 给你二叉树的根结点 root ，请你将它展开为一个单链表：
     *
     * 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
     * 展开后的单链表应该与二叉树 先序遍历 顺序相同。
     *  
     * 对应leetcode 第 114 题
     * @param root root
     */
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        flatten(root.left);
        flatten(root.right);

        TreeNode right = root.right;
        // 将左子树作为右子树
        root.right = root.left;
        root.left = null;

        // 将原先的右子树接到当前右子树的末端
        TreeNode p = root;
        while (p.right != null) {
            p = p.right;
        }
        p.right = right;
    }

    public void flatten_v2(TreeNode root) {
        while (root != null) {
            if (root.left == null) {
                root = root.right;
            } else {
                TreeNode p = root.left;
                while (p.right != null) {
                    p = p.right;
                }
                p.right = root.right;
                root.right = root.left;
                root.left = null;
                root = root.right;
            }
        }
    }

    /**
     * 我们知道题目给定的顺序是先序遍历的顺序，所以我们能不能利用先序遍历的代码，每遍历一个节点，就将上一个节点的右指针指向更新为当前节点。
     * 先序遍历的顺序是 1 2 3 4 5 6，
     * 遍历到2，把 1 的右指针指向 2. 1 -> 2 3 4 5 6
     * 遍历到3，把 2 的右指针指向 3. 1 -> 2 -> 3 4 5 6
     * ... 一直进行下去似乎就解决了这个问题，但现实是残酷的，原因是我们把 1 的右指针指向 2，那么 1 原本的有孩子就丢失了，也就是5 找不到了
     * 所以我们可以逆序进行。我们依次遍历 6 5 4 3 2 1，然后每遍历一个节点就将当前节点的右指针更新为上一个节点。
     * 这样就不会有丢失孩子的问题了，因为更新当前的右指针的时候，当前节点的右孩子已经访问过了。而这个访问顺序其实是 变形的 后序遍历，遍历顺序是 右子树 -> 左子树 -> 根节点
     *
     * @param root root
     */
    public void flatten_v3(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        TreeNode pre = null;

        while (cur != null || !stack.isEmpty()) {
            while (cur != null) {
                stack.push(cur);
                // 递归添加右节点
                cur = cur.right;
            }
            // 已经访问到最右的节点了
            cur = stack.peek();
            // 在不存在左子节点或者左子节点已经访问过的情况下，访问根节点
            if (cur.left == null || cur.left == pre) {
                stack.pop();
                cur.right = pre;
                cur.left = null;
                pre = cur;
                cur = null;
            } else {
                cur = cur.left;
            }
        }
    }

    // -------二叉树展开为链表 << end --------

    // -------最大二叉树 start >>--------

    /**
     * 给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：
     *
     * 二叉树的根是数组 nums 中的最大元素。
     * 左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
     * 右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
     * 返回有给定数组 nums 构建的 最大二叉树 。
     *
     * 对应leetcode 中的 654 题
     *
     * @param nums list of nums
     * @return root tree node
     */
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructMaxBinaryTreeBuild(nums, 0, nums.length - 1);
    }

    private TreeNode constructMaxBinaryTreeBuild(int[] nums, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        int maxValue = Integer.MIN_VALUE, index = -1;
        for (int i = lo; i <= hi; i++) {
            if (nums[i] > maxValue) {
                maxValue = nums[i];
                index = i;
            }
        }
        TreeNode node = new TreeNode(maxValue);
        node.left = constructMaxBinaryTreeBuild(nums, lo, index - 1);
        node.right = constructMaxBinaryTreeBuild(nums, index + 1, hi);
        return node;
    }

    // -------最大二叉树 << end --------

    // -------从前序与中序遍历序列构造二叉树 start >>--------

    /**
     * 给定一棵树的前序遍历 preorder 与中序遍历  inorder。请构造二叉树并返回其根节点。
     *
     * 对应 leetcode 中第 105 题
     *
     * @param preorder 前序遍历
     * @param inorder 中序遍历
     * @return root tree node
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTreeToBuild(preorder, 0, preorder.length - 1,
                inorder, 0, inorder.length - 1);
    }

    private TreeNode buildTreeToBuild(int[] preorder, int preStart, int preEnd,
                                      int[] inorder, int inStart, int inEnd) {
        if (preStart > preEnd) {
            return null;
        }
        int rootValue = preorder[preStart], index = -1;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == rootValue) {
                index = i;
                break;
            }
        }
        int leftSize = index - inStart;
        TreeNode root = new TreeNode(rootValue);
        root.left = buildTreeToBuild(preorder, preStart + 1, preStart + leftSize,
                                    inorder, inStart, index - 1);
        root.right = buildTreeToBuild(preorder, preStart + leftSize + 1, preEnd,
                                    inorder, index + 1, inEnd);
        return root;
    }

    public TreeNode buildTree_v2(int[] preOrder, int[] inOrder) {
        if (preOrder.length == 0) {
            return null;
        }
        Stack<TreeNode> roots = new Stack<>();
        int pre = 0, in = 0;
        // 先序遍历第一个值作为根节点
        TreeNode curRoot = new TreeNode(preOrder[pre]);
        TreeNode root = curRoot;
        roots.push(curRoot);
        pre++;
        // 遍历前序遍历的数组
        while (pre < preOrder.length) {
            // 出现了当前节点的值和中序遍历数组的值相等，寻找是谁的右子树
            if (curRoot.val == inOrder[in]) {
                // 每次进行出栈，实现倒着遍历
                while (!roots.isEmpty() && roots.peek().val == inOrder[in]) {
                    curRoot = roots.pop();
                    in++;
                }
                // 设为当前的右孩子
                curRoot.right = new TreeNode(preOrder[pre]);
                // 更新curRoot
                curRoot = curRoot.right;
            } else {
                // 否则的话，就一直作为左子树
                curRoot.left = new TreeNode(preOrder[pre]);
                curRoot = curRoot.left;
            }
            roots.push(curRoot);
            pre++;
        }
        return root;
    }

    // -------从前序与中序遍历序列构造二叉树 << end --------

    // -------从中序与后序遍历序列构造二叉树 start >>--------

    /**
     * 根据一棵树的中序遍历与后序遍历构造二叉树。
     *
     * 注意:
     * 你可以假设树中没有重复的元素
     *
     * 对饮leetcode 中的第 106题
     *
     * @param inorder 中序遍历的数组
     * @param postorder 后续遍历的数组
     * @return root of tree node
     */
    public TreeNode buildTreeByPost(int[] inorder, int[] postorder) {
        return buildTreeByPostToBuild(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1);
    }

    private TreeNode buildTreeByPostToBuild(int[] inorder, int inStart, int inEnd,
                                            int[] postOrder, int postStart, int postEnd) {
        if (postStart > postEnd) {
            return null;
        }
        // root 节点对应的值就是后序遍历数组的最后一个元素
        int rootVal = postOrder[postEnd], index = -1;
        for (int i = inStart; i <= inEnd; i++) {
            if (rootVal == inorder[i]) {
                index = i;
                break;
            }
        }
        int leftSize = index - inStart;
        TreeNode root = new TreeNode(rootVal);
        root.left = buildTreeByPostToBuild(inorder, inStart, index - 1,
                postOrder, postStart, postStart + leftSize - 1);
        root.right = buildTreeByPostToBuild(inorder, index + 1, inEnd,
                postOrder, postStart + leftSize, postEnd - 1);
        return root;
    }

    // -------从中序与后序遍历序列构造二叉树 << end --------

    // -------二叉搜索树中第K小的元素 start >>--------

    /**
     * 给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。
     *
     * 对应leetcode 中的第 230 题
     *
     * @param root root node of tree
     * @param k k
     * @return value
     */
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode head = root;
        int index = 0;
        while (!stack.isEmpty() || head != null) {
            if (head != null) {
                stack.push(head);
                head = head.left;
            } else {
                index++;
                head = stack.pop();
                if (index == k) {
                    return head.val;
                }
                head = head.right;
            }
        }
        return -1;
    }

    // -------二叉搜索树中第K小的元素 << end --------

    // -------把二叉搜索树转换为累加树 start >>--------

    /**
     * 给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
     *
     * 提醒一下，二叉搜索树满足下列约束条件：
     *
     * 节点的左子树仅包含键 小于 节点键的节点。
     * 节点的右子树仅包含键 大于 节点键的节点。
     * 左右子树也必须是二叉搜索树。
     *
     * 对应 leetcode 中第 538题
     *
     * @param root root node
     * @return root node
     */
    public TreeNode convertBST(TreeNode root) {
        convertBSTTraverse(root, 0);
        return root;
    }

    private int convertBSTTraverse(TreeNode root, int sum) {
        if (root == null) {
            return sum;
        }
        root.val += convertBSTTraverse(root.right, sum);
        return convertBSTTraverse(root.left, root.val);
    }

    // -------把二叉搜索树转换为累加树 << end --------

    // -------验证二叉搜索树 start >>--------

    /**
     * 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
     *
     * 有效 二叉搜索树定义如下：
     *
     * 节点的左子树只包含 小于 当前节点的数。
     * 节点的右子树只包含 大于 当前节点的数。
     * 所有左子树和右子树自身必须也是二叉搜索树。
     *
     * 对应 leetcode 中第 98题
     *
     * @param root root node
     * @return result of validation
     */
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, null, null);
    }

    private boolean isValidBST(TreeNode root, TreeNode min, TreeNode max) {
        // base case
        if (root == null)
            return true;
        if (min != null && root.val <= min.val)
            return false;
        if (max != null && root.val >= max.val)
            return false;
        return isValidBST(root.left, min, root)
                && isValidBST(root.right, root, max);
    }

    // -------验证二叉搜索树 << end --------

    // -------二叉搜索树中的插入操作 start >>--------

    /**
     * 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。
     * 注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。
     *
     * 对应 leetcode 中第 701 题
     *
     * @param root root tree node
     * @param val candidate to insert
     * @return root tree node
     */
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            return new TreeNode(val);
        }
        if (root.val > val) {
            root.left = insertIntoBST(root.left, val);
        }
        if (root.val < val) {
            root.right = insertIntoBST(root.right, val);
        }
        return root;
    }

    // -------二叉搜索树中的插入操作 << end --------

    // -------删除二叉搜索树中的节点 start >>--------

    /**
     * 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
     * 一般来说，删除节点可分为两个步骤：
     * 首先找到需要删除的节点；
     * 如果找到了，删除它。
     *
     * 对于解题思路，主要讲解找到目标节点了，如何删除这个节点。因为删除这个节点不能破坏BST 的性质，所以有三种情况。
     * <ol>
     *    <li>恰好湿末端节点，两个子节点都为空，那么它可以当场去世了</li>
     *    <li>只有一个非空节点，那么它要让这个孩子接替自己的位置</li>
     *    <li>有两个子节点，为了不破坏BST的性质，必须找到左子树中最大的那个节点；或者右子树中最小的那个节点来接替自己</li>
     * </ol>
     *  
     *
     * 对应 leetcode 中第 450题
     *
     * @param root root node
     * @param key key
     * @return root tree node
     */
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null)
            return null;
        if (root.val == key) {
            // 这两个if 将情况1 和 情况2 都处理了
            if (root.left == null)
                return root.right;
            if (root.right == null)
                return root.left;

            // 处理情况3，获得右子树中最小的节点
            TreeNode minNode = root.right;
            while (minNode.left != null)
                minNode = minNode.left;
            // 删除右子树中最小的节点
            root.right = deleteNode(root.right, minNode.val);
            // 用右子树最小的节点替换root节点
            minNode.left = root.left;
            minNode.right = root.right;
            root = minNode;
        }
        if (root.val > key) {
            root.left = deleteNode(root.left, key);
        } else if (root.val < key) {
            root.right = deleteNode(root.right, key);
        }
        return root;
    }

    // -------删除二叉搜索树中的节点 << end --------

    // -------不同的二叉搜索树II start >>--------

    /**
     * 给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。
     *
     * 解析思路：
     * 可以利用一下查找二叉树的性质。左子树的所有值小于根节点，右子树的所有值大于根节点。
     * 如果求 1...n的所有可能。只需要把 1 作为根节点，[]作为左子树，[2..n] 的所有可能作为右子树。
     * 把 2 作为根节点，[1]作为左子树，[3..n] 的所有可能作为右子树。
     * 把 3 作为根节点，[1,2]作为左子树，[4..n] 的所有可能作为右子树。
     * ...
     * 把 n 作为根节点，[1..n-1]作为左子树，[] 的所有可能作为右子树。
     *
     * 至于，[2..n]的所有可能以及其他情况的所有可能，可以利用上面的方法，把每个数字作为根节点，然后把所有可能的左子树和右子树组合起来即可。
     * 如果只有一个数字，那么所有可能就是一种情况，把该数字作为一棵树，而如果湿 []， 那就返回null
     */
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> ans = new ArrayList<>();
        if (n == 0) {
            return ans;
        }
        return getAns(1, n);
    }

    private List<TreeNode> getAns(int start, int end) {
        List<TreeNode> ans = new ArrayList<>();
        // 此时没有数字，将 null 加入结果中
        if (start > end) {
            ans.add(null);
            return ans;
        }
        // 只有一个数字，当前数字作为一棵树加入结果中
        if (start == end) {
            TreeNode tree = new TreeNode(start);
            ans.add(tree);
            return ans;
        }
        // 尝试每个数字作为根节点
        for (int i = start; i <= end; i++) {
            // 得到所有可能的左子树
            List<TreeNode> leftTrees = getAns(start, i - 1);
            // 得到所有可能的右子树
            List<TreeNode> rightTrees = getAns(i + 1, end);
            // 左子树和右子树两两组合
            for (TreeNode leftTree : leftTrees) {
                for (TreeNode rightTree : rightTrees) {
                    TreeNode root = new TreeNode(i);
                    root.left = leftTree;
                    root.right = rightTree;
                    // 加入到最终结果中
                    ans.add(root);
                }
            }
        }
        return ans;
    }

    /**
     * 利用上面的解法，就是分别把每个数字作为根节点，然后考虑左子树和右子树的可能，
     * 就是求出长度为 1 的所有可能，长度为 2 的所有可能... 直到 n。
     * 但是我们注意到，求长度为 2 的所有可能的时候，我们需要求 [1 2] 的所有可能，[2 3] 的所有可能，这只是 n = 3 的情况。如果 n = 100，我们需要
     * 求的更多了，[1 2],[2 3],[3 4]...[99 100] 太多了。能不能优化呢？
     * 仔细观察，我们可以发现长度是 2 的所有可能其实只有两种结构。
     * x             y
     *  \           /
     *   y         x
     * 看之前推导的 [1 2] 和 [2 3]， 只是数字不一样，结构是完全一样的。
     * 所以我们 n = 100 的时候，求长度是 2 的所有情况的时候，我们没必要把 [1 2], [2 3], [3 4]...[99 100] 所有的情况都求出来，只需要将 [1 2]
     * 的所有情况求出即可。
     * 推广到任意长度 len， 我们只需要求 [1 2 3 ... len] 的所有情况就可以了。下一个问题随之而来，这些 [2 3], [3 4] ... [99 100] 没求的怎么办呢？
     * 举个例子。 n = 100，此时我们求把 98 作为根节点的所有情况，根据之前的推导，我们需要长度是 97 的 [1 2 3 ...97] 的所有情况作为左子树，
     * 长度是 2 的 [99 100] 的所有情况作为右子树。
     * [1 2 3 ... len] 的所有情况刚好是 [1 2 ..len] ，已经求出来了，但 [99 100] 怎么办呢？ 我们只求了 [1 2] 的所有情况。
     * 答案很明显了，在 [1 2] 的所有情况每个数字加一个偏差 98， 即加上根节点的值就可以了。
     */
    public List<TreeNode> generateTrees_v2(int n) {
        ArrayList<TreeNode>[] dp = new ArrayList[n + 1];
        dp[0] = new ArrayList<>();
        if (n == 0) {
            return dp[0];
        }
        dp[0].add(null);
        // 长度为 1 到 n
        for (int len = 1; len <= n; len++) {
            dp[len] = new ArrayList<>();
            // 将不同的数字作为根节点，只需要考虑到 len
            for (int root = 1; root <= len; root++) {
                int left = root - 1;   // 左子树的长度
                int right = len - root;    // 右子树的长度
                for (TreeNode leftTree : dp[left]) {
                    for (TreeNode rightTree : dp[right]) {
                        TreeNode treeRoot = new TreeNode(root);
                        treeRoot.left = leftTree;
                        // 克隆右子树并且加上偏差
                        treeRoot.right = generateTreeClone(rightTree, root);
                        dp[len].add(treeRoot);
                    }
                }
            }
        }
        return dp[n];
    }

    private TreeNode generateTreeClone(TreeNode n, int offset) {
        if (n == null)
            return null;
        TreeNode node = new TreeNode(n.val + offset);
        node.left = generateTreeClone(n.left, offset);
        node.right = generateTreeClone(n.right, offset);
        return node;
    }

    // -------不同的二叉搜索树II << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------

    ///////-------------helper class-------------------

    static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int val, Node left, Node right) {
            this(val, left, right, null);
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }


}
