package com.study.leetcode;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

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

    /**
     * 对应 leetcode 中第 116 题
     */
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

    // -------填充每个节点的下一个右侧节点指针II start >>--------

    /**
     * 对应 leetcode 中第 117 题
     */
    public Node connect_Double(Node root) {
        if (root == null) {
            return null;
        }
        // cur 我们可以把他看成是每一层的链表
        Node cur = root;
        while (cur != null) {
            // 遍历当前层的时候，为了方便操作，在下一层添加一个哑节点（注意这里是访问当前层的节点，然后把下一层的节点串起来）
            Node dummy = new Node(0);
            // pre表示访问下一层节点的前一个节点
            Node pre = dummy;
            // 然后开始遍历当前层的链表
            while (cur != null) {
                if (cur.left != null) {
                    // 如果当前节点的左子节点不为空，就让pre节点的next指向他，也就是把它串起来
                    pre.next = cur.left;
                    // 更新pre
                    pre = pre.next;
                }
                // 同理参照右子树
                if (cur.right != null) {
                    pre.next = cur.right;
                    pre = pre.next;
                }
                // 继续访问这一行的下一个节点
                cur = cur.next;
            }
            // 把下一层串联成一个链表之后，让他赋值给cur，后续继续循环，直到cur为空为止。
            cur = dummy.next;
        }
        return root;
    }

    // -------填充每个节点的下一个右侧节点指针II << end --------

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
     * 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
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

    // -------二叉树的中序遍历 start >>--------

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || root != null) {
            if (root != null) {
                stack.push(root);
                root = root.left;
            } else {
                TreeNode temp = stack.pop();
                result.add(temp.val);
                root = temp.right;
            }
        }
        return result;
    }

    // -------二叉树的中序遍历 << end --------

    // -------恢复二叉搜索树 start >>--------

    /**
     * 给你二叉搜索树的根节点 root ，该树中的两个节点的值被错误地交换。请在不改变其结构的情况下，恢复这棵树。
     *
     * 解题思路：
     * 首先按照中序遍历的方法进行遍历，找到两个需要交换的node，然后进行交换就行了，主要是如何找到两个需要交换的node，
     * 例如对于序列 1 2 3 7 5 6 4，如何找到 node 7 与 4，
     * 分别记 需要交换的节点为 x, y, 如下图：
     * 1 2 3 7 5 6 4
     *       x     y
     * 用 pre 来记录上一次访问的node节点，如果满足 当前节点的值小于 上一次pre的值，那么就记录当前节点的值为 y，
     * 上一次访问的值为 x，并且x只能记录一次，那么就能满足要求，遂有如下代码。
     *
     * 对应 leetcode 中 99 题
     *
     * @param root root tree node
     */
    public void recoverTree(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null, y = null, x = null;
        while (!stack.isEmpty() || root != null) {
            if (root != null) {
                stack.push(root);
                root = root.left;
            } else {
                root = stack.pop();
                if (pre != null && root.val < pre.val) {
                    y = root;
                    if (x == null) {
                        x = pre;
                    } else {
                        break;
                    }
                }
                pre = root;
                root = root.right;
            }
        }
        // swap x and y value
        int temp = y.val;
        y.val = x.val;
        x.val = temp;
    }

    // -------恢复二叉搜索树 << end --------

    // -------相同的树 start >>--------

    /**
     * 对应 leetcode 中的第 100 题
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null || p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    // -------相同的树 << end --------

    // -------对称二叉树 start >>--------

    /**
     * 对应 leetcode 中的第101题
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null)
            return true;
        return doIsSymmetric(root.left, root.right);
    }

    private boolean doIsSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null)
            return true;
        if (left == null || right == null || left.val != right.val)
            return false;
        return doIsSymmetric(left.left, right.right)
                && doIsSymmetric(left.right, right.left);
    }

    public boolean isSymmetric_v2(TreeNode root) {
        if (root == null || (root.left == null && root.right == null))
            return true;
        Queue<TreeNode> queue = new LinkedList<>();
        // 将根节点的左右孩子节点放到队列中
        queue.offer(root.left);
        queue.offer(root.right);

        while (!queue.isEmpty()) {
            TreeNode left = queue.poll();
            TreeNode right = queue.poll();
            if (left == null && right == null)
                continue;
            if (left == null || right == null || left.val != right.val)
                return false;
            // 将左孩子的左孩子， 右孩子的右孩子放入到队列中
            queue.offer(left.left);
            queue.offer(right.right);
            // 将左孩子的右孩子， 右孩子的左孩子放入到队列中
            queue.offer(left.right);
            queue.offer(right.left);
        }
        return true;
    }

    // -------对称二叉树 << end --------

    // -------二叉树的层序遍历 start >>--------

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new LinkedList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new ArrayList<>(size);
            for (int i = 0; i < size; i++) {
                TreeNode current = queue.poll();
                list.add(current.val);
                if (current.left != null) {
                    queue.offer(current.left);
                }
                if (current.right != null) {
                    queue.offer(current.right);
                }
            }
            result.add(list);
        }
        return result;
    }

    // -------二叉树的层序遍历 << end --------

    // -------二叉树的锯齿形层序遍历 start >>--------

    /**
     * 给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
     *
     * 对应 leetcode 中第 103 题。
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new LinkedList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean isOrderLeft = true;
        while (!queue.isEmpty()) {
            int size = queue.size();
            Deque<Integer> levelList = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode current = queue.poll();
                if (isOrderLeft) {
                    levelList.offerLast(current.val);
                } else {
                    levelList.offerFirst(current.val);
                }
                if (current.left != null) {
                    queue.offer(current.left);
                }
                if (current.right != null) {
                    queue.offer(current.right);
                }
            }
            result.add(new LinkedList<>(levelList));
            isOrderLeft = !isOrderLeft;
        }
        return result;
    }

    // -------二叉树的锯齿形层序遍历 << end --------

    // -------路径总和 start >>--------

    /**
     * 给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，
     * 这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。
     * 叶子节点 是指没有子节点的节点。
     *
     * 对应 leetcode 中第 112 题
     *
     * @param root root tree node
     * @param targetSum target sum
     * @return whether has path
     */
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null)
            return false;
        if (root.left == null && root.right == null)
            return targetSum == root.val;
        return hasPathSum(root.left, targetSum - root.val)
                || hasPathSum(root.right, targetSum - root.val);
    }

    // -------路径总和 << end --------

    // -------路径总和II start >>--------

    /**
     * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
     * 叶子节点 是指没有子节点的节点。
     *
     * 深度优先遍历思想，加上回溯。
     *
     * 对应 leetcode 中第 113 题
     */
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> result = new ArrayList<>();
        Deque<Integer> list = new LinkedList<>();
        pathSumToAdd(root, targetSum, result, list);
        return result;
    }

    private void pathSumToAdd(TreeNode root, int targetSum, List<List<Integer>> result, Deque<Integer> list) {
        if (root == null)
            return;
        list.offerLast(root.val);
        targetSum -= root.val;
        if (root.left == null && root.right == null && targetSum == 0) {
            result.add(new LinkedList<>(list));
        }
        pathSumToAdd(root.left, targetSum, result, list);
        pathSumToAdd(root.right, targetSum, result, list);
        list.pollLast();
    }

    /**
     * 广度优先遍历思想
     */
    public List<List<Integer>> pathSum_v2(TreeNode root, int targetSum) {
        List<List<Integer>> result = new LinkedList<>();
        if (root == null) {
            return result;
        }
        // Map 用来存放当前节点的父节点
        Map<TreeNode, TreeNode> map = new HashMap<>();
        Queue<TreeNode> queueNode = new LinkedList<>();
        Queue<Integer> queueSum = new LinkedList<>();
        queueNode.offer(root);
        queueSum.offer(0);

        while (!queueNode.isEmpty()) {
            TreeNode node = queueNode.poll();
            int rec = queueSum.poll() + node.val;
            if (node.left == null && node.right == null) {
                if (rec == targetSum) {
                    result.add(pathSumGetPath(node, map));
                }
            } else {
                if (node.left != null) {
                    map.put(node.left, node);
                    queueNode.offer(node.left);
                    queueSum.offer(rec);
                }
                if (node.right != null) {
                    map.put(node.right, node);
                    queueNode.offer(node.right);
                    queueSum.offer(rec);
                }
            }
        }
        return result;
    }

    private List<Integer> pathSumGetPath(TreeNode node, Map<TreeNode, TreeNode> map) {
        List<Integer> temp = new LinkedList<>();
        while (node != null) {
            temp.add(0, node.val);
            node = map.get(node);
        }
        return temp;
    }

    // -------路径总和II << end --------

    // -------二叉树中的最大路径和 start >>--------

    /**
     * 路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
     * 路径和 是路径中各节点值的总和。
     * 给你一个二叉树的根节点 root ，返回其 最大路径和 。
     *
     * 对应 leetcode 中第 124 题
     *
     * @param root root tree node
     * @return max path sum
     */
    public int maxPathSum(TreeNode root) {
        if (root == null)
            return 0;
        AtomicInteger globalSum = new AtomicInteger(Integer.MIN_VALUE);
        maxPathSumDfs(root, globalSum);
        return globalSum.get();
    }

    /**
     * 返回经过 root 的单边分支最大和，即 Math.max(root, root + left, root + right)
     *
     * @param root root tree node
     * @param globalSum global sum
     */
    private int maxPathSumDfs(TreeNode root, AtomicInteger globalSum) {
        if (root == null)
            return 0;
        // 计算左边分支最大值，左边分支如果为负数还不如不选择
        int leftGain = Math.max(maxPathSumDfs(root.left, globalSum), 0);
        // 计算右边分支最大值，右边分支如果为负数还不如不选择
        int rightGain = Math.max(maxPathSumDfs(root.right, globalSum), 0);
        // left->root->right 作为路径与已经计算过历史最大值做比较
        globalSum.set(Math.max(globalSum.get(), leftGain + rightGain + root.val));
        // 返回经过 root 的单边最大分支给当前 root 的父节点使用
        return root.val + Math.max(leftGain, rightGain);
    }

    // -------二叉树中的最大路径和 << end --------

    // -------将有序数组转换为二叉搜索树 start >>--------

    /**
     * 对应 leetcode 中第 108 题
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTHelper(int[] nums, int start, int end) {
        if (start > end)
            return null;
        int middle = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[middle]);
        root.left = sortedArrayToBSTHelper(nums, start, middle - 1);
        root.right = sortedArrayToBSTHelper(nums, middle + 1, end);
        return root;
    }

    // -------将有序数组转换为二叉搜索树 << end --------

    // -------二叉树的右视图 start >>--------

    /**
     * 我们按照 【根节点 -> 右子树 -> 左子树】的顺序访问，就可以保证每层都是最先被访问最右边的节点的。
     *
     * 对应 leetcode 中第 199 题
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        rightSideViewDfs(root, 0, res);
        return res;
    }

    private void rightSideViewDfs(TreeNode root, int depth, List<Integer> res) {
        if (root == null)
            return;
        // 先访问当前节点，在递归的访问右子树 和 左子树
        if (depth == res.size()) {
            // 如果当前节点所在深度还没有出现在 res 中，说明在该深度下当前节点是第一个被访问的节点，因此将当前节点加入 res 中
            res.add(root.val);
        }
        depth++;
        rightSideViewDfs(root.right, depth, res);
        rightSideViewDfs(root.left, depth, res);
    }

    // -------二叉树的右视图 << end --------

    // -------完全二叉树的节点个数 start >>--------

    /**
     * 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
     * 完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
     *
     * 对应 leetcode 中第 222 题
     */
    public int countNodes(TreeNode root) {
        if (root == null)
            return 0;
        int level = 0;
        TreeNode node = root;
        while (node.left != null) {
            level++;
            node = node.left;
        }
        int low = 1 << level, high = (1 << (level + 1)) - 1;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (countNodesExists(root, level, mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    private boolean countNodesExists(TreeNode root, int level, int k) {
        int bits = 1 << (level - 1);
        TreeNode node = root;
        while (node != null && bits > 0) {
            if ((bits & k) == 0) {
                node = node.left;
            } else {
                node = node.right;
            }
            bits >>= 1;
        }
        return node != null;
    }

    // -------完全二叉树的节点个数 << end --------

    // -------二叉树搜索子树的最大键值和 start >>--------

    /**
     * 给你一棵以 root 为根的 二叉树 ，请你返回 任意 二叉搜索子树的最大键值和。
     *
     * 二叉搜索树的定义如下：
     * 任意节点的左子树中的键值都 小于 此节点的键值。
     * 任意节点的右子树中的键值都 大于 此节点的键值。
     * 任意节点的左子树和右子树都是二叉搜索树。
     *
     * 解题思路：
     * 如果我们想计算子树中 BST 的最大和，站在当前节点的视角，需要做什么呢？
     * 1、我肯定得知道左右子树是不是合法的 BST，如果这俩儿子偶一个不是BST，以我为根的这棵树肯定不会是BST，
     * 2、如果左右子树都是合法的 BST，我得瞅瞅左右子树加上自己还是不是合法的 BST。因为按照 BST 的定义，当前节点的值应该大于左子树的
     *    最大值，小于右子树的最小值，否则就破坏BST 的性质。
     * 3、因为题目要计算最大的节点之和，如果左右子树加上我自己还是一颗合法的BST，也就是说以我为根的整棵树是一颗BST，那我需要知道我们这颗
     *    BST 的所有节点值之和是多少，方便和别的BST争个高下。
     *
     * 根据以上三点，站在当前节点的视角，需要知道以下具体信息：
     * 1、左右子树是否是 BST。
     * 2、左子树的最大值和 右子树的最小值。
     * 3、左右子树的节点值之和。
     *
     * 返回一个大小为 4 的int数组，我们称之为 res，其中：
     * res[0] 记录以 root 为根的二叉树是否是 BST，若为 1 则说明是 BST，若为 0 则说明不是 BST；
     * res[1] 记录以 root 为根的二叉树所有节点中的最小值；
     * res[2] 记录以 root 为根的二叉树所有节点中的最大值；
     * res[3] 记录以 root 为根的二叉树所有节点值之和。
     * 其实就是之前分析中说到的几个值放到了 res 数组中，最重要的是，我们要试图通过 left 和 right 正确推导出 res 数组。
     *
     * 对应 leetcode 总第 1373 题
     */
    public int maxSumBST(TreeNode root) {
        AtomicInteger maxSum = new AtomicInteger();
        maxSumBSTTraverse(root, maxSum);
        return maxSum.get();
    }

    private int[] maxSumBSTTraverse(TreeNode root, AtomicInteger maxSum) {
        // base case
        if (root == null)
            return new int[] {
                    1, Integer.MAX_VALUE, Integer.MIN_VALUE, 0
            };
        // 递归计算左右子树
        int[] left = maxSumBSTTraverse(root.left, maxSum);
        int[] right = maxSumBSTTraverse(root.right, maxSum);

        /* 后序遍历位置 */
        int[] res = new int[4];
        // 这个if 在判断以 root为根的二叉树是不是 BST
        if (left[0] == 1 && right[0] == 1
                && root.val > left[2] && root.val < right[1]) {
            // 以 root 为根的二叉树是 BST
            res[0] = 1;
            // 计算以 root 为根的这颗BST的最小值
            res[1] = Math.min(left[1], root.val);
            // 计算以 root 为根的这颗BST的最大值
            res[2] = Math.max(right[2], root.val);
            // 计算以 root 为根的这颗BST的所有节点之和
            res[3] = left[3] + right[3] + root.val;
            // 更新全局变量
            maxSum.set(Math.max(maxSum.get(), res[3]));
        } else {
            // 以root为根的二叉树不是 BST
            res[0] = 0;
        }

        return res;
    }

    // -------二叉树搜索子树的最大键值和 << end --------

    // -------二叉树的序列化与反序列化 start >>--------

    /**
     * 前序遍历二叉树的序列化
     *
     * 对应 leetcode 中第 297 题
     */
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serializeToSB(root, sb);
        return sb.toString();
    }

    private void serializeToSB(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append("#").append(",");
            return;
        }
        sb.append(root.val).append(",");
        serializeToSB(root.left, sb);
        serializeToSB(root.right, sb);
    }

    /**
     * 前序遍历反序列化
     */
    public TreeNode deserialize(String data) {
        Deque<String> nodes = new LinkedList<>();
        for (String s : data.split(",")) {
            nodes.addLast(s);
        }
        return deserializeSB(nodes);
    }

    private TreeNode deserializeSB(Deque<String> nodes) {
        if (nodes.isEmpty())
            return null;
        String first = nodes.removeFirst();
        if ("#".equals(first))
            return null;
        TreeNode root = new TreeNode(Integer.parseInt(first));

        root.left = deserializeSB(nodes);
        root.right = deserializeSB(nodes);
        return root;
    }

    /**
     * 后序遍历 序列化
     */
    public String serialize_v2(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serializeToSB_v2(root, sb);
        return sb.toString();
    }

    private void serializeToSB_v2(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append("#").append(",");
            return;
        }
        serializeToSB_v2(root.left, sb);
        serializeToSB_v2(root.right, sb);
        sb.append(root.val).append(",");
    }

    /**
     * 后续遍历 反序列化
     */
    public TreeNode deserialize_v2(String data) {
        Deque<String> nodes = new LinkedList<>();
        for (String s : data.split(",")) {
            nodes.addLast(s);
        }
        return deserializeSB_v2(nodes);
    }

    private TreeNode deserializeSB_v2(Deque<String> nodes) {
        if (nodes.isEmpty())
            return null;
        String last = nodes.removeLast();
        if ("#".equals(last))
            return null;
        TreeNode root = new TreeNode(Integer.parseInt(last));
        root.right = deserializeSB_v2(nodes);
        root.left = deserializeSB_v2(nodes);
        return root;
    }

    /**
     * 层序遍历序列化
     */
    public String serialize_levelOrder(TreeNode root) {
        if (root == null)
            return "";
        StringBuilder sb = new StringBuilder();
        // 初始化队列，将root 加入队列
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);

        while (!q.isEmpty()) {
            TreeNode cur = q.poll();
            if (cur == null) {
                sb.append("#").append(",");
                continue;
            }
            sb.append(cur.val).append(",");
            q.offer(cur.left);
            q.offer(cur.right);
        }
        return sb.toString();
    }

    /**
     * 层序遍历 反序列化
     */
    public TreeNode deserialize_levelOrder(String data) {
        if (data.isEmpty())
            return null;
        String[] nodes = data.split(",");
        // 第一个元素就是 root 的值
        TreeNode root = new TreeNode(Integer.parseInt(nodes[0]));
        // 队列 q 记录父节点，将 root 加入队列
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);

        for (int i = 1; i < nodes.length;) {
            // 队列中存的都是父节点
            TreeNode parent = q.poll();
            // 父节点对应的左侧子节点的值
            String left = nodes[i++];
            if (!"#".equals(left)) {
                parent.left = new TreeNode(Integer.parseInt(left));
                q.offer(parent.left);
            } else {
                parent.left = null;
            }
            // 父节点对应的右侧子节点的值
            String right = nodes[i++];
            if (!"#".equals(right)) {
                parent.right = new TreeNode(Integer.parseInt(right));
                q.offer(parent.right);
            } else {
                parent.right = null;
            }
        }
        return root;
    }

    // -------二叉树的序列化与反序列化 << end --------

    // -------二叉树的最近公共祖先 start >>--------

    /**
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，
     * 满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     *
     * 解题思路：
     * 1、这个函数是干嘛的？
     * 我们可以这么定义这个函数：给该函数输入三个参数 root, p, q。然会返回一个节点。
     * case 1：如果p 和 q 都在以root为根的树中，函数返回的即是 p 和 q 的最近公共祖先节点。
     * case 2：如果 p 和 q 都不在以root为根的树中，函数返回的是 null。
     * case 3：如果 p 和 q 只有一个存在于 root 为根的树中，函数返回那个节点。
     * 2、函数的参数中，变量是什么？
     * 函数参数中的变量是root，因为这个函数会递归的调用 root.left 和 root.right；至于 q 和 p，我们要求他俩的公共祖先，肯定是不会变化的。
     * 3、得到函数的递归结果，你该干啥？
     * 先想base case，如果root为空，肯定返回null。如果root本身就是p 或者 q，即使有一个不存在于以root为根的树中，按照情况3的定义，也应该返回
     * root 节点。
     * 然后是真正的挑战了，用递归调用的结果 left 和 right 来搞点事情，我们可以分情况讨论：
     * case 1：如果 p 和 q 都在以 root 为根的树中，那么left 和 right一定分别是 p 和 q。
     * case 2：如果 p 和 q 都不在以 root 为根的树中，直接返回 null。
     * case 3：如果 p 和 q 只有一个存在于以 root 为根的树中，函数返回该节点。
     *
     * 对应 leetcode 中第236题。
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 只要当前根节点是 p 和 q中的任意一个，就返回
        if (root == null || root == q || root == p) return root;
        // 根节点不是 p 和 q中的任意一个，那么就继续分别往左子树和右子树中找 p 和 q
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        // p 和 a 都没找到，那就没有
        if (left == null && right == null) return null;
        // 左子树中没有 p，也没有q，就返回右子树的结果
        if (left == null) return right;
        // 右子树中没有 p，也没有q，就返回左子树中的结果
        if (right == null) return left;
        // 左右子树都找到 p 和 q了，那就说明p 和 q分别在左右两个子树上，所以此时的最近公共祖先就是 root
        return root;
    }

    // -------二叉树的最近公共祖先 << end --------

    // -------不同的二叉搜索树 start >>--------

    /**
     * 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数
     *
     * 解题思路：动态规划
     * 假设n个节点存在二叉排序树的个数是 G(n),令 f(i) 为以 i为根的二叉搜索树的个数，则
     * G(n) = f(1) + f(2) + f(3) + f(4) + ... + f(n)
     * 当 i 为根节点时，其左子树节点个数为 i - 1个，右子树节点为 n - i，则
     * f(i) = G(i - 1) * G(n - i)
     * 综合上面两个公式，可以得到：
     * G(n) = G(0) * G(n - 1) + G(1) * G(n - 2) + .. + G(n - 1) * G(0)
     *
     * 对应 leetcode 中第 96 题
     */
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            for (int j = 1; j < i + 1; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    // -------不同的二叉搜索树 << end --------

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
