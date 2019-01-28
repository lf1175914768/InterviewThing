package com.study.interview.tree;

public class DistributeTree {
	
	int res = 0;
	
	/**
	 *	<p>The Fact:</p>
	 *	If the leaf of a tree has 0 coins (an excess of -1 from what it needs), then we should push a coin 
	 * from its parent onto the leaf. If has 5 coins, (an excess of 4), then we should push 4 coins off the 
	 * leaf.In total, the number of moves from that leaf to or from its parent is      
	 * <pre> excess =  Math.abs(num_coins - 1).</pre> Afterwards, we never have to consider this leaf 
	 * again in the rest of our calculation.
	 * <p>Algorithm:</p>
	 * We can use the fact above to build our answer. Let <strong>dfs(node)</strong> be the excess number of 
	 * coins in the subtree at or below this node: namely, the number of the coins in the subtree, minus the number 
	 * of nodes in the subtree. Then, the number of moves we make from this node to and from its children is 
	 * <strong>abs(dfs(node.left)) + abs(dfs(node.right))</strong> After, we have an excess of 
	 * <strong>node.value + dfs(node.left) + dfs(node.right) - 1</strong> coins at this node.
	 */
	public int distributeCoins(Node root) {
		dfs(root);
		return res;
	}

	private int dfs(Node root) {
		if(root == null) return 0;
		int left = dfs(root.left), right = dfs(root.right);
		res += Math.abs(left) + Math.abs(right);
		return root.value + left + right - 1;
	}

}
