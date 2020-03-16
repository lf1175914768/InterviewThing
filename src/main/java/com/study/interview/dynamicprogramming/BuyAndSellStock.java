package com.study.interview.dynamicprogramming;

/**
 * Say you have an array for which the ith element is the price of a given stock on day i.
 * Design an algorithm to find the maximum profit. You may complete at most k transactions.
 * 
 * <pre>Note:</pre>
 * You may not engage in multiple transactions at the same time
 *  (ie, you must sell the stock before you buy again).
 *  
 * <pre>Example 1:</pre>
 * 	Input: [2,4,1], k = 2
 * 	Output: 2
 * 	Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.
 * 
 * <pre>Example 2:</pre>
 * 	Input: [3,2,6,5,0,3], k = 2
 * 	Output: 7
 * 	Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4.
             Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
 * @author LiuFeng
 */
public class BuyAndSellStock {
	
	
	/**
	 *     我们维护两种量， 
	 * The first one: 
	 *   local[i][j]表示当前到达第i天，最多可进行j次交易，并且最后一次交易在当天卖出的最好的利润， 称之为局部最优解
	 * The second one: 
	 * 	 global[i][j]表示当前到达第i天，最多可以进行j次交易，最好的利润， 称之为全局最优解
	 *      递推式：
	 *      global[i][j] = max(local[i][j], global[i - 1][j])
	 *   也就是说去当前局部最好的，和过往全局最好的中大的一个（因为最后一次交易如果包含当前天，那么一定在局部最优解里面；如果不包含当前天，
	 *   那么一定在全局最优解里面）
	 *   	local[i][j] = max(global[i - 1][j - 1] + max(diff, 0), local[i - 1][j] + diff)
	 *   也就是说看两个量，第一个是全局到 i - 1天进行j - 1次交易，然后加上今天的交易，如果今天是赚钱的话（也就是前面只要j - 1次交易，
	 *   最后一次交易取当前天），第二个量是取local第i天j次交易，然后加上今天的差值，（这里因为local[i - 1][j]比如包含第i - 1天卖出的交易
	 *   所以现在变成第i天卖出，并不会增加交易次数，而且这里无论diff是否大于0 都一定要加上， 因为否则就不满足local[i][j]必须在最后一天卖出的条件了）  
	 */
	public int maxProfit(int k, int[] prices) {
        if(prices.length == 0) return 0;
        if(k >= prices.length) return solveMaxProfit(prices);
        int[] global = new int[k + 1];
        int[] local = new int[k + 1];
        for(int i = 0; i < prices.length - 1; i++) {
        	int diff = prices[i + 1] - prices[i];
        	for(int j = k; j >= 1; j--) {
        		local[j] = Math.max(global[j - 1] + Math.max(diff, 0), local[j] + diff);
        		global[j] = Math.max(global[j], local[j]);
        	}
        }
        return global[k];
    }

	private int solveMaxProfit(int[] prices) {
		int res = 0;
		for(int i = 1; i < prices.length; i++) {
			if(prices[i] - prices[i - 1] > 0) {
				res += prices[i] - prices[i - 1];
			}
		}
		return res;
	}

}
