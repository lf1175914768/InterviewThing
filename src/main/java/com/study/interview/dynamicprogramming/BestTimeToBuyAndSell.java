package com.study.interview.dynamicprogramming;

/**
 * 最多只能有两个事务进行执行， 且不能同时拥有两个事务，必须等待一个事务结束， 另一个事务才能开始。
 * 例子： 
 *  Input： [3,3,5,0,0,3,1,4]
 *  Output: 6
 *  Explanation： 在第4天买入，然后在第6天卖出，差值是3， 然后在第7天买入， 在第8天卖出，差值是3.
 *     			 满足两个事务的要求， 且是最大值。
 * @author Liufeng
 * Created on 2018年9月9日 上午12:00:37
 */
public class BestTimeToBuyAndSell {
	
	/**
	 * 这里使用“局部最优和全局最优”的解法。我们维护两种量，
	 * 一个是当前到达第i天最多可以进行j次交易，最好的利润， 用 global[i][j]来表示。
	 * 另一个是当前到达第i天，最多可进行j次交易，并且最后一次交易在当天卖出的最好利润， 用local[i][j]来表示。
	 * 下面进行递推式，全局的比较简单：
	 * 		global[i][j] = max( local[i][j], global[i-1][j] )
	 * 也就是去当前局部最好的， 和过往全局最好的中选择较大的那个进行赋值，（因为最后一次交易如果包含当前天一定在局部最好的里面，否则一定在过往全局最好的里面）。
	 * 局部变量的递推式： 
	 * 		local[i][j] = max( global[i - 1][j - 1] + max(diff, 0), local[i - 1][j] + diff )
	 * 也就是看两个量， 第一个是全局到i - 1天进行j - 1次交易，然后加上今天的交易，如果今天是赚钱的话（也就是说前面只要i - 1次交易，最后一次交易取当前天），
	 * 第二个量是取 local第i - 1天j次交易，然后加上今天的差值（这里因为local[i - 1][j]比如包含第i - 1天卖出的交易，所以现在变成第i天卖出，并不会增加交易次数，
	 * 而且这里无论diff是不是大于0都一定要加上，因为否则就不满足local[i][j]必须在最后一天卖出的条件了）
	 * 
	 * 上面的算法中对于天数需要一次扫描，而每次要对交易次数进行递推式求解，所以时间复杂度是O(n*k)，如果是最多进行两次交易，
	 * 那么复杂度还是O(n)。空间上只需要维护当天数据皆可以，所以是O(k)，当k=2，则是O(1)。
	 * 
	 * 优化： 
	 *    local[i][j] = max(global[i - 1][j - 1], local[i - 1][j]) + diff
	 * 	对于local[i][j] 就是下面几种情况：
	 * 	1、 今天刚买的
	 * 		local[i][j] = global[i - 1][j - 1] 
	 * 		相当于什么也没干
	 *  2、 昨天买的
	 *  	local[i][j] = global[i - 1][j - 1] + diff
	 *  	相当于global[i - 1][j - 1]中的交易，加上今天干的这一票
	 *  3、更早之前买的
	 *  	local[i][j] = local[i - 1][j] + diff
	 *  	相当于昨天不卖了， 留到今天卖。
	 *  对于第一种情况是不需要考虑的，因为当天买当天卖是不会增加利润的，所以经过整理就是上面优化之后的内容。
	 * 
	 * 最终结果：
	 * local[i][j] = max(global[i - 1][j - 1], local[i - 1][j]) + diff
	 * global[i][j] = max( local[i][j], global[i-1][j] )
	 */
	public int maxProfit_1(int[] prices) {
		if(prices == null || prices.length == 0) return 0;
		int [] local = new int[3];
		int [] global = new int[3];
		for(int i = 0; i < prices.length - 1; i++) {
			int diff = prices[i + 1] - prices[i];
			for(int j = 2; j >= 1; j--) {
				local[j] = Math.max(global[j - 1], local[j]) + diff;  //这条语句和下条语句的顺序不能换。
				global[j] = Math.max(local[j], global[j]);
			}
		}
		return global[2];
	}
	
	
	public int maxProfit_2(int[] prices) {
		int firstBuy = Integer.MIN_VALUE, firstSell = 0;
		int secondBuy = Integer.MIN_VALUE, secondSell = 0;
		for(int curPrice : prices) {
			if(firstBuy < -curPrice) firstBuy = -curPrice;  // the max profit after you buy first stock.
			if(firstSell < firstBuy + curPrice) firstSell = firstBuy + curPrice;	// the max profit after you sell it.
			if(secondBuy < firstSell - curPrice) secondBuy = firstSell - curPrice;	// the max profit after you buy second stock.
			if(secondSell < secondBuy + curPrice) secondSell = secondBuy + curPrice; // the max profit after you sell the second stock.
		}
		return secondSell;
	}

}
