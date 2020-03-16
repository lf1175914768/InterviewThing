package com.study.interview.dynamicprogramming;

/**
 * n个座位，n个人， 求每个人都不在自己座位上的种类数。
 * @author Liufeng
 * @createData Created on: 2018年9月11日 下午1:12:19
 */
public class NPersonSeatOtherPlace {

	/**
	 * 设dp[i]为 i个人，i个座位的所求结果，那么对于dp[i]的求法，有结果
	 *  dp[i] = (dp[i - 1] + dp[i - 2]) * (i - 1)
	 * 推导如下：
	 * 	假设一共有i个人，依次排列， 编号为从1到i，现在把第i个人拿出来， 往前找位置做，
	 * 一共有 （i- 1）种情况；现在假设第i个人找到一个位置，编号为 m，则对于m位置的安排，
	 * 有两种情况，现在分情况来说：
	 * 第一种情况，安排m坐到i的位置上， 剩下的 i - 2个人的情况和初始条件一样， 每个人都不能坐在自己的位置上，
	 * 一共有 i - 2个座位，（因为 m和i已经互换了位置，剩下 i - 2个位置）   也即 dp[i - 2]
	 * 第二种情况, 安排m坐到 除了i位置上的其他位置（当然还有m他自己的位置也不能坐，因为已经被i占用了），
	 * 这种情况下， 在 i - 1个人中（除了i之外的剩下的人），m不能坐到i位置上， i - 2个人不能做到自己的位置上，
	 * 发生了情况的一致， 也就是 dp[i - 1].
	 * 综上， 可以推导出   dp[i] = (dp[i - 1] + dp[i - 2]) * (i - 1)
	 * @param persons 人数， 座位数都是
	 * @return
	 */
	public int getNumbers(int persons) {
		//初始情况
		if(persons <= 1) return 0; 
		else if(persons == 2) return 1;
		int[] result = new int[persons];
		result[0] = 0; result[1] = 1;
		for(int i = 3; i <= persons; i++) {
			result[i - 1] = (i - 1) * (result[i - 2] + result[i - 3]);
		}
		return result[persons - 1];
 	}
	
}
