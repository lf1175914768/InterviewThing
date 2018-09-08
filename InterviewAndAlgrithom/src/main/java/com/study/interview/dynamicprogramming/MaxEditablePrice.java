package com.study.interview.dynamicprogramming;

/**
 * 最小编辑代价  将str1 编辑称为 str2 需要的最小编辑代价
 * 其中包括 增加、 删除、 替换， 分别代表不同的编辑代价
 * @author LiuFeng
 *
 */
public class MaxEditablePrice {
	
	/**
	 * str1[0..i-1] 可以先编辑成 str1[0..i-2], 删除字符str1[i-1], 
	 * 然后由str1[0..i-2]编辑成str2[0..j-1], 即dp[i][j] = dc + dp[i-1][j];
	 * 
	 * str1[0..i-1] 可以先编辑成 str2[0..i-2], 然后将str2[0..j-2] 插入字符 str2[j-1],编辑成str2[0..j-1],
	 * 那么 dp[i][j-1] + ic;
	 * 
	 * 如果str1[i-1] != str2[j-1]，先把str1[0..i-1]中str1[0..j-2]部分变成str2[0..j-2], 然后
	 * 把字符str1[i-1]替换成 str2[i-1], 那么dp[i][j] = dp[i -1][j - 1] + rc;
	 * 
	 * 如果 str1[i-1] == str2[j-1], 那么就直接有dp[i][j] = dp[i - 1][j  - 1];
	 * 
	 * 求上述情况中的最小满足者
	 * @param str1
	 * @param str2
	 * @param ic
	 * @param dc
	 * @param rc
	 * @return
	 */
	public int minCost1(String str1, String str2, int ic, int dc, int rc) {
		if(str1 == null || str2 == null) {
			return 0;
		}
		char[] chas1 = str1.toCharArray();
		char[] chas2 = str2.toCharArray();
		int row = chas1.length + 1;
		int col = chas2.length + 1;
		int[][] dp = new int[row][col];
		for(int i = 1; i < row; i++) {
			dp[i][0] = dc * i;
		}
		for(int j = 1; j < col; j++) {
			dp[0][j] = ic * j;
		}
		for(int i = 1; i < row; i++) {
			for(int j = 1; j < col; j++) {
				if(chas1[i - 1] == chas2[j - 1]) {
					dp[i][j] = dp[i - 1][j - 1];
				} else {
					dp[i][j] = dp[i - 1][j - 1] + rc;
				}
				dp[i][j] = Math.min(dp[i][j], dp[i][j - 1] + ic);
				dp[i][j] = Math.min(dp[i][j], dp[i - 1][j] + dc);
			}
		}
		return dp[row - 1][col - 1];
	}
	
	public int minCost_2(String str1, String str2, int ic, int dc, int rc) {
		if(str1 == null || str2 == null) {
			return 0;
		}
		char[] chas1 = str1.toCharArray();
		char[] chas2 = str2.toCharArray();
		char[] longs = chas1.length >= chas2.length ? chas1 : chas2;
		char[] shorts = chas1.length < chas2.length ? chas1 : chas2;
		if(chas1.length < chas2.length) {
			// str2 较长就交换ic和dc的值
			int tmp = ic;
			ic = dc;
			dc = tmp;
		}
		int[] dp = new int[shorts.length + 1];
		for(int i = 1; i <= shorts.length; i++) {
			dp[i] = ic * i;
		}
		for(int i = 1; i <= longs.length; i++) {
			int pre = dp[0];  // pre 表示左上角的值
			dp[0] = dc * i;
			for(int j = 1; j <= shorts.length; j++) {
				int tmp = dp[j]; // dp[j] 没更新之前先保存下来
				if(longs[i - 1] == shorts[j - 1]) {
					dp[j] = pre;
				} else {
					dp[j] = pre + rc;
				}
				dp[j] = Math.min(dp[j], dp[j - 1] + ic);
				dp[j] = Math.min(dp[j], tmp + dc);
				pre = tmp;   // pre 变成dp[j] 没更新的值
			}
		}
		return dp[shorts.length];
	}

}
