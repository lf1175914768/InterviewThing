package com.study.interview.dynamicprogramming;

/**
 * 字符串的交错组成， 判断aim是否能被str1和str2 交错组成
 * @author LiuFeng
 *
 */
public class IntersectionOfString {

	public boolean isCross_1(String str1, String str2, String aim) {
		if(str1 == null || str2 == null || aim == null) {
			return false;
		}
		char[] chas1 = str1.toCharArray();
		char[] chas2 = str2.toCharArray();
		char[] chain = aim.toCharArray();
		if(chain.length != chas1.length + chas2.length) {
			return false;
		}
		boolean[][] dp = new boolean[chas1.length + 1][chas2.length + 1];
		dp[0][0] = true;
		for(int i = 1; i <= chas1.length; i++) {
			if(chas1[i - 1] != chain[i - 1]) {
				break;
			}
			dp[i][0] = true;
		}
		for(int j = 1; j <= chas2.length; j++) {
			if(chas2[j - 1] != chain[j - 1]) {
				break;
			}
			dp[0][j] = true;
		}
		for(int i = 1; i <= chas1.length; i++) {
			for(int j = 1; j <= chas2.length; j++) {
				if((chas1[i - 1] == chain[i + j - 1] && dp[i - 1][j]) ||
						(chas2[j - 1] == chain[i + j - 1] && dp[i][j - 1])) {
					dp[i][j] = true;
				}
			}
		}
		return dp[chas1.length][chas2.length];
	}
	
	public boolean isCross_2(String str1, String str2, String aim) {
		if(str1 == null || str2 == null || aim == null) {
			return false;
		}
		char[] chas1 = str1.toCharArray();
		char[] chas2 = str2.toCharArray();
		char[] chain = aim.toCharArray();
		if(chain.length != chas1.length + chas2.length) {
			return false;
		}
		char[] longs = chas1.length >= chas2.length ? chas1 : chas2;
		char[] shorts = chas1.length < chas2.length ? chas1 : chas2;
		boolean[] dp = new boolean[shorts.length + 1];
		dp[0] = true;
		for(int i = 1; i <= shorts.length; i++) {
			if(shorts[i - 1] != chain[i - 1]) {
				break;
			}
			dp[i] = true;
		}
		for(int i = 1; i <= longs.length; i++) {
			dp[0] = dp[0] && longs[i - 1] == chain[i - 1];
			for(int j = 1; j <= shorts.length; j++) {
				if((longs[i - 1] == chain[i + j - 1] && dp[j]) ||
						(shorts[i - 1] == chain[i + j - 1] && dp[j - 1])) {
					dp[j] = true;
				} else {
					dp[j] = false;
				}
			}
		}
		return dp[shorts.length];
	}
	
}
