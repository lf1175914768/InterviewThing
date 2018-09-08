package com.study.interview.dynamicprogramming;

/**
 * 最长公共子串问题
 * @author LiuFeng
 *
 */
public class MaxCommonSubString {
	
	//-----------------------------------
	// 经典的动态规划方法
	//-----------------------------------
	
	public int[][] getdp(char[] str1, char[] str2) {
		int[][] dp = new int[str1.length][str2.length];
		for(int i = 0; i < str1.length; i++) {
			if(str1[i] == str2[0]) {
				dp[i][0] = 1;
			}
		}
		for(int j = 1; j < str2.length; j++) {
			if(str1[0] == str1[j]) {
				dp[0][j] = 1;
			}
		}
		for(int i = 1; i < str1.length; i++) {
			for(int j = 1; j < str2.length; j++) {
				if(str1[i] == str2[j]) {
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}
			}
		}
		return dp;
	}
	
	public String lcst1(String str1, String str2) {
		if(str1 == null || str2 == null || str1.equals("") || str2.equals("")) {
			return "";
		}
		char[] chas1 = str1.toCharArray();
		char[] chas2 = str2.toCharArray();
		int[][] dp = getdp(chas1, chas2);
		int max = 0;
		int end = 0;
		for(int i = 0; i < chas1.length; i++) {
			for(int j = 0; j < chas2.length; j++) {
				if(dp[i][j] > max) {
					max = dp[i][j];
					end = i;
				}
			}
		}
		return str1.substring(end - max + 1, end + 1);
	}
	
	//--------------------------------------------------
	//  优化之后的动态规划
	//--------------------------------------------------
	
	public String lcst2(String str1, String str2) {
		if(str1 == null || str2 == null || str1.equals("") || str2.equals("")) {
			return "";
		}
		char[] chas1 = str1.toCharArray();
		char[] chas2 = str2.toCharArray();
		int row = 0;    // 斜线开始位置的行
		int col = chas2.length - 1;   // 斜线开始位置的列
		int max = 0;    // 记录最大的长度
		int end = 0;    // 最大长度更新时，记录子串的结尾位置。
		while(row < chas1.length) {
			int i = row;
			int j = col;
			int len = 0;  
			// 从 (i, j) 开始向右下方遍历
			while(i < chas1.length && j < chas2.length) {
				if(chas1[i] != chas2[j]) {
					len = 0;
				} else {
					len++;
				}
				// 记录最大值， 一级结束字符的位置。
				if(len > max) {
					end = i;
					max = len;
				}
				i++; 
				j++;	
			}
			if(col > 0) {   // 斜线开始的位置列先向左移动
				col--;
			} else {   // 列移动到最左之后， 行向下移动
				row++;
			}
		}
		return str1.substring(end - max + 1, end + 1);
	}
	
}
