package com.study.interview.string;

/**
 * 去掉字符串中连续出现 K 个0 的子串
 * @author LiuFeng
 *
 */
public class DeleteKthSubstring {
	
	public String removeKZeros(String str, int k) {
		if(str == null || k < 1) {
			return str;
		}
		char[] chas = str.toCharArray();
		int count = 0, start = -1;
		for(int i = 0; i < chas.length; i++) {
			if(chas[i] == '0') {
				start = start == -1 ? i : start;
				count++;
			} else {
				if(count == k) {
					while(count-- != 0) chas[start++] = 0;
				}
				count = 0;
				start = -1;
			}
		}
		if(count == k) {
			while(count-- != 0) chas[start++] = 0;
		}
		return String.valueOf(chas);
	}

}
