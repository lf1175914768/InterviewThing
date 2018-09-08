package com.study.interview.string;

/**
 * 将整数字符串转换成整数值
 * @author LiuFeng
 *
 */
public class ConvertStringToInteger {
	
	public int convert(String str) {
		if(str == null || str.equals("")) {
			return 0;  // 不能转
		}
		char[] chas = str.toCharArray();
		if(!isValid(chas)) {
			return 0;  // 不能转
		}
		boolean posi = chas[0] == '-' ? false : true;
		int minq = Integer.MIN_VALUE / 10;
		int minr = Integer.MIN_VALUE % 10;
		int cur = 0, res = 0;
		for(int i = posi ? 0 : 1; i < chas.length; i++) {
			cur = '0' - chas[i];
			if((res < minq) || (res == minq && cur < minr)) {
				return 0;   // 不能转
			}
			res = res * 10 + cur;
		}
		if(posi && res == Integer.MIN_VALUE) {
			return 0;   // 不能转。
		}
		return posi ? -res : res;
	}
	
	public boolean isValid(char[] chas) {
		if(chas[0] != '-' && (chas[0] < '0' || chas[0] > '9')) {
			return false;
		}
		if(chas[0] == '-' && (chas.length == 1 || chas[1] == '0')) {
			return false;
		}
		if(chas[0] == '0' && chas.length > 1) {
			return false;
		}
		for(int i = 1; i < chas.length; i++) {
			if(chas[i] < '0' || chas[i] > '9') {
				return false;
			}
		}
		return true;
	}
	
}
