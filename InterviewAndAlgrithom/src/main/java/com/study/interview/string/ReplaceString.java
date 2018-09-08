package com.study.interview.string;

/**
 * 替换字符串中连续出现的指定字符串
 * @author LiuFeng
 *
 */
public class ReplaceString {
	
	public String replace_1(String str, String from, String to) {
		if(str == null || from == null || 
				str.equals("") || from.equals("")) {
			return str;
		}
		StringBuilder sb = new StringBuilder();
		int index = 0, pos = 0;
		int fromLen = from.length();
		while((index = str.indexOf(from, pos)) != -1) {
			if(index != 0 && index == pos) {
				pos += fromLen;
				continue;
			}
			sb.append(str.substring(pos, index));
			pos = index + fromLen;
			sb.append(to);
		}
		sb.append(str.substring(pos));
		return sb.toString();
	}
	
	public String replace_2(String str, String from, String to) {
		if(str == null || from == null || str.equals("") || from.equals("")) {
			return str;
		}
		char[] chas = str.toCharArray();
		char[] chaf = from.toCharArray();
		int match = 0;
		for(int i = 0; i < chas.length; i++) {
			if(chas[i] == chaf[match++]) {
				if(match == chaf.length) {
					clear(chas, i, chaf.length);
					match = 0;
				} 
			} else {
				match = 0;
			}
		}
		StringBuilder res = new StringBuilder();
		StringBuilder cur = new StringBuilder();
		for(int i = 0; i < chas.length; i++) {
			if(chas[i] != 0) {
				cur.append(String.valueOf(chas[i]));
			}
			if(chas[i] == 0 && (i == 0 || chas[i - 1] != 0)) {
				res.append(cur).append(to);
				cur.delete(0, cur.length());
			}
		}
		res.append(cur);
		return res.toString();
	}

	private void clear(char[] chas, int end, int length) {
		while(length-- != 0) {
			chas[end--] = 0;
		}
	}

}
