package com.study.interview.string;

import java.util.HashMap;
import java.util.Map;

/**
 * @author liufeng
 * @version 2020/5/2
 **/
public class LengthOfLongestSubstring {

    public int lengthOfLongestSubstring(String s) {
        char[] chas = s.toCharArray();
        int result = 0, start = 0, temp = 0;
        Map<Character, Integer> map = new HashMap<>();
        for(int i = 0; i < chas.length; i++) {
            if(!map.containsKey(chas[i])) {
                map.put(chas[i], i);
                temp += 1;
            } else {
                result = Math.max(result, temp);
                int end = map.get(chas[i]), count = 0;
                while(start <= end) {
                    map.remove(chas[start]);
                    count++;
                    start++;
                }
                map.put(chas[i], i);
                temp -= count - 1;
            }
        }
        return Math.max(result, temp);
    }

    /**
     * 采用滑动窗口来解决这个问题。
     */
    public int lengthOfLongestSubstring1(String s) {
        if(s.length() == 0)
            return 0;
        Map<Character, Integer> map = new HashMap<>();
        int max = 0, left = 0;
        for(int i = 0; i < s.length(); i++) {
            if(map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }

    public static void main(String[] args) {
        String s = "bbbbbb";
        int result = new LengthOfLongestSubstring().lengthOfLongestSubstring(s);
        System.out.println(result);
    }
}
