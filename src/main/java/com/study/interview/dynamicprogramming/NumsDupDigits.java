package com.study.interview.dynamicprogramming;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>至少有一位重复的數字</p>
 *
 * <em>  Difficulty: Hard </em>
 *
 * 给定一个正整数N，返回小于等于N且具有至少一位重复数字的正整数。
 * 示例：
 * input: 20,
 * output: 1
 * explanation: 具有至少1位重复数字的正数（<= 20） 只有11.
 *
 * input: 100
 * output: 10
 * explanation: 具有至少 1 位重复数字的正数（<= 100）有 11，22，33，44，55，66，77，88，99 和 100
 *
 * input: 1000
 * output: 262
 *
 * @author liufeng
 * @version 2020/5/2
 **/
public class NumsDupDigits {

    /*
    本题的思路是将其转换成数位DP，利用排列组合来求解
    原题可以转换成数字N有多少个不重复的数字
    总体的思路是： 设剩下的位置为 i， 剩下的数字为 j，则不重复的数字是在每一位依次填入与前几位不同的数字，即选取剩下的
    j个数字填入剩下的i个位置，即有A(j, i)种可能，最后将其累加就是不重复数字的个数。
    实际遍历中，我们只需要剩下的位置 i 这个变量，设数字N的位数为 k， 则剩下的数字 j = 10 - (k - i)
    对于以上思路，我们还可以分为以下两种情况，第一种是高位带0，第二种是高位不带0
    我们知道数学中0这个数字比较特别，高位数为0即等于没有高位数，比如0096就是数字96，这个数字尽管两个0重复了，
    但是这两个0属于高位，所以0096这个数字不是重复数字，即第一种情况允许高位的0可以重复
    使用第一种情况求位数小于k的不重复数字的个数：因为最高位总是为0，因此一开始剩下的数字j总是为9个（1-9），
    然后剩下的低位可选的数字总共有A(10-1,i);
    使用第二种情况求位数为k的不重复数字的个数：一开始剩下的数字j受数字N每位上的数字影响，
    设N的当前位的数字为n，则j<=n，然后剩下的低位可选的数字总共有A(10-(k-i),i)
     */
    public int numDupDigitsAtMostN(int N) {
        return N - dp(N);
    }

    private int dp(int n) {
        List<Integer> digits = new ArrayList<>();
        while(n > 0) {
            digits.add(n % 10);
            n /= 10;
        }
        int k = digits.size(), total = 0;
        int[] used = new int[10];

        for(int i = 1; i < k; i++) {
            total += 9 * A(9, i - 1);
        }

        for(int i = k - 1; i >= 0; i--) {
            int num = digits.get(i);
            for(int j = i == k - 1 ? 1 : 0; j < num; j++) {
                if(used[j] != 0) {
                    continue;
                }
                total += A(10 - (k - i), i);
            }
            if(++used[num] > 1)
                break;
            if(i == 0) {
                total += 1;
            }
        }
        return total;
    }

    private int A(int m, int n) {
        return fact(m) / fact(m - n);
    }

    private int fact(int n) {
        if(n == 1 || n == 0) {
            return 1;
        }
        int result = 1;
        for(int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
}
