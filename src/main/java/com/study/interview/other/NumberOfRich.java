package com.study.interview.other;

import java.util.Scanner;

/**
 * @author feng.liu
 * @date 2020/5/7
 */
public class NumberOfRich {

    /**
     * 一个大于等于2的整数，如果可以分解为8个或8个以上的素数相乘，则称其为发财数，让你输出第n个发财数（n最大到1w）
     *
     * @param args args
     */
    public static void main(String[] args) {
        int[] set = new int[1000];
        for (int i = 2, j = 0; j < 1000; i++) {
            if (isOdd(i)) {
                set[j++] = i;
            }
        }
        Scanner sc = new Scanner(System.in);
        while(sc.hasNext()) {
            int count = sc.nextInt();
            int i = 253;
            for(int j = 0; i < Integer.MAX_VALUE && j < count; i++) {
                if(isFa(i + 1, set)) {
                    j++;
                }
            }
            if(i < Integer.MAX_VALUE) {
                System.out.println(i);
            } else {
                System.out.println("Infinity");
            }
        }
    }

    static boolean isFa(int n, int[] set) {
        int count = 0;
        for(int i = 0; i < set.length; i++) {
            if(count == 8) {
                return true;
            }
            if(n % set[i] == 0) {
                count++;
                n /= set[i];
                i--;
            }
        }
        return false;
    }

    static boolean isOdd(int n) {
        if(n < 2) {
            return false;
        }
        if(n == 2) {
            return true;
        }
        int i = 2;
        while(i * i < n) {
            i++;
        }
        for(int j = 2; j <= i; j++) {
            if(n % j == 0) {
                return false;
            }
        }
        return true;
    }
}
