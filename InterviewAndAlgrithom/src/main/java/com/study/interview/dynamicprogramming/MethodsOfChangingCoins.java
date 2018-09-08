package com.study.interview.dynamicprogramming;

/**
 * 换钱的方法数
 * @author LiuFeng
 *
 */
public class MethodsOfChangingCoins {
	
	//-----------------------------------------
	// 暴力递归， 分别用0张、1张、2张、.... arr[index] 剩下的组成剩余金额
	//-----------------------------------------
	
	public int coins(int[] arr, int aim) {
		if(arr == null || arr.length == 0 || aim < 0) {
			return 0;
		}
		return process1(arr, 0, aim);
	}
	
	public int process1(int[] arr, int index, int aim) {
		int res = 0;
		if(index == arr.length) {
			res = aim == 0 ? 1 : 0;
		} else {
			for(int i = 0; i * arr[index] <= aim; i++) {
				res += process1(arr, index + 1, aim - arr[index] * i);
			}
		}
		return res;
	}
	
	//-----------------------------------------------
	// 基于前面的基础上， 加上记忆搜索的方法
	//-----------------------------------------------
	
	// map[i][j] == 0 表示递归过程 p[i][j]没有计算过, 
	// map[i][j] == -1 表示递归过程 p[i][j]已经计算过,
	public int coins_2(int [] arr, int aim) {
		if(arr == null || arr.length == 0 || aim < 0) {
			return 0;
		}
		int[][] map = new int[arr.length + 1][aim + 1];
		return process2(arr, 0, aim, map);
	}
	
	public int process2(int[] arr, int index, int aim, int[][] map) {
		int res = 0;
		if(index == arr.length) {
			res = aim == 0 ? 1 : 0;
		} else{
			int mapValue = 0;
			for(int i = 0; i * arr[index] <= aim; i++) {
				mapValue = map[index + 1][aim - arr[index] * i];
				if(mapValue != 0) {  //说明已经计算过
					res += mapValue == -1 ? 0 : mapValue;
				} else {
					res += process2(arr, index + 1, aim - arr[index] * i, map);
				}
			}
		}
		map[index][aim] = res == 0 ? -1 : res;
		return res;
	}

}
