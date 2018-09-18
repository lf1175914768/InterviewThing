package com.study.interview.array;

/**
 * Given a 2D board and a word, find if the word exists in the grid.
 * The word can be constructed from letters of sequentially adjacent cell, 
 * where "adjacent" cells are those horizontally or vertically neighboring. 
 * The same letter cell may not be used more than once.
 * 
 * @author Liufeng
 * @createData Created on: 2018年9月17日 下午3:45:23
 */
public class WordSearch {
	
	public boolean exist(char[][] board, String word) {
		if(board.length == 0 || board[0].length == 0) return false;
		int m = board.length, n = board[0].length;
		boolean [][] visited = new boolean[m][n];
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				if(search(board, word, 0, i, j, visited)) return true;
			}
		}
		return false;
	}

	private boolean search(char[][] board, String word, int idx, int i, int j, boolean[][] visited) {
		if(idx == word.length()) return true;
		int m = board.length, n = board[0].length;
		if(i < 0 || j < 0 || i >= m || j >= n ||
				visited[i][j] || board[i][j] != word.charAt(idx)) return false;
		visited[i][j] = true;   // 表示该位置已经被访问过了。
		boolean res = search(board, word, idx + 1, i - 1, j, visited) || 
				search(board, word, idx + 1, i + 1, j, visited) ||
				search(board, word, idx + 1, i, j - 1, visited) ||
				search(board, word, idx + 1, i, j + 1, visited);
		visited[i][j] = false;
		return res;
	}

}
