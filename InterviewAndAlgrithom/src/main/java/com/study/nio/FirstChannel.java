package com.study.nio;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

public class FirstChannel {
	
	public void readAndWrite() {
		try {
			RandomAccessFile aFile = new RandomAccessFile("F:\\test.txt", "rw");
			FileChannel inChannel = aFile.getChannel();
			
			ByteBuffer buf = ByteBuffer.allocate(48);
			
			int bytesRead = inChannel.read(buf);
			while(bytesRead != -1) {
				System.out.println("Read " + bytesRead);
				buf.flip();  // 首先读取数据到buffer， 反转buffer，从buffer中读取数据。
				 
				while(buf.hasRemaining()) {
					System.out.println((char) buf.get());
				}
				buf.clear();
				bytesRead = inChannel.read(buf);
			}
			aFile.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void transferChannelToAnother() {
		try {
			RandomAccessFile fromFile = new RandomAccessFile("F:\\test.txt", "rw");
			FileChannel fromChannel = fromFile.getChannel();
			
			RandomAccessFile toFile = new RandomAccessFile("F:\\toFile.txt", "rw");
			FileChannel toChannel = toFile.getChannel();
			
			long position = 0 ;
			long count = fromChannel.size();
			fromChannel.transferTo(position, count, toChannel);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
