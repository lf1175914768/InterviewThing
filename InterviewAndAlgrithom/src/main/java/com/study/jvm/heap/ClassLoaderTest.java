package com.study.jvm.heap;

/**
 * Created by Liufeng on 2020/1/9.
 */
public class ClassLoaderTest {
    public static void main(String[] args) {
        Son son = new Son();
    }
}

class Parent {
    private static int b;
    private int c = initc();
    static {
        b = 1;
        System.out.println("init Parent Static Block, initialize b successfully!");
    }
    private static final int a = 1;
    static {
        System.out.println("init parent static block, initialize a successfully! And the value of A is " + a);
    }
    int initc(){
        System.out.println("3.父类成员变量赋值：---> c的值"+c);
        this.c=12;
        System.out.println("3.父类成员变量赋值：---> c的值"+c);
        return c;
    }
    public Parent(){
        System.out.println("4.父类构造方式开始执行---> a:"+a+",b:"+b);
        System.out.println("4.父类构造方式开始执行---> c:"+c);
    }
}

class Son extends Parent {
    private int sc = initc2();
    private static int sb;
    static {
        sb = 1;
        System.out.println("init the child class static block, initialize sb successfully!");
    }
    private static int sa = 1;
    static {
        System.out.println("init the child class static block, initialize sa successfully! And the value of Sa is " + sa);
    }
    int initc2(){
        System.out.println("5.子类成员变量赋值--->：sc的值"+sc);
        this.sc=12;
        return sc;
    }
    public Son(){
        System.out.println("6.子类构造方式开始执行---> sa:"+sa+",sb:"+sb);
        System.out.println("6.子类构造方式开始执行---> sc:"+sc);
    }
}
