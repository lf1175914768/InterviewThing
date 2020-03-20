package com.study.my;

import java.util.concurrent.atomic.AtomicReference;

/**
 * @Description: TODO
 * @Author: Liufeng
 * @Date: 2020/3/20 11:29
 */
public class AtomicReferenceTest {

    static AtomicReference<User> atomicUserRef = new AtomicReference<User>();

    public static void main(String[] args) {
        User user = new User("kenan", 15);
        atomicUserRef.set(user);
        User updateUser = new User("newKenan", 16);
        atomicUserRef.compareAndSet(user, updateUser);
        System.out.println(atomicUserRef.get().getName());
        System.out.println(atomicUserRef.get().getOld());
    }

    static class User {
        private String name;
        private int old ;
        public User(String name, int old) {
            this.name = name;
            this.old = old;
        }

        public String getName() {
            return name;
        }

        public int getOld() {
            return old;
        }
    }
}
