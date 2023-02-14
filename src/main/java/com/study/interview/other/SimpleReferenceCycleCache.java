package com.study.interview.other;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

/**
 * <p>description: 简单的循环引用缓存 解决方式  </p>
 * <p>className:  SimpleReferenceCycleCache </p>
 * <p>create time:  2022/12/13 16:15 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class SimpleReferenceCycleCache {

    private static final Map<String, Object> singletonObjects = new HashMap<>();

    private static final Map<String, Object> earlySingletonObjects = new HashMap<>();

    private static final Map<String, ObjectFactory<?>> singletonFactories = new HashMap<>();

    public interface ObjectFactory<T> {
        T getObject();
    }

    /**
     * 通过二级缓存的方式进行实现
     */
    @SuppressWarnings("unchecked")
    public static <T> T getBeanByTwiceCache(Class<T> beanClass) {
        String beanName = beanClass.getSimpleName();
        if (singletonObjects.containsKey(beanName)) {
            return (T) singletonObjects.get(beanName);
        } else if (earlySingletonObjects.containsKey(beanName)) {
            return (T) earlySingletonObjects.get(beanName);
        }
        T object = null;
        try {
            object = beanClass.getConstructor().newInstance();
            earlySingletonObjects.put(beanName, object);
            for (Field field : beanClass.getDeclaredFields()) {
                field.setAccessible(true);
                Class<?> type = field.getType();
                field.set(object, getBeanByTwiceCache(type));
            }
        } catch (Exception ignore) {}

        // 当所有的属性都填充完成后，放置到 singletonObjects 中
        singletonObjects.put(beanName, object);
        earlySingletonObjects.remove(beanName);
        return object;
    }

    /**
     * 通过三级缓存的方式进行实现
     */
    @SuppressWarnings("unchecked")
    public static <T> T getBeanByThirdCache(Class<T> beanClass) {
        String beanName = beanClass.getSimpleName();
        if (singletonObjects.containsKey(beanName)) {
            return (T) singletonObjects.get(beanName);
        } else if (earlySingletonObjects.containsKey(beanName)) {
            return (T) earlySingletonObjects.get(beanName);
        }
        ObjectFactory<?> factory = singletonFactories.get(beanName);
        if (factory != null) {
            return (T) factory.getObject();
        }
        T object = null;
        try {
            object = beanClass.getConstructor().newInstance();
            final Object obj = object;
            singletonFactories.put(beanName, () -> {
                Object proxy = createProxy(obj);
                singletonFactories.remove(beanName);
                earlySingletonObjects.put(beanName, proxy);
                return proxy;
            });
            for (Field field : beanClass.getDeclaredFields()) {
                field.setAccessible(true);
                Class<?> type = field.getType();
                field.set(object, getBeanByThirdCache(type));
            }
        } catch (Exception ignore) {}
        object = createProxy(object);
        singletonObjects.put(beanName, object);
        singletonFactories.remove(beanName);
        earlySingletonObjects.remove(beanName);
        return object;
    }

    private static <T> T createProxy(T object) {
        return object;
    }
}
