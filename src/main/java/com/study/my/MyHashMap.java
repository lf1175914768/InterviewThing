package com.study.my;

public class MyHashMap<K, V> implements MyMap<K, V> {
	
	// 默认的数组初始化长度
	private static final int DEFAULT_INITIAL_CAPACITY = 1 << 4;
	
	private static final int MAXNUM = 1 << 30;
	
	// 默认的阈值比例
	private static final float DEFAULT_LOADER_FACTOR = 0.75f;
	
	private Entry<K, V>[] table = null;
	
	private int defaultInitialSize;
	
	private float defaultLoaderFactor;
	
	private int entryUseSize;
	
	public MyHashMap() {
		this(DEFAULT_INITIAL_CAPACITY, DEFAULT_LOADER_FACTOR);
	}
	
	@SuppressWarnings("unchecked")
	public MyHashMap(int defaultInitialCapacity, float defaultLoaderFactor) {
		if(defaultInitialCapacity < 0) {
			throw new IllegalArgumentException("Illegal initial capacity is : " +
							defaultInitialCapacity);
		} 
		if(defaultInitialCapacity > MAXNUM) {
			defaultInitialCapacity = MAXNUM;
		}
		if(defaultLoaderFactor < 0) {
			throw new IllegalArgumentException("Illegal loader factor: " + defaultLoaderFactor);
		}
		this.defaultInitialSize = defaultInitialCapacity;
		this.defaultLoaderFactor = defaultLoaderFactor;
		this.table = new Entry[defaultInitialCapacity];
	}

	@Override
	public V put(K k, V v) {
		if(k == null) {
			return putForNullKey(v);
		}
		V oldValue = null;
		// 是否需要扩容？ 
		if(entryUseSize >= defaultInitialSize * defaultLoaderFactor) {
			// 如果需要，那么重新排列
			resize(2 * defaultInitialSize);
		}
		int index = indexFor(hash(k), defaultInitialSize - 1);
		if(table[index] == null) {
			table[index] = new Entry<K, V>(k, v, null);
		} else {
			Entry<K, V> entry = table[index];
			Entry<K, V> key = entry;
			while(key != null) {
				if(k == key.getKey() || k.equals(key.getKey())) {
					// if found in the list, just update
					oldValue = key.getValue();
					key.value = v;
					return oldValue;
				}
				key = key.next;
			}
			// If not found 
			table[index] = new Entry<K, V>(k, v, entry);
		}
		++entryUseSize;
		return oldValue;
	}

	private int indexFor(int hash, int i) {
		return hash & i;
	}

	/**
	 * 将空值放在第一个位置上面的链表中
	 * @param v
	 * @return
	 */
	private V putForNullKey(V v) {
		for(Entry<K, V> entry = table[0]; entry != null; entry = entry.next) {
			if(entry.key == null) {
				V oldValue = entry.value;
				entry.value = v;
				return oldValue;
			}
		}
		// if not found , just add the value to the first.
		Entry<K, V> e = table[0];
		table[0] = new Entry<K, V>(null, v, e);
		if(entryUseSize++ >= defaultInitialSize) {
			resize(defaultInitialSize * 2);
		}
		return null;
	}

	/**
	 * 参考JDK 里面的 HashMap里面的hash函数， 
	 * @param k
	 * @return
	 */
	private int hash(Object k) {
		int hashCode = k.hashCode();
		hashCode ^= (hashCode >>> 20) ^ (hashCode >>> 12);
		return hashCode ^ (hashCode >>> 7) ^ (hashCode >>> 4);
	}

	@SuppressWarnings("unchecked")
	protected void resize(int i) {
		Entry<K, V>[] newTable = new Entry[i];
		this.defaultInitialSize = i;
		rehash(newTable);
	}
	
	private void rehash(Entry<K, V>[] newTable) {
		for(Entry<K, V> entry : table) {
			if(entry != null) {
				do {
					Entry<K, V> next = entry.next;
					int i = indexFor(hash(entry.key), defaultInitialSize - 1);
					entry.next = newTable[i];
					newTable[i] = entry; 
					entry = next;
				} while(entry != null);
			}
		}
		// 覆盖旧的引用
		if(newTable.length > 0) {
			table = newTable;
		}
	}

	@Override
	public V get(K k) {
		if(k == null) {
			return getForNullKey();
		}
		int index = indexFor(hash(k), defaultInitialSize - 1);
		if(table[index] == null) {
			return null;
		} else {
			Entry<K, V> entry = table[index];
			do {
				if(k == entry.getKey() || k.equals(entry.getKey())) {
					return entry.value;
				}
				entry = entry.next;
			} while(entry != null);
		}
		return null; 
	} 
	
	private V getForNullKey() {
		for(Entry<K, V> e = table[0]; e != null; e = e.next) {
			if(e.key == null) return e.value;
		}
		return null;
	}

	class Entry<K, V> implements MyMap.Entry<K, V> {
		
		final K key;
		V value;
		Entry<K, V> next;
		
		Entry(K k, V v, Entry<K, V> next) {
			this.key = k;
			this.value = v;
			this.next = next;
		}

		@Override
		public final K getKey() {
			return this.key;
		}

		@Override
		public final V getValue() {
			return this.value;
		}
		
	}

}
