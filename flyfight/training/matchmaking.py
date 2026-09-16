def choose(pool,count,rng,random=True):
    if count>len(pool): raise ValueError('Population smaller than arena agent count')
    indices=rng.choice(len(pool),count,replace=False) if random else range(count)
    return [pool[int(i)] for i in indices]
