def Dih(n):
    MASK = (1 << n) - 1
    rev = [-i % n for i in range(n)]

    def rotate(x, k):
        return ((x >> k) | (x << (n - k))) & MASK

    def reflect(x):
        out = 0
        for i in range(n):
            if (x >> i) & 1:
                out |= (1 << rev[i])
        return out

    G = []
    for i in range(n):
        G.append(lambda x, i=i: rotate(x, i))
    for i in range(n):
        G.append(lambda x, i=i: reflect(rotate(x, i)))
    return G

def canonicals(n, k):
    MASK = (1 << n) - 1
    G = Dih(n)

    def is_canonical(x):
        for g in G:
            if g(x) < x:
                return False
        return True

    if k <= 1 or k >= n:
        return

    if k > n // 2:
        k = n - k

    stack = [(1, 0, 1)]  # bitset, last index, count
    while stack:
        bitset, last_idx, count = stack.pop()

        if count == k:
            if is_canonical(bitset):
                yield bitset
            continue

        min_i = last_idx + 1
        max_i = n - (k - count)
        for i in range(min_i, n):
            if i > max_i:
                break
            new_bitset = bitset | (1 << i)
            stack.append((new_bitset, i, count + 1))

if __name__ == '__main__':
    # Example output
    out = list(canonicals(30, 15))
    out.sort()
    print(out)
    print(len(out))
