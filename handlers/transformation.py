
class TransformationHandler:
    def __init__(self, bitmask, ET):
        self._ET         = ET
        self._N          = 1 << self._ET
        self._MASK       = self._N - 1
        self.mask        = bitmask
        self.cardinality = self.mask.bit_count()
                    
    @staticmethod    
    def _rotate(x, n, k):
        return ((x >> k) | (x << (n - k))) & ((1 << n) - 1)
    
    @staticmethod
    def _reflect(x, n, k):
        out = 0
        for i in range(n):
            if (x >> i) & 1:
                reflected = (2 * k - i) % n
                out |= (1 << int(reflected))
        return out
    
    def rotate(self, k=1):
        return self._rotate(self.mask, self._ET, k)
    
    def reflect(self, k=0):
        return self._reflect(self.mask, self._ET, k)
    
    @property
    def complement(self):
        """Returns the complement pitch-class set."""
        return self._MASK & (~self.mask)
    
    @property
    def Dih(self):
        """Return the dihedral group D_n = {T_k, I_k}."""
        group = []
        for i in range(self._ET):
            group.append(lambda x, i=i: self._rotate(x, self._ET, i))
        for i in range(self._ET):
            group.append(lambda x, i=i: self._reflect(x, self._ET, i))
        return group
    
    @property
    def prime(self):
        """Returns the pitch-class set in prime form."""
        return min(self.orbit)
        
    @property
    def is_palindromic(self):
        return self.mask == self.reflect()

    @property
    def is_chiral(self):
        return self.reflect() not in self.modes

    # Transformation Collections
    @property
    def orbit(self):
        return tuple(g(self.mask) for g in self.Dih)
    
    @property
    def rotations(self):
        return tuple(self.rotate(i) for i in range(self._ET))
    
    @property
    def modes(self):
        return tuple(i for i in self.rotations if i & 1)

    @property
    def rotational_axes(self):
        return tuple(i for i in range(self._ET) if self.rotations[i] == self.mask)

    @property
    def reflective_axes(self):
        return tuple(i / 2 for i in range(self._ET) if self.reflect(i / 2) == self.mask)

    @property
    def ridge_tones(self):
        return tuple(int(2 * i) for i in self.reflective_axes)

    @property
    def neighbors(self):
        seen = set()
        x = self.mask
        n = self._ET
        for i in range(n):
            if (x >> i) & 1:
                for d in (-1, 1):
                    j = (i + d) % n
                    if not (x >> j) & 1:
                        y = x ^ (1 << i) ^ (1 << j) 
                        seen.add(y)
        return seen


    @property
    def canonicals(self):
        """
        Returns canonical k-sets in n-ET.
        """
        n = self._ET
        k = self.cardinality
        MASK = (1 << n) - 1
    
        def is_canonical(x):
            G = TransformationHandler(x, n).Dih
            return all(g(x) >= x for g in G)
    
        if k <= 1 or k >= n - 1:
            return
    
        if k > n // 2:
            k = n - k
    
        stack = [(1, 0, 1)] 
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
        