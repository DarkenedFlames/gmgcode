import cmath, math, os, json
from functools import cached_property, total_ordering
from itertools import combinations
from typing import Dict, Tuple, Set, Iterable

def _load_mapping(name: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), f'{name}.json')
    with open(path) as f:
        return json.load(f)

_FORTES = _load_mapping('FORTES')

#----------------------------------------------------------------
# Helper vars
#----------------------------------------------------------------

_BITS = 12
_N    = 1 << _BITS
_MASK = _N - 1

#----------------------------------------------------------------
# 𝙎𝙘𝙖𝙡𝙚 𝘾𝙡𝙖𝙨𝙨
#----------------------------------------------------------------

@total_ordering
class PCSet:
    """A pitch-class set class with rich interval, spectral, and geometric analysis."""

    def __init__(self, data):
        """Initialize from either:
              data: 12‐bit binary tuple: (1,0,1,0,1,1,0,1,0,1,0,1)
              data: iterable of ints (pcs): (0,2,4,5,7,9,11)
              data: integer: 2741"""
          
        if isinstance(data, int):
            self._d = data & _MASK
        elif (isinstance(data, tuple)
              and len(data) == _BITS
              and set(data) <= {0,1}):
            self._d = sum(b << i for i, b in enumerate(data))
        else:
            pcs = sorted({int(p) % _BITS for p in data})
            self._d = sum(1 << p for p in pcs)

        self._b = tuple((self._d >> i) & 1 for i in range(_BITS))
        self._p = tuple(i for i, b in enumerate(self._b) if b)
        
    # ----------------------------------------------------------------
    # 𝙁𝙪𝙣𝙙𝙖𝙢𝙚𝙣𝙩𝙖𝙡 𝙍𝙚𝙥𝙧𝙚𝙨𝙚𝙣𝙩𝙖𝙩𝙞𝙤𝙣𝙨
    # ----------------------------------------------------------------

    @property
    def pcs(self) -> Tuple[int, ...]:
        """Returns the pitch-class set of the pitch-class set."""
        return self._p

    @property
    def binary(self) -> Tuple[int, ...]:
        """Returns the binary representation of the pitch-class set."""
        return self._b

    @property
    def decimal(self) -> int:
        """Returns the decimal representation of the pitch-class set."""
        return self._d

    # 𝘼𝙡𝙩𝙚𝙧𝙣𝙖𝙩𝙚 𝙄𝙙𝙚𝙣𝙩𝙞𝙛𝙞𝙚𝙧𝙨
    @cached_property
    def forte(self) -> str: 
        """Returns the Forte number of the pitch-class set."""
        for k1, v1 in _FORTES.items():
            for k2, pcs in v1.items():
                if PCSet(tuple(pcs)) in self.all_forms:
                    return f'{k1}-{k2}'
       
    # ----------------------------------------------------------------
    # 𝘾𝙤𝙧𝙚 𝙏𝙧𝙖𝙣𝙨𝙛𝙤𝙧𝙢𝙖𝙩𝙞𝙤𝙣𝙨
    # ----------------------------------------------------------------
    
    def reflect(self, k: float = 0) -> 'PCSet':
        """Returns the pitch-class set reflected over axis k."""
        return PCSet(sum(1 << int((2 * k - p) % _BITS) for p in self))

    def rotate(self, k: int) -> 'PCSet':
        """Returns the pitch-class set rotated CW by k semitones."""
        return PCSet(self._d >> (k % _BITS) | self._d << (-k % _BITS))
    
    
    @property
    def complement(self) -> 'PCSet':
        """Returns the complement pitch-class set."""
        return PCSet(~self._d)
        
    @property
    def prime(self) -> 'PCSet':
        """Returns the pitch-class set in prime form."""
        return min(self.all_forms)
        

    # Transformation Collections
    @cached_property
    def modes(self) -> Tuple['PCSet', ...]:
        """Returns the modes of the pitch-class set."""
        return tuple(self << p for p in self)

    @cached_property
    def all_forms(self) -> Tuple['PCSet', ...]:
        """Returns all rotational and reflective transformations of the pitch-class
          set."""
        return self.modes + (~self).modes

    # ----------------------------------------------------------------
    # Intervallic Metrics
    # ----------------------------------------------------------------

    # Count-based
    @property
    def cardinality(self) -> int:
        """Returns the number of pitch-classes in the pitch-class set."""
        return len(self)

    @property
    def palindromic(self) -> bool:
        """Returns whether the pitch-class set is palindromic."""
        return self == ~self

    @property
    def chiral(self) -> bool:
        """Returns whether the pitch-class set is chiral."""
        return self not in (~self).modes

    @property
    def deep(self) -> bool:
        """Returns whether the pitch-class set is deep."""
        return len(set(self.vector)) == 6

    @cached_property
    def myhill(self) -> bool:
        """Returns whether the pitch-class set has Myhill's' property."""
        return all(len(v) == 2 for v in self.spectrum.values())

    @cached_property
    def variation(self) -> float:
        """Returns the spectra variation of the pitch-class set."""
        return sum(max(r) - min(r) for r in self.spectrum.values()) / len(self)

    @cached_property
    def rotational_axes(self) -> Tuple[int, ...]:
        """Returns the rotational axes of the pitch-class set."""
        m = self.modes
        return tuple(self[i] for i in range(1, len(self)) if m[i] == m[0])

    @cached_property
    def reflective_axes(self) -> Tuple[float, ...]:
        """Returns the reflective axes of the pitch-class set."""
        return tuple(i / 2 for i in range(12) if self.reflect(i / 2) == self)

    @cached_property
    def ridges(self) -> Tuple[int, ...]:
        """Returns the ridge tones of the pitch-class set."""
        return tuple(int(2 * i) for i in self.reflective_axes)

    @property
    def propriety(self) -> str:
        """Returns the propriety of the pitch-class set."""
        a, c = self.ambiguities, self.contradictions
        return 'Improper' if a * c else 'Proper' if a else 'Strictly Proper'

    @property
    def brightness(self) -> int:
        """Returns the brightness of the pitch-class set."""
        return sum(self._p)

    @property
    def cohemitonia(self) -> int:
        """Returns the count of cohemitones in the pitch-class set."""
        return len(self._auto((1, 2)))

    @property
    def imperfections(self) -> int:
        """Returns the count of imperfections in the pitch-class set."""
        return len(self) - self.vector[4]

    def _count_failure(self, attr: str = 'contradiction') -> int:
        """Returns the total number of coherency failures of the given type
          (contradiction, ambiguity) in the pitch-class set."""
        M = self.matrix
        n = len(self)
    
        comp = (lambda x, y: x < y) if attr == 'contradiction' else (lambda x, y: x == y)
    
        return sum(
            1
            for i in range(1, n)
            for j in range(n)
            for p in range(i)
            for q in range(n)
            if comp(M[i][j], M[p][q])
        )

    @property
    def ambiguities(self) -> int:
        """Returns the count of ambiguities in the pitch-class set."""
        return self._count_failure('ambiguity')

    @property
    def contradictions(self) -> int:
        """Returns the count of contradictions in the pitch-class set."""
        return self._count_failure()

    @cached_property
    def differences(self) -> int:
        """Returns the count of differences in the pitch-class set."""
        return sum(
            1 for row in self.matrix
              for a, b in combinations(row, 2)
              if a != b
        )

    @cached_property
    def heteromorphic_profile(self) -> Tuple[int, int, int]:
        """Returns the tuple: (contradictions, ambiguities, differences)."""
        return (self.contradictions, self.ambiguities, self.differences)

    @cached_property
    def max_failures(self) -> int:
        """Returns the maximum number of coherency failures possible for the cardinality of the pitch-class set."""
        n = len(self)
        return n * (n - 1) * (n - 2) * (3 * n - 5) // 24

    @cached_property
    def max_differences(self) -> int:
        """Returns the maximum number of interval differences possible for the
          cardinality of the pitch-class set."""
        n = len(self)
        return n * (n - 1) * (n - 1) // 2

    @cached_property
    def coherence_quotient(self) -> float:
        """Returns the coherence quotient of the pitch-class set."""
        c, a, d = self.heteromorphic_profile
        m = self.max_failures
        return 1 - ((c + a) / m if m else 1)

    @cached_property
    def sameness_quotient(self) -> float:
        """Returns the sameness quotient of the pitch-class set."""
        c, a, d = self.heteromorphic_profile
        m = self.max_differences
        return 1 - (d / m if m else 1)



    # ----------------------------------------------------------------
    # Interval Tensors
    # ----------------------------------------------------------------
    def _cauto(self, i: int) -> 'PCSet':
        """Returns the autocorrelation pitch-class set of the pitch-class set and its 
        complement."""
        return self & (-self << i)
        
    def _auto(self, rotations: Tuple[int, ...]) -> 'PCSet':
        """Returns the autocorrelation pitch-class set; repeatedly autocorrelated at 
        provided offsets."""
        auto = PCSet(self._d)
        for r in rotations:
            auto &= (self << r)
        return auto

    @cached_property
    def vector(self) -> Tuple[int, int, int, int, int, int]:
        """Returns the interval vector of the pitch-class set."""
        v = [len(self._auto((i,))) for i in range(1, 7)]
        v[5] //= 2
        return tuple(v)

    def _extreme(self, minimal: bool) -> Tuple[int, int, int, int, int, int]:
        """Returns the vector of the minimum or maximum possible count of each interval
          class for the cardinality of the pitch-class set."""
        n = len(self)
        if minimal:
            if n < 7:
                v = [0] * 6
            else:
                v = [2 * n - 12] * 6
                v[5] //= 2
            if n in (5,6,7):
                v[3] += 1
        else:
            v = [n - (1 if (i * n) % _BITS else 0) for i in range(1, 6)] + [n // 2]
        return tuple(v)

    @cached_property
    def p_sat_vector(self) -> Tuple[float, float, float, float, float, float]:
        """Returns the proportional saturation vector of the pitch-class set."""
        M, m = self._extreme(False), self._extreme(True)
        return tuple((self.vector[i] - m[i]) / (M[i] - m[i]) if (M[i] - m[i]) != 0 else 1 for i in range(6))

    @cached_property
    def matrix(self) -> Tuple[Tuple[int, ...], ...]:
        """Returns the interval matrix of the pitch-class set."""
        return tuple(tuple(r) for r in list(zip(*(i.pcs for i in self.modes)))[1:])

    @cached_property
    def spectrum(self) -> Dict[int, Set[int]]:
        """Returns the interval spectra of the pitch-class set."""
        return {i: set(r) for i, r in enumerate(self.matrix, start=1)}

    @property
    def structure(self) -> Tuple[int, ...]:
        """Returns the interval structure of the pitch-class set."""
        return self.matrix[0]

    @cached_property
    def _gen_origin(self) -> Tuple[int, int]:
        """Returns generator/origin pair of the pitch-class set, if it has one."""
        for g in range(1, 7):
            for o in range(_BITS):
                if { (o + g * k) % _BITS for k in range(len(self)) } == set(self):
                    return g, o
        return None, None
    
    @property
    def generator(self) -> int:
        """Returns the generator of the pitch-class set, if it has one."""
        return self._gen_origin[0]

    @property
    def origin(self) -> int:
        """Returns the origin of the pitch-class set, if it has one."""
        return self._gen_origin[1]

    # ----------------------------------------------------------------
    # 𝙂𝙚𝙤𝙢𝙚𝙩𝙧𝙞𝙘 𝙋𝙧𝙤𝙥𝙚𝙧𝙩𝙞𝙚𝙨
    # ----------------------------------------------------------------

    @cached_property
    def _circle(self) -> Tuple[complex, ...]:
        """Returns the complex unit‐circle positions of the pitch-class set."""
        return tuple(cmath.exp(p * 2j * math.pi / _BITS) for p in self)

    @property
    def _consec(self) -> Iterable:
        """Returns consecutive pairs in self._circle."""
        c = self._circle
        return zip(c, c[1:] + c[:1])

    @cached_property
    def area(self) -> float:
        """Returns the internal area of the pitch-class set."""
        return 0.5 * abs(sum((p * q.conjugate()).imag for p, q in self._consec))

    @cached_property
    def perimeter(self) -> float:
        """Returns the perimeter of the pitch-class set."""
        return sum(abs(p - q) for p, q in self._consec)

    @cached_property
    def fourier(self) -> Tuple[float, float, float, float, float, float, float]:
        """Returns the Lewin-Quinn Fourier Coefficients of the pitch-class set."""
        return tuple(abs(sum(c ** -k for c in self._circle)) for k in range(7))
    
    @cached_property
    def _ideal_set(self) -> 'PCSet':
        """Returns the maximally even pitch-class set given the cardinality."""
        n = len(self)
        return PCSet(tuple(_BITS * i // n for i in range(n)))

    @property
    def has_max_area(self) -> bool:
        """Returns whether the internal area of the pitch-class set is maximal for its cardinality."""
        return abs(self.area - self._ideal_set.area) < 1e-9

    @property
    def has_max_evenness(self) -> bool:
        """Returns whether the pitch-class set is maximally even for its cardinality."""
        return self.prime == self._ideal_set.prime

    @property
    def centroid(self) -> complex:
        """Returns the complex center of the pitch-class set."""
        return -1j * sum(self._circle) / len(self)

    @property
    def centroid_distance(self) -> float:
        """Returns the distance of the complex centroid of the pitch-class set."""
        return abs(self.centroid)

    @property
    def balanced(self) -> bool:
        """Returns whether the complex centroid of the pitch-class set is centered at the origin."""
        return abs(self.centroid) < 1e-9

    @property
    def centroid_angle_degrees(self) -> float:
        """Returns the angle of the complex centroid clockwise from the root (0°)."""
        return None if self.balanced else (math.degrees(cmath.phase(self.centroid)) + 90) % 360

    @property
    def centroid_angle_cents(self) -> float:
        """Returns the angle of the complex centroid of the Sclae in cents."""
        a = self.centroid_angle_degrees
        return None if a is None else 10 * a / 3

    # ----------------------------------------------------------------
    # Dunder Methods for DRY Transformations
    # ----------------------------------------------------------------

    def __invert__(self):
        """Bitwise NOT → inverse."""
        return self.reflect()

    def __neg__(self):
        """Unary minus → complement."""
        return self.complement

    def __lshift__(self, k):
        """Left shift (<< k) → rotate CCW by k semitones."""
        return self.rotate(k)

    def __rshift__(self, k):
        """Right shift (>> k) → rotate CW by k semitones."""
        return self.rotate(-k)

    def __eq__(self, other):
        """Equality: same bitmask."""
        return isinstance(other, PCSet) and self._d == other._d

    def __lt__(self, other):
        """For total_ordering: compare bitmask."""
        return isinstance(other, PCSet) and self._d < other._d

    def __iand__(self, other):
        """In-place bitwise AND."""
        self._d &= other._d
        return self

    def __and__(self, other):
        """Bitwise AND → new pitch-class set."""
        return PCSet(self._d & other._d)

    def __or__(self, other):
        """Bitwise OR → new pitch-class set."""
        return PCSet(self._d | other._d)

    def __xor__(self, other):
        """Bitwise XOR → new pitch-class set."""
        return PCSet(self._d ^ other._d)

    def __len__(self):
        """Number of pcs (popcount)."""
        return self._d.bit_count()

    def __hash__(self):
        """Hash based on integer representation."""
        return hash(self._d)

    def __iter__(self):
        """Iterate over pcs."""
        return iter(self._p)

    def __getitem__(self, index):
        """Index into pcs."""
        return self._p[index]

    def __contains__(self, pc):
        """Membership test in pcs."""
        return pc in self._p

    def __repr__(self):
        """Debug representation."""
        return f'PCSet({self._p})'
        
    def __str__(self):
        """String Representation."""
        return f'PCSet({self._p})'
