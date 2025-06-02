
from forte_numbers import FORTES
import cmath
import math
import string
from collections import defaultdict, Counter
from functools import lru_cache, cached_property
from itertools import combinations
import numpy as np
from typing import List, Dict, Set, Tuple, Optional

#----------------------------------------------------------------
# Helper vars and functions
#----------------------------------------------------------------

_UNIT_CIRCLE: Tuple[complex] = tuple(cmath.exp(2j * math.pi * k / 12) * (-1j) for k in range(12))
_DIGITS: str = string.digits + string.ascii_uppercase + string.ascii_lowercase + "+/"

def rotate_pcs(pcs: Tuple[int], k: int) -> Tuple[int]:
    """Returns a pitch-class set rotated by k semitones (mod 12)."""
    return tuple(sorted(((p - k) % 12) for p in pcs))

def pcs_to_binary(pcs: Tuple[int]) -> str:
    """Convert a pitch-class set tuple into a 12‐bit binary string."""
    bits = ['0'] * 12
    for pc in pcs:
        bits[11 - pc] = '1'
    return ''.join(bits)

def binary_to_pcs(binary: str) -> Tuple[int]:
    """Convert a 12‐bit binary string into a pitch-class set tuple."""
    return tuple(i for i, b in enumerate(binary[::-1]) if b == '1')

def maptup(matrix) -> Tuple[Tuple[int]]:
    return tuple(map(tuple, matrix))

def tupsortset(array: iter) -> Tuple:
    return tuple(sorted(set(array)))

#----------------------------------------------------------------
# Scale Class
#----------------------------------------------------------------

class Scale:
    """A pitch‐class set (scale) with rich interval, spectral, and geometric analysis."""

    def __init__(self, data: str | Tuple[int]):
        """Initialize from either:
          - data: 12‐bit binary string, e.g. '101010110101'
          - data: iterable of ints (pcs), e.g. [0,2,4,5,…]"""
          
        if isinstance(data, str):
            pcs = binary_to_pcs(data)
        else:
            pcs = tupsortset(data)
        self._pcs = pcs
        
        if len(self._pcs) < 2:
            raise ValueError(f"Scales must have more than one note in them. Problem Scale: {self._pcs} of length {len(self._pcs)}")

    # ----------------------------------------------------------------
    # Fundamental Properties
    # ----------------------------------------------------------------

    @cached_property
    def pcs(self) -> Tuple[int]:
        """Returns the pitch-class set as a tuple."""
        return self._pcs

    @cached_property
    def binary(self) -> str:
        """Returns the 12‐bit string representation of the pitch-class set."""
        return pcs_to_binary(self._pcs)

    @cached_property
    def _n(self) -> int:
        """Integer bitmask for fast comparisons & hashing."""
        return int(self.binary, 2)

    @cached_property
    def cardinality(self) -> int:
        """Returns the number of pitches in the pitch-class set."""
        return len(self._pcs)
        
    @cached_property
    def forte(self) -> str:
        """Returns the Forte number of the pitch-class set."""
        for card, names in FORTES.items():
            for name, pcs in names.items():
                if pcs in self.modes + self.inverse_modes:
                    return f'{card}-{name}'
                    

    # ----------------------------------------------------------------
    # Interval‐vector and related
    # ----------------------------------------------------------------
    
    @cached_property
    def vector(self) -> Tuple[int, int, int, int, int, int]:
        """Returns the interval vector of the pitch-class set."""
        counts = np.bincount(np.array(self.matrix).ravel(), minlength=7)
        counts[6] //= 2
        return tuple(counts[1:7])

    def interval_class_count(self, ic: int) -> int:
        """Returns the vector count for interval‐class `ic` (1...6)."""
        return self.vector[ic - 1]

    @cached_property
    def alpha_vector(self) -> Dict[str, int]:
        """Returns Hanson's interval vector of the pitch-class set.'"""
        labels = ['p','m','n','s','d','t']
        order  = [4,3,2,1,0,5]
        return { labels[i]: self.vector[j] for i, j in enumerate(order) }

    def _extreme(self, minimal: bool) -> tuple[int, ...]:
        """Returns the vector of the minimum or maximum possible count of each interval class for the cardinality of the pitch-class set."""
        n = self.cardinality
        if minimal:
            if n < 7:
                vec = [0]*6
            else:
                vec = [2*(n-6)]*6
                vec[5] //= 2
            if n in (5,6,7):
                vec[3] += 1
        else:
            vec = [n - (1 if (i*n)%12 else 0) for i in range(1,6)] + [n//2]
        return tuple(vec)

    @cached_property
    def proportional_saturation_vector(self) -> Tuple[float]:
        """Returns the proportional saturation vector of the pitch-class set."""
        n = self.cardinality
        M = self._extreme(False)
        m = self._extreme(True)
        v = self.vector
        return tuple((v[i] - m[i]) / (M[i] - m[i]) if (M[i] - m[i]) != 0 else 1 for i in range(6))


    # ----------------------------------------------------------------
    # Cached spectral/mode computations
    # ----------------------------------------------------------------
    
    @cached_property
    def inverse(self) -> Tuple[int]:
        """Returns the inversion of the pitch-class set."""
        return self.inverse_modes[0]
    
    @cached_property
    def complement(self) -> Tuple[int]:
        """Returns the complement of the pitch-class set in prime form."""
        mask = (~self._n) & 0xFFF
        best = min(((mask >> i) | ((mask << (12 - i)) & 0xFFF)) 
            for i in range(12))
        return Scale(format(best, '012b')).prime
    
    @cached_property
    def modes(self) -> Tuple[Tuple[int]]:
        """Returns the modes of the pitch-class set as a tuple of tuples."""
        return maptup(np.sort(self.t_matrix, axis=1))
    
    @cached_property
    def inverse_modes(self) -> Tuple[Tuple[int]]:
        """Returns the modes of the inverse of the pitch-class set."""
        return maptup(np.sort(self.t_matrix, axis=0).T)

    @cached_property
    def matrix(self) -> Tuple[Tuple[int]]:
        """Spectral matrix: matrix[n] is the nth interval‐class spectrum."""
        return maptup((np.array(self.modes).T)[1:])

    @cached_property
    def k_matrix(self) -> Tuple[Tuple[int]]:
        """Returns the K‑matrix of the pitch-class set."""
        p = np.array(self._pcs)
        K = (p[None, :] + p[:, None]) % 12
        return maptup(K)

    @cached_property
    def t_matrix(self) -> Tuple[Tuple[int]]:
        """Returns the transformation matrix of the pitch-class set."""
        p = np.array(self._pcs)
        T = (p[None, :] - p[:, None]) % 12
        return maptup(T)

    @cached_property
    def s_k_matrix(self) -> Tuple[Tuple[int]]:
        """Return the shifted K matrix of the pitch-class set. """
        K = np.array(self.k_matrix)
        n = self.cardinality
        S_K = np.zeros((n,n), dtype=int)

        for i in range(n):
            for j in range(n):
                S_K[i,j] = K[(i-j)%n,j]
        return maptup(S_K)

    @cached_property
    def spectrum(self) -> Dict[int, Tuple[int]]:
        """Returns the interval spectrum of the pitch-class set."""
        return {size + 1: tupsortset(row) for size, row in enumerate(self.matrix)}

    @cached_property
    def neighbors(self) -> Tuple[Tuple[int]]:
        """Returns the parsimonious neighbors of the pitch-class set."""
        n = self.cardinality
        p = np.array(self._pcs).reshape(1, -1)
        
        S = np.eye(n, dtype=int)
        S[0] = (S[0] + 1) % 2
        
        ones_col = np.ones((n, 1), dtype=int)
        U = (ones_col @ p + S) % 12
        L = (ones_col @ p - S) % 12
    
        return tuple(tupsortset(row) for row in np.vstack((U, L)))

    @cached_property
    def same_size_neighbors(self) -> Tuple[Tuple[int]]:
        """Return each neighbor of a pitch-class set if and only if it is of equal cardinality to the original pitch-class set."""
        return tuple(i for i in self.neighbors if len(i) == self.cardinality)

    # ----------------------------------------------------------------
    # Structural properties
    # ----------------------------------------------------------------
    
    @cached_property
    def structure(self) -> Tuple[int]:
        """Returns the interval structure of the pitch-class set."""
        return self.matrix[0]

    @cached_property
    def variation(self) -> float:
        """Returns the spectrum variation of the pitch-class set."""
        return sum(max(row) - min(row) for row in self.spectrum.values()) / self.cardinality

    @cached_property
    def rot_axes(self) -> Tuple[int]:
        """Returns the rotational axes of the pitch-class set."""
        m = self.modes
        n = self.cardinality
        return tuple(self._pcs[i] for i in range(1, n) if m[i] == m[0])

    @cached_property
    def ref_axes(self) -> Tuple[float]:
        """Returns the reflective axes of the pitch-class set."""
        return tuple(self.ridges[i]/2 for i in range(len(self.ridges)))

    @cached_property
    def ridges(self) -> Tuple[int]:
        """Returns the ridge tones of the pitch-class set."""
        return tuple(row[0] for row in self.s_k_matrix if all(value == row[0] for value in row))

    # ----------------------------------------------------------------
    # Category tests
    # ----------------------------------------------------------------
    
    @cached_property
    def deep(self) -> bool:
        """Returns whether the pitch-class set is deep."""
        v = self.vector
        return len(set(v)) == len(v)

    @cached_property
    def myhill(self) -> bool:
        """Returns whether the pitch-class set has the Myhill property."""
        return all(len(vals) == 2 for vals in self.spectrum.values())

    @cached_property
    def palindromic(self) -> bool:
        """Returns whether the pitch-class set is palindromic."""
        return self._pcs == self.inverse

    @cached_property
    def chiral(self) -> bool:
        """Returns whether the pitch-class set is chiral."""
        return self._pcs not in self.inverse_modes

    @cached_property
    def prime(self) -> Tuple[int]:
        """Returns the prime form of the pitch-class set."""
        all_forms = self.modes + self.inverse_modes
        binaries = [pcs_to_binary(m) for m in all_forms]
        return binary_to_pcs(min(binaries))

    @cached_property
    def propriety(self) -> str:
        """Returns the propriety of the pitch-class set.'"""
        a = self.ambiguities
        c = self.contradictions
        if a == 0 and c == 0:
            return 'Strictly Proper'
        elif c == 0:
            return 'Proper'
        else:
            return 'Improper'

    @cached_property
    def has_max_evenness(self) -> bool:
        """Returns whether the pitch-class set is maximally even."""
        n = self.cardinality
        ideal = tuple(int(math.floor(12*i / n)) for i in range(n))
        return self.prime == ideal

    # ----------------------------------------------------------------
    # Geometric & combinatorial metrics
    # ----------------------------------------------------------------
    
    @cached_property
    def i_circ(self) -> Tuple[complex]:
        """Unit‐circle complex positions of each pitch-class in order."""
        return tuple( _UNIT_CIRCLE[p] for p in self._pcs)
    
    @cached_property
    def brightness(self) -> int:
        """Returns the brightness of the pitch-class set."""
        return sum(self._pcs)

    @cached_property
    def stability(self) -> float: # not working
        """Returns the stability of the pitch-class set."""
        n = self.cardinality
        return 1 - (2*self.ambiguities / (n*(n-1)))

    @cached_property
    def cohemitonia(self) -> int:
        """Returns the count of cohemitones in the pitch-class set."""
        p = self._pcs
        return sum(1 for i in p if ((i+1)%12 in p and (i+2)%12 in p))

    @cached_property
    def imperfections(self) -> int:
        """Returns the count of imperfections in the pitch-class set."""
        return self.cardinality - self.vector[4]


    def _count_failure(self, relation: str = "contradiction") -> int:
        """Returns the total number of element-wise failures of the given type (contradiction, ambiguity) in the pitch-class set."""
        M = np.array(self.matrix)
        rows, cols = M.shape
        count = 0
    
        comparator = {
            "contradiction": lambda x, y: x < y,
            "ambiguity": lambda x, y: x == y
        }.get(relation)
    
        for i in range(1, rows):
            for j in range(cols):
                for p in range(i):
                    for q in range(cols):
                        if comparator(M[i, j], M[p, q]):
                            count += 1
        return count

    @cached_property
    def ambiguities(self) -> int:
        """Returns the count of ambiguities in the pitch-class set."""
        return self._count_failure('ambiguity')

    @cached_property
    def contradictions(self) -> int:
        """Returns the count of contradictions in the pitch-class set."""
        return self._count_failure()

    @cached_property
    def differences(self) -> int:
        """Returns the count of differences in the pitch-class set."""
        return sum(
            1 for row in self.matrix
              for s1, s2 in combinations(row, 2)
              if s1 != s2
        )

    @cached_property
    def heteromorphic_profile(self) -> Tuple[int]:
        """Returns the tuple: (contradictions, ambiguities, differences)."""
        return (self.contradictions, self.ambiguities, self.differences)

    @cached_property
    def max_failures(self) -> int:
        """Returns the maximum number of coherency failures possible for the cardinality of pitch-class set'"""
        n = self.cardinality
        return n * (n - 1) * (n - 2) * (3*n - 5) // 24

    @cached_property
    def max_differences(self) -> int:
        """Returns the maximum number of interval differences possible for the scale's cardinality.'"""
        n = self.cardinality
        return n * (n - 1) * (n - 1) // 2

    @cached_property
    def coherence_quotient(self) -> float:
        """Returns the coherence quotient of the pitch-class set.'"""
        return 1 - ((self.contradictions + self.ambiguities) / self.max_failures if self.max_failures else 1)

    @cached_property
    def sameness_quotient(self) -> float:
        """Returns the sameness quotient of the pitch-class set.'"""
        return 1 - (self.differences / self.max_differences if self.max_differences else 1)

    def _gen_origin(self) -> Tuple[int]:
        """Returns generator, origin pair of the pitch-set class."""
        p = self._pcs
        for g in range(1,12):
            for o in range(12):
                if {(o + k*g)%12 for k in range(self.cardinality)} == set(p):
                    return g, o
        return None, None

    @cached_property
    def generator(self) -> int:
        """Returns the generator of the pitch-class set, if it has one."""
        return self._gen_origin()[0]

    @cached_property
    def origin(self) -> int:
        """Returns the origin of the pitch-class set, if it has one."""
        return self._gen_origin()[1]

    # ----------------------------------------------------------------
    # Geometry on the unit‐circle
    # ----------------------------------------------------------------
    
    @cached_property
    def area(self) -> float:
        """Returns the internal area of the cyclical polygon formed by inscribing the pitch-class set onto the unit-circle."""
        n = self.cardinality
        return 0.5 * abs(sum(
            (self.i_circ[i] * self.i_circ[(i+1)%n].conjugate()).imag
            for i in range(n)
        ))

    @cached_property
    def has_max_area(self) -> bool:
        """Returns whether the internal area of the pitch-set is maximal for its cardinality."""
        n = self.cardinality
        ideal = tuple(sorted(round(i * 12 / n) % 12 for i in range(n)))
        return abs(self.area - Scale(ideal).area) < 1e-3

    @cached_property
    def perimeter(self) -> float:
        """Returns the perimeter of the cyclical polygon formed by inscribing the pitch-class set onto the unit-circle."""
        n = self.cardinality
        return sum(
            abs(self.i_circ[i] - self.i_circ[(i+1) % n])
            for i in range(n)
        )

    @cached_property
    def centroid(self) -> complex:
        """Returns the complex center of the cyclical polygon formed by inscribing the pitch-class set onto the unit-circle."""
        n = self.cardinality
        return sum(self.i_circ) / n

    @cached_property
    def centroid_distance(self) -> float:
        """Returns the distance of the complex centroid of the pitch-class set."""
        return abs(self.centroid)

    @cached_property
    def balanced(self) -> bool:
        """Returns whether the complex centroid of the pitch-class set is centered at the origin."""
        return abs(self.centroid) < 1e-10

    @cached_property
    def centroid_angle_degrees(self) -> float:
        """Returns the angle of the complex centroid clockwise from the root (0 degrees), with a 90° shift."""
        if self.balanced:
            return 0.0
        angle = cmath.phase(self.centroid)
        return (math.degrees(angle) + 90) % 360

    @cached_property
    def centroid_angle_cents(self) -> float:
        """Returns the angle of the complex centroid of the pitch-class set in cents."""
        return 10 * self.centroid_angle_degrees / 3

    @cached_property
    def fourier_components(self) -> Tuple[float]:
        """ Returns the Lewin-Quinn Fourier Coefficients of the pitch-class set.
        FC_k = |sum_{n=0}^11 (x_n * e^(-2πink/12))| for k in 0..6
        """
        x = [int(bit) for bit in self.binary[::-1]]

        def fc(k):
            total = sum(x[n] * cmath.exp(-2j * math.pi * n * k / 12) for n in range(12))
            return abs(total)

        return tuple(round(fc(k), 6) for k in range(7))

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------
    def to_base(self, base: int, rev: bool = False) -> str:
        """Returns the base_{base} representation of the binary string of the pitch-class set."""
        num = self._n if not rev else int(self.binary[::-1], 2)
        if num == 0:
            return "0"
        if not 2 <= base <= len(_DIGITS):
            raise ValueError(f"base must be 2–{len(_DIGITS)}")
        s = ""
        while num:
            num, r = divmod(num, base)
            s = _DIGITS[r] + s
        return s

    # ----------------------------------------------------------------
    # Dunder Methods: arithmetic, comparisons, hashing
    # ----------------------------------------------------------------
    def __add__(self, other):
        """ Scale + Scale = Scale Union;
            Scale + integer = Counterclockwise Rotated Scale"""
        if isinstance(other, Scale):
            return Scale(set(self._pcs) | set(other._pcs))
        if isinstance(other, int):
            return Scale(rotate_pcs(self._pcs, other))
        return NotImplemented

    def __radd__(self, other):
        return self + other if isinstance(other, int) else NotImplemented

    def __sub__(self, other):
        """Scale - Scale = Scale Difference;
        Scale - integer = Clockwise Rotated Scale"""
        if isinstance(other, Scale):
            return Scale(set(self._pcs) - set(other._pcs))
        if isinstance(other, int):
            return Scale(rotate_pcs(self._pcs, -other))
        return NotImplemented

    def __eq__(self, other):
        """Compares binary values."""
        if not isinstance(other, Scale):
            return NotImplemented
        return self._n == other._n

    def __gt__(self, other):
        """Returns True if self is a strict superset of other else False."""
        if isinstance(other, Scale):
            return set(self._pcs) > set(other._pcs)
        if isinstance(other, int):
            return self.cardinality > other
        return NotImplemented

    def __ge__(self, other):
        """Returns True if self is a non-strict superset of other else False."""
        if isinstance(other, Scale):
            return set(self._pcs) >= set(other._pcs)
        if isinstance(other, int):
            return self.cardinality >= other
        return NotImplemented

    def __lt__(self, other):
        """Returns True if self is a strict subset of other else False."""
        if isinstance(other, Scale):
            return set(self._pcs) < set(other._pcs)
        if isinstance(other, int):
            return self.cardinality < other
        return NotImplemented

    def __le__(self, other):
        """Returns True if self is a non-strict subset of other else False."""
        if isinstance(other, Scale):
            return set(self._pcs) <= set(other._pcs)
        if isinstance(other, int):
            return self.cardinality <= other
        return NotImplemented

    def __len__(self):
        """Returns the cardinality of the pitch-set class."""
        return self.cardinality

    def __hash__(self):
        return hash(self._n)
        
    def __repr__(self):
        return f"Scale({self._pcs})"

    @staticmethod
    def _format_matrix(matrix):
        return "\n" + "\n".join("  " + str(row) for row in matrix)

    @property
    def report(self) -> str:
        """Returns a comprehensive report of all properties of the pitch-class set."""
        return (
            f"----Interval Representations----\n"
            f"Forte Number                           : {self.forte}\n"
            f"Binary                                       : {self.binary}\n"
            f"Decimal                                    : {self.to_base(10)}\n"
            f"Xenome                                   : {self.to_base(16, True)}\n"
            f"Base-64                                   : {self.to_base(64)}\n"
            f"Inverse                                     : {self.inverse}\n"
            f"Prime Form                              : {self.prime}\n"
            f"Pitch-Class Set                        : {self._pcs}\n"
            f"Complement                            : {self.complement}\n"
            f"Vector                                      : {self.vector}\n"
            f"Alpha Vector                            : {self.alpha_vector}\n"
            f"Proportional Saturation Vector: {tuple(round(i,3) for i in self.proportional_saturation_vector)}\n"
            f"Structure                                  : {self.structure}\n"
            f"Neighbors: {self._format_matrix(sorted(self.neighbors,key=lambda x: (len(x), x)))}\n"
            f"Neighbors (No Collisions): {self._format_matrix(self.same_size_neighbors)}\n"
            f"Modes: {self._format_matrix(self.modes)}\n"
            f"Spectrum                                 :\n{self.spectrum}\n"
            f"Matrix: {self._format_matrix(self.matrix)}\n"
            f"KMatrix: {self._format_matrix(self.k_matrix)}\n"
            f"Shifted KMatrix: {self._format_matrix(self.s_k_matrix)}\n"
            f"TMatrix: {self._format_matrix(self.t_matrix)}\n"
            f"\n"
            
            f"-----------Scale Categories-----------\n"
            f"Propriety                                   : {self.propriety}\n"
            f"Deep Scale                               : {self.deep}\n"
            f"Myhill Property                          : {self.myhill}\n"
            f"Palindromic                               : {self.palindromic}\n"
            f"Chirality                                     : {self.chiral}\n"
            f"\n"
            
            f"-----------Interval Values-----------\n"
            f"Cardinality                                 : {self.cardinality}\n"
            f"Spectra Variation                       : {self.variation:.3f}\n"
            f"Stability                                      : {self.stability:.3f}\n"
            f"Brightness                                 : {self.brightness:.3f}\n"
            f"Maximally Even Set                  : {self.has_max_evenness}\n"
            f"Hemitonia                                  : {self.interval_class_count(1)}\n"
            f"Tritonia                                      : {self.interval_class_count(6)}\n"
            f"Cohemitonia                              : {self.cohemitonia}\n"
            f"Imperfections                            : {self.imperfections}\n"

            f"Contradictions                           : {self.contradictions}\n"
            f"Ambiguities                               : {self.ambiguities}\n"
            f"Differences                                : {self.differences}\n"
            f"Heteromorphic Profile               : {self.heteromorphic_profile}\n"
            f"Coherence Quotient                  : {self.coherence_quotient:.3f}\n"
            f"Sameness Quotient                  : {self.sameness_quotient:.3f}\n"
            
            f"Generator/Origin                       : {self.generator}, {self.origin}\n"
            f"Ridge Tones                              : {self.ridges}\n"
            f"Rotational Axes                         : {self.rot_axes}\n"
            f"Reflective Axes                         : {self.ref_axes}\n"
            f"\n"
            
            f"---------Geometric Values---------\n"
            f"Area                                          : {self.area:.3f}\n"
            f"Maximal Area Set                      : {self.has_max_area}\n"
            f"Perimeter                                  : {self.perimeter:.3f}\n"
            f"Balanced                                   : {self.balanced}\n"
            f"Centroid                                    : {self.centroid:.3f}\n"
            f"Centroid Distance                     : {self.centroid_distance:.3f}\n"
            f"Centroid Angle (°)                     : {self.centroid_angle_degrees:.3f}°\n"
            f"Centroid Angle (¢)                     : {self.centroid_angle_cents:.3f}¢\n"
            f"Lewin-Quinn FC Components  : {tuple(round(i,3) for i in self.fourier_components)}\n"
        )
