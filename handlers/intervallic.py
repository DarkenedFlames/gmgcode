from .vector_extremes import min_vector, max_vector
from .transformation import TransformationHandler
from typing import Tuple, Dict, Set
from itertools import combinations

class IntervallicHandler:
    def __init__(self, pcs, ET):
        self.pcs = pcs
        self._ET = ET
        self.cardinality = len(self.pcs)
    
    
    # ---------------------------------------------------------------------------------
    # Interval Tensors
    # ---------------------------------------------------------------------------------

    def interval_class(self, a, b):
        """Returns the interval class between two pitch-classes."""
        return min((b - a) % self._ET, (a - b) % self._ET)
    
    @property
    def vector(self):
        """Returns the interval vector of a pitch-class set."""
        ics = self._ET // 2
        vector = [0] * ics
        for a, b in combinations(self.pcs, 2):
            ic = self.interval_class(a, b)
            if 1 <= ic <= ics:
                vector[ic - 1] += 1
        return tuple(vector)
    
    @property
    def p_sat_vector(self) -> Tuple[float, float, float, float, float, float]:
        """Returns the proportional saturation vector of the pitch-class set."""
        M, m = max_vector[self.cardinality], min_vector[self.cardinality]
        return tuple((self.vector[i] - m[i]) / (M[i] - m[i]) if (M[i] - m[i]) != 0 else 1 for i in range(self._ET // 2))

    @property
    def matrix(self) -> Tuple[Tuple[int, ...], ...]:
        """Returns the interval matrix of the pitch-class set."""
        p = self.pcs
        n = self.cardinality
        return tuple(tuple(
        (p[(j + i) % n] - p[j]) % self._ET 
            for j in range(n))
            for i in range(1, n)
        )
    
    @property
    def spectra(self) -> Dict[int, Set[int]]:
        """Returns the interval spectra of the pitch-class set."""
        return {i: set(r) for i, r in enumerate(self.matrix, start=1)}

    @property
    def structure(self) -> Tuple[int, ...]:
        """Returns the interval structure of the pitch-class set."""
        return self.matrix[0]
    
    @property
    def is_deep(self) -> bool:
        """Returns whether the pitch-class set is deep."""
        return len(set(self.vector)) == 6

    @property
    def has_myhill(self) -> bool:
        """Returns whether the pitch-class set has Myhill's property."""
        return all(len(v) == 2 for v in self.spectra.values())

    @property
    def variation(self) -> float:
        """Returns the spectra variation of the pitch-class set."""
        return sum(max(r) - min(r) for r in self.spectra.values()) / self.cardinality

    @property # 12-ET ONLY
    def is_maximally_even(self):
        return self.variation < 1

    @property
    def propriety(self) -> str:
        """Returns the propriety of the pitch-class set."""
        a, c = self.ambiguities, self.contradictions
        return 'Improper' if a * c else 'Proper' if a else 'Strictly Proper'

    @property
    def brightness(self) -> int:
        """Returns the brightness of the pitch-class set."""
        return sum(self.pcs)

    @property
    def cohemitonia(self) -> int:
        """Returns the count of cohemitones in the pitch-class set."""
        return sum(
            1 for i in self.pcs
            if { (i + 1) % self._ET, (i + 2) % self._ET }.issubset(set(self.pcs))
        )

    @property
    def imperfections(self) -> int:
        """Returns the count of imperfections in the pitch-class set."""
        return self.cardinality - self.vector[4]

    def _count_failure(self, attr: str = 'contradiction') -> int:
        """Returns the total number of coherency failures of the given type
          (contradiction, ambiguity) in the pitch-class set."""
        M = self.matrix
        n = self.cardinality
    
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

    @property
    def differences(self) -> int:
        """Returns the count of differences in the pitch-class set."""
        return sum(
            1 for row in self.matrix
              for a, b in combinations(row, 2)
              if a != b
        )

    @property
    def heteromorphic_profile(self) -> Tuple[int, int, int]:
        """Returns the tuple: (contradictions, ambiguities, differences)."""
        return (self.contradictions, self.ambiguities, self.differences)

    @property
    def max_failures(self) -> int:
        """Returns the maximum number of coherency failures possible for the cardinality of the pitch-class set."""
        n = self.cardinality
        return n * (n - 1) * (n - 2) * (3 * n - 5) // 24

    @property
    def max_differences(self) -> int:
        """Returns the maximum number of interval differences possible for the
          cardinality of the pitch-class set."""
        n = self.cardinality
        return n * (n - 1) * (n - 1) // 2

    @property
    def coherence_quotient(self) -> float:
        """Returns the coherence quotient of the pitch-class set."""
        c, a, d = self.heteromorphic_profile
        m = self.max_failures
        return 1 - ((c + a) / m if m else 1)

    @property
    def sameness_quotient(self) -> float:
        """Returns the sameness quotient of the pitch-class set."""
        c, a, d = self.heteromorphic_profile
        m = self.max_differences
        return 1 - (d / m if m else 1)