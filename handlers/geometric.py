import cmath, math
from typing import Tuple, Iterable


class GeometricHandler:
    def __init__(self, pcs, ET):
        self.pcs = pcs
        self._ET = ET
        self.cardinality = len(self.pcs)

    @property
    def generator(self) -> Tuple[int, int]:
        """Returns generator/origin pair of the pitch-class set, if it has one."""
        N = self._ET
        for g in range(1, 1 + N // 2):
            for o in range(N):
                if { (o + g * k) % N for k in range(self.cardinality) } == set(self.pcs):
                    return (g, o)
        return tuple()

    @property
    def _circle(self) -> Tuple[complex, ...]:
        """Returns the complex unit‐circle positions of the pitch-class set."""
        return tuple(cmath.exp(p * 2j * math.pi / self._ET) for p in self.pcs)

    @property
    def _consec(self) -> Iterable:
        """Returns consecutive pairs in self._circle."""
        c = self._circle
        return zip(c, c[1:] + c[:1])

    @property
    def area(self) -> float:
        """Returns the internal area of the pitch-class set."""
        return 0.5 * abs(sum((p * q.conjugate()).imag for p, q in self._consec))

    @property
    def perimeter(self) -> float:
        """Returns the perimeter of the pitch-class set."""
        return sum(abs(p - q) for p, q in self._consec)

    @property
    def fourier(self) -> Tuple[float, float, float, float, float, float, float]:
        """Returns the Lewin-Quinn Fourier Coefficients of the pitch-class set."""
        return tuple(abs(sum(c ** -k for c in self._circle)) for k in range(1 + self._ET // 2))
    

    @property
    def has_max_area(self) -> bool:
        """Returns whether the internal area of the pitch-class set is maximal for its cardinality."""
        n = self.cardinality
        N = self._ET
        ideal = tuple(N * i // n for i in range(n))
        ideal_area = GeometricHandler(ideal, N).area
        return abs(self.area - ideal_area) < 1e-9

    @property
    def has_max_evenness(self) -> bool:
        """Returns whether the pitch-class set is maximally even for its cardinality."""
        pass # move to different module

    @property
    def centroid(self) -> complex:
        """Returns the complex center of the pitch-class set."""
        return -1j * sum(self._circle) / self.cardinality

    @property
    def centroid_distance(self) -> float:
        """Returns the distance of the complex centroid of the pitch-class set."""
        return abs(self.centroid)

    @property
    def is_balanced(self) -> bool:
        """Returns whether the complex centroid of the pitch-class set is centered at the origin."""
        return abs(self.centroid) < 1e-9

    @property
    def centroid_angle_degrees(self) -> float:
        """Returns the angle of the complex centroid clockwise from the root (0°)."""
        return None if self.is_balanced else (math.degrees(cmath.phase(self.centroid)) + 90) % 360

    @property
    def centroid_angle_cents(self) -> float:
        """Returns the angle of the complex centroid of the Sclae in cents."""
        a = self.centroid_angle_degrees
        return None if a is None else 10 * a / 3
