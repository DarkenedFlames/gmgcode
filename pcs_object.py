from .handlers.conversion import ConversionHandler
from .handlers.transformation import TransformationHandler
from .handlers.intervallic import IntervallicHandler
from .handlers.geometric import GeometricHandler
from .display.clocks import draw_clock

class PCSet:
    def __init__(self, initial, ET):
        self._initial = initial
        self._ET      = ET
        self.pcs      = self.convert.to_pcs
        self.mask     = self.convert.to_mask
        self.binary   = self.convert.to_binary
        self.cardinality = self.mask.bit_count()
        
    @property
    def convert(self):
        return ConversionHandler(self._initial, self._ET)
    
    @property
    def transform(self):
        return TransformationHandler(self.mask, self._ET)

    @property
    def intervallic(self):
        return IntervallicHandler(self.pcs, self._ET)
        
    @property
    def geometric(self):
        return GeometricHandler(self.pcs, self._ET)

    def draw(self, **kwargs):
        args = {
            'n_nodes': self._ET,
            'active_nodes': self.pcs,
        }
        args.update(kwargs)
        draw_clock(**args)
    
    @property
    def play(self):
        """Launch the interactive pitch-class set game using this instance."""
        from . import game
        game.start_with(self)
    
    @property
    def power_set(self):
        N = self._ET
        return set(PCSet(i, n) for i in range(1 << n))
    
    
    
    def __invert__(self):
        """Bitwise NOT → inverse."""
        return self.transform.reflect()

    def __neg__(self):
        """Unary minus → complement."""
        return self.transform.complement

    def __lshift__(self, k):
        """Left shift (<< k) → rotate CCW by k semitones."""
        return self.transform.rotate(k)

    def __rshift__(self, k):
        """Right shift (>> k) → rotate CW by k semitones."""
        return self.transform.rotate(-k)

    def __eq__(self, other):
        """Equality: same bitmask."""
        return isinstance(other, PCSet) and self.mask == other.mask

    def __lt__(self, other):
        """For total_ordering: compare bitmask."""
        return isinstance(other, PCSet) and self.mask < other.mask

    def __len__(self):
        """Cardinality."""
        return self.cardinality

    def __hash__(self):
        """Hash based on integer representation."""
        return hash(self.mask)

    def __iter__(self):
        """Iterate over pcs."""
        return iter(self.pcs)

    def __getitem__(self, index):
        """Index into pcs."""
        return self.pcs[index]

    def __contains__(self, p):
        """Membership test in pcs."""
        return p in self.pcs

    def __repr__(self):
        """Debug representation."""
        return f'PCSet({self.pcs}, {self._ET})'
        
    def __str__(self):
        """String Representation."""
        return f'PCSet({self.pcs}, {self._ET})'

    