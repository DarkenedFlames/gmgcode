                
class ConversionHandler:
    def __init__(self, data, ET):
        self._ET = ET
        self._N = 1 << self._ET
        self._MASK = self._N - 1
        
        if isinstance(data, (float, int)):
            self._mask   = int(data) & self._MASK
            self._binary = self.dec2bin(self._mask)
            self._pcs    = self.bin2pcs(self._binary)
            
        elif (isinstance(data, (tuple, list))
              and len(data) <= self._ET
              and set(data) <= {0,1}):
            data = list(data)
            while len(data) < self._ET: data.append(0)
            self._binary = tuple(data)
            self._pcs    = self.bin2pcs(self._binary)
            self._mask   = self.pcs2dec(self._pcs)
            
        elif (isinstance(data, (tuple, set, list))
              and len(data) <= self._ET
              and set(data) <= set(range(self._ET))):
            self._pcs    = tuple(sorted(set(p % self._ET for p in data)))
            self._mask   = self.pcs2dec(self._pcs)
            self._binary = self.dec2bin(self._mask)
             
        else:
            raise ValueError(f'PCSet cannot interpret {data} for initialization.')
        
        
    
    def dec2bin(self, dec):
        return tuple((dec >> i) & 1 for i in range(self._ET))
            
    def pcs2dec(self, pcs):
        return sum(1 << p for p in pcs) 
        
    def bin2pcs(self, bi):
        return tuple(i for i, b in enumerate(bi) if b)
        
    @property
    def to_mask(self):
        return self._mask
            
    @property
    def to_binary(self):
        return self._binary
            
    @property
    def to_pcs(self):
        return self._pcs   