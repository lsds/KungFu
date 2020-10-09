class EMA:
    def __init__(self, alpha, scale_cap=None):
        self._alpha = alpha
        self._value = None
        self._scale_cap = scale_cap

    def _cap(self, x):
        if self._scale_cap is None:
            return x
        up = self._value * self._scale_cap
        if x > up:
            return up
        down = self._value / self._scale_cap
        if x < down:
            return down
        return x

    def update(self, x):
        if self._value is None:
            self._value = x
        else:
            x = self._cap(x)
            self._value = self._alpha * self._value + (1 - self._alpha) * x
        return self._value

    def get(self):
        return self._value

    def reset(self):
        self._value = None
