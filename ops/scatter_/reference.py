"""Native oracle. Evaluate before installing A1 overrides."""


def scatter_reference(self, dim, index, src):
    return self.scatter_(dim, index, src)
