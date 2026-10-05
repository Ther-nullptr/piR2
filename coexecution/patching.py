"""Transactional attribute replacement shared by scoped runtime adapters."""


class ScopedReplacements:
    def __init__(self):
        self.restore = []

    def replace(self, target, name, replacement):
        existed = name in vars(target)
        original = getattr(target, name)
        self.restore.append((target, name, existed, original))
        setattr(target, name, replacement)

    def __enter__(self):
        try:
            return self._install()
        except BaseException:
            self.__exit__()
            raise

    def _install(self):
        raise NotImplementedError

    def __exit__(self, *exception):
        for target, name, existed, original in reversed(self.restore):
            if existed:
                setattr(target, name, original)
            else:
                delattr(target, name)
        self.restore.clear()
