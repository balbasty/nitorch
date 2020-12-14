class version:
    """Utility class to handle version numbers"""

    _version: tuple = None

    def __new__(cls, version=None):
        """

        Parameters
        ----------
        version : version or str or sequence[int or str] or None, default=None
            A version number.
            Note that bool(version(None)) == False,
            otherwise bool(version(v)) == True
        """
        if isinstance(version, cls):
            return version
        if isinstance(version, str):
            return cls.from_str(version)
        if version is not None:
            return cls.from_tuple(version)
        return object.__new__(cls)

    @classmethod
    def from_str(cls, v):
        """Build version object from a dot-separated version string

        Parameters
        ----------
        v : str
            Version string of the form "<MAJOR>[.<MINOR>[.<PATCH>]]]"
        """
        version = []
        for v0 in v.split('.'):
            try:
                version.append(int(v0))
            except ValueError:
                version.append(v0)
        return cls.from_tuple(version)

    @classmethod
    def from_tuple(cls, v):
        """Build version object from a tuple of version numbers

        Parameters
        ----------
        v : sequence[int or str]
            Version sequence of the form (<MAJOR>, [<MINOR>], [<PATCH>], ...)
            If some components are strings representing integers, they are
            converted to integers. Letters are kept as is (they are usually
            used for development versions: 'a', 'b', 'rc', etc.)
        """
        version = []
        for v0 in v:
            try:
                version.append(int(v0))
            except ValueError:
                version.append(v0)
        obj = cls()
        obj._version = tuple(version)
        return obj

    def __str__(self):
        """Return a dot-separated string representation.

        Returns
        -------
        str
            "<MAJOR>[.<MINOR>][.<PATCH>]"

        """
        if self._version:
            version = ''
            for v in self._version:
                if v is not None:
                    version += str(v)
                else:
                    version += '*'
                version += '.'
            version = version[:-1]
        else:
            version = 'None'
        return version

    def __repr__(self):
        return 'version("{}")'.format(str(self))

    def __getitem__(self, item):
        """Subslice the tuple of version numbers"""
        return self._version[item]

    def __bool__(self):
        """True if the underlying version is not None"""
        return bool(self._version)

    def __len__(self):
        """Number of version numbers effectively stored"""
        return len(self._version) if self._version else 0

    def major(self):
        return self[0] if len(self) > 0 else None

    def minor(self):
        return self[1] if len(self) > 1 else None

    def patch(self):
        return self[2] if len(self) > 2 else None

    def __eq__(self, other):
        """True if versions are equal (as for pip).

        E.g., `version("10.1") == version("10.1.2")` returns `True`.
        """
        if isinstance(other, bool):
            return bool(self) == other
        other = version(other)
        v1 = self._version
        v2 = other._version
        while len(v1) > 0 and len(v2) > 0:
            lead1, *v1 = v1
            lead2, *v2 = v2
            if lead1 is None or lead2 is None:
                continue
            if lead1 != lead2:
                return False
        return True

    def __gt__(self, other):
        v1 = self._version
        v2 = other._version
        while len(v1) > 0 and len(v2) > 0:
            lead1, *v1 = v1
            lead2, *v2 = v2
            if lead1 is None or lead2 is None:
                continue
            if isinstance(lead1, str) and not isinstance(lead2, str):
                return True
            if isinstance(lead2, str) and not isinstance(lead1, str):
                return True
            if lead1 > lead2:
                return True
        return False

    def __lt__(self, other):
        v1 = self._version
        v2 = other._version
        while len(v1) > 0 and len(v2) > 0:
            lead1, *v1 = v1
            lead2, *v2 = v2
            if lead1 is None or lead2 is None:
                continue
            if lead1 < lead2:
                return True
        return False

    def __ge__(self, other):
        return self == other or self > other

    def __le__(self, other):
        return self == other or self < other
