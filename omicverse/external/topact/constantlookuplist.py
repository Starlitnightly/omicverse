from collections.abc import MutableSequence
from typing import Any, Iterator, Iterable


class ConstantLookupList(MutableSequence):
    """A list with constant time index lookup."""

    def __init__(self, values: Iterable = ()):
        self._lst = list(values)
        self._lookup = {}
        self._generate_lookup()

    def _generate_lookup(self):
        for i, value in enumerate(self._lst):
            self._lookup[value] = i

    def __delitem__(self, index: int):
        self._lookup.pop(self._lst[index])
        del self._lst[index]
        self._generate_lookup()

    def __getitem__(self, index: int) -> Any:
        return self._lst[index]

    def __setitem__(self, index: int, value: Any):
        if value not in self:
            old_value = self._lst[index]
            self._lookup.pop(old_value)
            self._lst[index] = value
            self._lookup[value] = index % len(self)
        else:
            old_index = self.index(value)
            if index % len(self) != old_index:
                raise ValueError(f"{value} already at index {old_index}")

    def __len__(self):
        return len(self._lst)

    def insert(self, index: int, value: Any):
        if value in self:
            raise ValueError(f"{value} already at index {self.index(value)}")
        try:
            self._lst.insert(index, value)
            self._generate_lookup()
        except OverflowError as error:
            raise OverflowError from error

    def __contains__(self, value: Any) -> bool:
        return value in self._lookup

    def __iter__(self) -> Iterator:
        yield from self._lst

    def __reversed__(self) -> Iterator:
        yield from reversed(self._lst)

    def index(self,
              value: Any,
              start: int | None = None,
              stop: int | None = None
              ) -> int:
        start = start or 0
        stop = stop or len(self)
        try:
            index = self._lookup[value]
            if start <= index < stop:
                return index
            raise ValueError(f"{value} not in slice from {start} to {stop}.")
        except KeyError as error:
            raise KeyError(error) from error

    def count(self, value: Any) -> int:
        return 1 if value in self else 0

    def reverse(self):
        self._lst.reverse()
        self._generate_lookup()

    def __bool__(self) -> bool:
        return bool(self._lst)

    def __repr__(self) -> str:
        return repr(self._lst)

    def __str__(self) -> str:
        return str(self._lst)
