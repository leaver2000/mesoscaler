from __future__ import annotations

import enum
import functools
import itertools
import json
import types
import typing
import urllib.parse
from collections.abc import Sequence

import numpy as np
import toml

try:
    get_ipython  # type: ignore
    import tqdm.notebook as tqdm  # pyright: ignore
except (NameError, ImportError):
    try:
        import tqdm  # type: ignore
    except ImportError:
        tqdm = None  # type: ignore


from ._typing import (
    TYPE_CHECKING,
    Any,
    Array,
    ArrayLike,
    Callable,
    ChainableItems,
    GenericAliasType,
    Hashable,
    Iterable,
    Iterator,
    ListLike,
    Literal,
    Mapping,
    N,
    NamedTuple,
    NDArray,
    NewType,
    Number_T,
    NumpyGeneric_T,
    NumpyNumber_T,
    Pair,
    Quad,
    Self,
    Sequence,
    Sized,
    StrPath,
    TimeSlice,
    Trips,
    TypeGuard,
    TypeVar,
    overload,
)

__NoDefault = enum.Enum("", "NoDefault")
NoDefault = __NoDefault.NoDefault
LiteralNoDefault = Literal[__NoDefault.NoDefault]
del __NoDefault

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


# =====================================================================================================================
# - logic
# =====================================================================================================================
def has_attrs(x: Any, *attrs: str) -> bool:
    return all(hasattr(x, attr) for attr in attrs)


def is_named_tuple(x: Any) -> TypeGuard[NamedTuple]:
    return isinstance(x, tuple) and hasattr(x, "_fields")


def is_pair(x: Any, strict: bool = False) -> TypeGuard[Pair[Any]]:
    condition = isinstance(x, tuple) and len(x) == 2
    if condition and strict:
        y, z = x
        condition &= type(y) == type(z)
    return condition


if TYPE_CHECKING:

    def pair(x: Pair[_T1] | _T1, /) -> Pair[_T1]:
        ...

    def triples(x: Trips[_T1] | _T1, /) -> Trips[_T1]:
        ...

    def quads(x: Quad[_T1] | _T1, /) -> Quad[_T1]:
        ...

else:

    def _fixed_tuple(n: int):
        def parse(x: Any):
            if isinstance(x, Iterable) and not isinstance(x, str):
                t = tuple(x)
                if len(t) != n:
                    raise ValueError(f"invalid tuple: {t}")
                return t
            return tuple(itertools.repeat(x, n))

        return parse

    pair = _fixed_tuple(2)
    triples = _fixed_tuple(3)
    quads = _fixed_tuple(4)


def iter_pair(x: Pair[_T1] | Iterable[Pair[_T1]], /) -> Iterator[Pair[_T1]]:
    if is_pair(x):
        yield x
    elif isinstance(x, Iterable):
        if isinstance(x, str):
            yield x, x  # type: ignore
        else:
            yield from typing.cast(Iterable[Pair[_T1]], x)
    else:
        raise TypeError(f"invalid pair: {x}")


# =====================================================================================================================
# - time utils
# =====================================================================================================================
# TODO: move this to time64
def slice_time(t: Array[[...], np.datetime64], s: TimeSlice, /) -> Array[[N], np.datetime64]:
    if s.start is None or s.stop is None or s.step is not None:
        raise ValueError(f"invalid slice: {s}")
    return t[(s.start <= t) & (t <= s.stop)]


# =====================================================================================================================
# - array/tensor utils
# =====================================================================================================================
def batch(x: Array[[N], NumpyGeneric_T], n: int, *, strict: bool = True) -> Array[[N, N], NumpyGeneric_T]:
    """
    >>> from src.mesoscaler import utils
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> utils.batch(a, 5)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> utils.batch(a, 4)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/leaver/mesoscaler/src/mesoscaler/utils.py", line 169, in batch
        raise ValueError(
    ValueError: Array size 10 is not divisible by batch size 4.
    try using a size of:
    [ 1  2  5 10]
    >>> utils.batch(a, 4, strict=False)
    array([[0, 1, 2, 3],
           [4, 5, 6, 7],
           [8, 9, 0, 0]])
    """
    s = x.size
    m = s % n
    if m and strict:
        y = np.arange(1, s + 1)
        raise ValueError(
            f"Array size {s} is not divisible by batch size {n}.\ntry using a size of:\n{y[(s % y) == 0]}"
        )
    elif m:
        x = np.pad(x, (0, n - m))

    return x.reshape(-1, n)


def normalize(x: Array[[...], np.number[Any]]) -> Array[[...], np.float_]:
    """
    Normalize the input tensor along the specified dimensions.

    Args:
        x: Input tensor to be normalized.
        **kwargs: Additional arguments to be passed to the `min` and `max` functions.

    Returns:
        Normalized tensor.

    Raises:
        TypeError: If the input tensor is not a numpy array or a PyTorch tensor.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input tensor must be a numpy array or a PyTorch tensor.")
    return (x - x.min()) / (x.max() - x.min())


def normalized_scale(x: NDArray[np.number[Any]], rate: float = 1.0) -> NDArray[np.float_]:
    """
    Scales the input tensor `x` by a factor of `rate` after normalizing it.

    Args:
        x (numpy.ndarray or torch.Tensor): The input tensor to be normalized and scaled.
        rate (float, optional): The scaling factor. Defaults to 1.0.

    Returns:
        numpy.ndarray or torch.Tensor: The normalized and scaled tensor.
    """
    x = normalize(x)
    x *= rate
    x += 1

    return x


def log_scale(x: NDArray[np.number[Any]], rate: float = 1.0) -> NDArray[np.float_]:
    return normalized_scale(np.log(x), rate=rate)


def nd_union(__x: Iterable[ArrayLike[NumpyGeneric_T]], /, *, sort: bool = False) -> Array[[N], NumpyGeneric_T]:
    x = np.asarray(functools.reduce(np.union1d, __x))
    if sort:
        x.sort()
    return x


def nd_intersect(__x: Iterable[ArrayLike[NumpyGeneric_T]], /, *, sort: bool = False) -> Array[[N], NumpyGeneric_T]:
    """
    >>> x = ms.hours.batch("2022-01-01", "2023-01-02", 3, size=4)
    >>> y = ms.hours.batch("2023-01-01", "2023-02-01", 3, size=4)
    >>> z = ms.hours.batch("2023-01-01", "2023-03-01", 3, size=4)
    >>> ms.nd_intersect([x, y, z], sort=True)
    array(['2023-01-01T00', '2023-01-01T03', '2023-01-01T06', '2023-01-01T09',
           '2023-01-01T12', '2023-01-01T15', '2023-01-01T18', '2023-01-01T21'],
            dtype='datetime64[h]')
    """
    x = np.asarray(functools.reduce(np.intersect1d, __x))
    if sort:
        x.sort()
    return x


@overload
def sort_unique(
    __x: Array[[...], NumpyNumber_T], /, *, descending: bool = False, axis: int | None = None
) -> Array[[...], NumpyNumber_T]:
    ...


@overload
def sort_unique(__x: ListLike[Number_T], /, *, descending: bool = False) -> list[Number_T]:
    ...


def sort_unique(
    __x: Iterable[Number_T] | Array[[...], NumpyNumber_T], /, *, descending: bool = False, axis: int | None = None
) -> list[Number_T] | Array[[...], NumpyNumber_T]:
    x = (
        np.sort(np.unique(__x, axis=axis)) if isinstance(__x, np.ndarray) else sorted(set(__x))
    )  # type: list[Number_T] | Array[[...], NumpyNumber_T]

    if descending:
        x = x[::-1]
    return x


# =====================================================================================================================
# - repr utils
# =====================================================================================================================
class Representation(str):
    array_ = staticmethod(
        functools.partial(np.array2string, max_line_width=100, precision=2, separator=" ", floatmode="fixed")
    )

    @classmethod
    def map_(cls, x: Iterable[Any], *, none: Any = None) -> map[Self]:
        return map(lambda x: cls(x, none=none), x)

    @classmethod
    def join_(cls, sep: str, iterable: Iterable[Any], *, none: Any = None) -> str:
        return sep.join(cls.map_(iter(iterable), none=none))

    @classmethod
    def slice_(cls, s: slice) -> str:
        return cls.join_(":", [s.start, s.stop, s.step])

    @classmethod
    def generic_alias_(cls, x: types.GenericAlias) -> str:
        return f"{x.__origin__.__name__}[{cls.join_(', ', cls.map_(x.__args__))}]"

    @classmethod
    def sequence_(cls, x: Sequence[Any]) -> str:
        if isinstance(x, tuple):
            left, right = "(", ")"
        elif isinstance(x, list):
            left, right = "[", "]"
        else:
            left, right = "{", "}"

        return left + cls.join_(", ", x) + right

    def __new__(cls, x: Any, *, none: Any = None) -> Representation:
        if isinstance(x, Representation):
            return x
        elif isinstance(x, str) or is_named_tuple(x):
            pass
        elif isinstance(x, np.datetime64):
            x = np.datetime_as_string(x, unit="s") + "Z"
        elif isinstance(x, slice):
            x = cls.slice_(x)
        elif np.isscalar(x):
            x = repr(x)
        elif isinstance(x, np.ndarray):
            x = cls.array_(x)
        elif isinstance(x, Sequence):
            x = cls.sequence_(x)
        elif x is ...:
            x = "..."
        elif isinstance(x, (types.GenericAlias, GenericAliasType)):
            x = cls.generic_alias_(x)
        elif x is None:
            x = none
        elif isinstance(x, (type, NewType)):
            x = getattr(x, "__name__", repr(x))
        return super().__new__(cls, x)


@overload
def repr_(x: Any, *, none: Any = None, map_values: Literal[False] = False) -> Representation:
    ...


@overload
def repr_(x: Iterable[Any], *, none: Any = None, map_values: Literal[True]) -> map[Representation]:
    ...


def repr_(x: Any, *, none: Any = None, map_values: bool = False) -> Representation | map[Representation]:
    return Representation.map_(x, none=none) if map_values else Representation(x, none=none)


def _repr_generator(*args: tuple[str, Any], prefix: str = "- ") -> Iterator[str]:
    k, _ = zip(*args)
    width = max(map(len, k))
    for key, value in args:
        key = f"{prefix}{key.rjust(width)}: "
        if isinstance(value, np.ndarray) and value.ndim > 1:
            key += "\n"

        yield f"{key}{repr_(value)}"


def join_kv(
    head: tuple[Hashable, Any] | str | type,
    *args: tuple[Hashable, Any],
    sep="\n",
    start: int | LiteralNoDefault = NoDefault,
    stop: int | LiteralNoDefault = NoDefault,
) -> str:
    if isinstance(head, tuple):
        args = (head, *args)
        head = ""
    elif isinstance(head, type):
        head = f"{head.__name__}:"

    if start is not NoDefault and stop is not NoDefault:
        text = sep.join(_repr_generator(*((str(k), v) for k, v in args[start:stop])))
        text += "\n...\n"
        k, v = args[-1]
        text += sep.join(_repr_generator((str(k), v)))

    else:
        text = sep.join(_repr_generator(*((str(k), v) for k, v in args)))
    return sep.join([head, text])


# =====================================================================================================================
# - iterable utils
# =====================================================================================================================
SizedIterFunc = Callable[[Iterable[Sized]], _T1]
map_size: functools.partial[map[int]] = functools.partial(map, len)
acc_size: SizedIterFunc[Iterable[int]] = lambda x: itertools.accumulate(map_size(x))
max_size: SizedIterFunc[int] = lambda x: max(map_size(x))
min_size: SizedIterFunc[int] = lambda x: min(map_size(x))
sum_size: SizedIterFunc[int] = lambda x: sum(map_size(x))


@overload
def chain_items(__x: tuple[_T1, _T2], /, *args: tuple[_T1, _T2]) -> itertools.chain[tuple[_T1, _T2]]:
    ...


@overload
def chain_items(__x: ChainableItems[_T1, _T2], /, *args: tuple[_T1, _T2]) -> itertools.chain[tuple[_T1, _T2]]:
    ...


@overload
def chain_items(
    __x: ChainableItems[str, _T2], /, *args: tuple[str, _T2], **kwargs: _T2
) -> itertools.chain[tuple[str, _T2]]:
    ...


def chain_items(
    __x: tuple[_T1, _T2] | ChainableItems[_T1, _T2], /, *args: tuple[_T1, _T2], **kwargs: _T2
) -> itertools.chain[tuple[Any, Any]]:
    """
    >>> assert (
        list(utils.chain_items({"a": 1, "b": 2}))
        == list(utils.chain_items([("a", 1), ("b", 2)]))
        == list(utils.chain_items(("a", 1), ("b", 2)))
        == list(utils.chain_items(zip("ab", (1, 2))))
        == [("a", 1), ("b", 2)]
    )
    """
    if isinstance(__x, tuple):
        x = iter_pair(__x)
    elif isinstance(__x, Mapping):
        x = __x.items()
    else:
        x = __x

    return itertools.chain(x, args, kwargs.items())


# =====================================================================================================================
# - IO utils
# =====================================================================================================================
def dump_toml(obj: Any, src: StrPath, preserve=True, numpy: bool = False) -> None:
    with open(src, "w") as f:
        toml.dump(
            obj, f, encoder=toml.TomlNumpyEncoder(preserve=preserve) if numpy else toml.TomlEncoder(preserve=preserve)
        )


def load_toml(src: StrPath) -> Any:
    with open(src, "r") as f:
        return toml.load(f)


def dump_json(obj: Any, src: StrPath) -> None:
    with open(src, "w") as f:
        json.dump(obj, f)


def load_json(src: StrPath) -> Any:
    with open(src, "r") as f:
        return json.load(f)


def dump_jsonl(obj: Iterable[Any], src: StrPath) -> None:
    with open(src, "w") as f:
        for x in obj:
            json.dump(x, f)
            f.write("\n")


def iter_jsonl(src: StrPath) -> Iterable[Any]:
    with open(src, "r") as f:
        for line in f:
            yield json.loads(line)


def url_join(url: str, *args: str, allow_fragments: bool = True) -> str:
    """
    >>> url_join('https://example.com', 'images', 'cats')
    'https://example.com/images/cats'
    >>> url_join('https://example.com', '/images')
    'https://example.com/images'
    >>> url_join('https://example.com/', 'images')
    'https://example.com/images'
    """
    if not url.startswith("http"):
        raise ValueError(f"invalid url: {url}")
    return urllib.parse.urljoin(url, "/".join(x.strip("/") for x in args), allow_fragments=allow_fragments)
