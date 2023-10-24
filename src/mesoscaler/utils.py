from __future__ import annotations

__all__ = [
    "normalize",
    "normalized_scale",
    "sort_unique",
    "arange_slice",
    "interp_frames",
    "square_space",
    "url_join",
    "dump_json",
    "dump_jsonl",
    "dump_toml",
    "iter_jsonl",
    "load_json",
    "load_toml",
    "tqdm",
]
import enum
import functools
import itertools
import json
import types
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
    Any,
    AnyArrayLike,
    Array,
    Callable,
    CanBeItems,
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
    Self,
    Sequence,
    Sized,
    StrPath,
    TimeSlice,
    TypeGuard,
    TypeVar,
    overload,
)

__NoDefault = enum.Enum("", "NoDefault")
NoDefault = __NoDefault.NoDefault
LiteralNoDefault = Literal[__NoDefault.NoDefault]
del __NoDefault

_T1 = TypeVar("_T1", bound=Any)
_T2 = TypeVar("_T2")


# =====================================================================================================================
# - logic
# =====================================================================================================================
def is_ipython() -> bool:
    try:
        get_ipython  # type: ignore
        return True
    except NameError:
        return False


def has_attrs(x: Any, *attrs: str) -> bool:
    return all(hasattr(x, attr) for attr in attrs)


def is_sequence(x: Any) -> TypeGuard[Sequence]:
    return not isinstance(x, (str, bytes)) and has_attrs(x, "__len__", "__iter__")


def is_function(x: Any) -> TypeGuard[function]:
    return isinstance(x, types.FunctionType)


def is_enumtype(x: Any) -> TypeGuard[type[enum.Enum]]:
    return isinstance(x, type) and issubclass(x, enum.Enum)


def is_exception(x: Any) -> TypeGuard[type[Exception]]:
    return isinstance(x, type) and issubclass(x, BaseException)


def is_hashable(x: Any) -> TypeGuard[Hashable]:
    return isinstance(x, Hashable)


def is_scalar(x: Any) -> TypeGuard[np.generic | bool | int | float | complex | str | bytes | memoryview | enum.Enum]:
    return np.isscalar(x) or isinstance(x, enum.Enum)


def is_array_like(x: Any) -> TypeGuard[AnyArrayLike]:
    return hasattr(x, "ndim") and not is_scalar(x)


def is_named_tuple(x: Any) -> TypeGuard[NamedTuple]:
    return isinstance(x, tuple) and hasattr(x, "_fields")


def is_pair(x: Any, strict: bool = False) -> TypeGuard[Pair[Any]]:
    condition = isinstance(x, tuple) and len(x) == 2
    if condition and strict:
        y, z = x
        condition &= type(y) == type(z)
    return condition


def is_null(value: Any) -> bool:
    """
    Check if value is NaN or None
    """
    # pylint: disable=comparison-with-itself
    return value != value or value is None


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


def square_space(in_size: int, out_size: int) -> tuple[Pair[Array[[N], Any]], Pair[Array[[N, N], Any]]]:
    """
    >>> points, values = squarespace(4, 6)
    >>> points
    (array([0.        , 0.08333333, 0.16666667, 0.25      ]), array([0.        , 0.08333333, 0.16666667, 0.25      ]))
    >>> grid
    (array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
           [0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ],
           [0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
           [0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 ],
           [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]), array([[0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25]]))
    """
    xy1 = np.linspace(0, 1.0 / in_size, in_size)
    xy2 = np.linspace(0, 1.0 / in_size, out_size)
    return (xy1, xy1), tuple(np.meshgrid(xy2, xy2, indexing="ij"))  # type: ignore[return-value]


def interp_frames(
    arr: Array[[N, N, ...], _T1], *, img_size: int = 256, method: str = "linear"
) -> Array[[N, N, ...], _T1]:
    """
    Interpolate the first two equally shaped dimensions of an array to the new `patch_size`.
    using `scipy.interpolate.RegularGridInterpolator`.

    >>> import numpy as np
    >>> import atmoformer.utils
    >>> arr = np.random.randint(0, 255, (384, 384, 49))
    >>> atmoformer.utils.interpatch(arr, 768).shape
    (768, 768, 49)
    """
    from scipy.interpolate import RegularGridInterpolator  # type: ignore[import]

    x, y = arr.shape[:2]
    if x != y:  # first two dimensions must be equal
        raise ValueError(f"array must be square, but got shape: {arr.shape}")
    elif x == img_size == y:  # no interpolation needed
        return arr  # pyright: ignore
    points, values = square_space(x, img_size)
    interp = RegularGridInterpolator(points, arr, method=method)
    return interp(values).astype(arr.dtype)


def overlapping(data: Sequence[Array[[N], NumpyGeneric_T]], sort: bool = True) -> Array[[N], NumpyGeneric_T]:
    # TODO: need to check if the function below is equivalent to this function
    # import functools
    # import numpy as np
    # f = functools.reduce(np.intersect1d, data)

    x = np.unique(np.concatenate(data))
    mask = np.stack([np.isin(x, y) for y in data], axis=1).all(axis=1)
    x = x[mask]
    if sort:
        x.sort()
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


# repr_: Final = Representation
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


@overload
def sort_unique(
    __x: Array[[...], NumpyNumber_T], /, *, descending=False, axis: int | None = None
) -> Array[[...], NumpyNumber_T]:
    ...


@overload
def sort_unique(__x: ListLike[Number_T], /, *, descending=False) -> list[Number_T]:
    ...


def sort_unique(
    __x: Iterable[Number_T] | Array[[...], NumpyNumber_T],
    /,
    *,
    descending=False,
    axis: int | None = None,
) -> list[Number_T] | Array[[...], NumpyNumber_T]:
    x = (
        np.sort(np.unique(__x, axis=axis)) if isinstance(__x, np.ndarray) else sorted(set(__x))
    )  # type: list[Number_T] | Array[[...], NumpyNumber_T]

    if descending:
        x = x[::-1]
    return x


# =====================================================================================================================
# - list utils
# =====================================================================================================================
def arange_slice(
    start: int, stop: int | None, rows: int | None, ppad: int | None, step: int | None = None
) -> list[slice]:
    if stop is None:
        start, stop = 0, start
    if ppad == 0:
        ppad = None
    elif stop < start:
        raise ValueError(f"stop ({stop}) must be less than start ({start})")

    stop += 1  # stop is exclusive

    if rows is None:
        rows = 1

    if ppad is not None:
        if ppad > rows:
            raise ValueError(f"pad ({ppad}) must be less than freq ({rows})")
        it = zip(range(start, stop, rows // ppad), range(start + rows, stop, rows // ppad))
    else:
        it = zip(range(start, stop, rows), range(start + rows, stop, rows))
    return [np.s_[i:j:step] for i, j in it if j <= stop]


# =====================================================================================================================
# - iterable utils
# =====================================================================================================================
SizedIterFunc = Callable[[Iterable[Sized]], _T1]
map_size: SizedIterFunc[map[int]] = lambda x: map(len, x)
acc_size: SizedIterFunc[itertools.accumulate[int]] = lambda x: itertools.accumulate(map_size(x))
max_size: SizedIterFunc[int] = lambda x: max(map_size(x))
min_size: SizedIterFunc[int] = lambda x: min(map_size(x))
sum_size: SizedIterFunc[int] = lambda x: sum(map_size(x))


@overload
def find(__func: Callable[[_T1], bool], __x: Iterable[_T1]) -> _T1:
    ...


@overload
def find(__func: Callable[[_T1], bool], __x: Iterable[_T1], /, *, default: _T2) -> _T1 | _T2:
    ...


def find(
    __func: Callable[[_T1], bool], __x: Iterable[_T1], /, *, default: _T2 | LiteralNoDefault = NoDefault
) -> _T1 | _T2:
    try:
        return next(filter(__func, __x))
    except StopIteration as e:
        if default is not NoDefault:
            return default
        raise ValueError(f"no element in {__x} satisfies {__func}") from e


def squish_iter(x: _T1 | Iterable[_T1]) -> Iterator[_T1]:
    """

    >>> list(squish_iter((1,2,3,4)))
    [1, 2, 3, 4]
    >>> list(squish_iter('hello'))
    ['hello']
    >>> list(squish_iter(['hello', 'world']))
    ['hello', 'world']
    """
    if not isinstance(x, Iterable):
        raise TypeError(f"expected an iterable, but got {type(x)}")
    return iter([x] if isinstance(x, str) else x)  # type: ignore


def squish_chain(__iterable: _T1 | Iterable[_T1], /, *args: _T1) -> itertools.chain[_T1]:
    return itertools.chain(squish_iter(__iterable), iter(args))


def squish_map(__func: Callable[[_T1], _T2], __iterable: _T1 | Iterable[_T1], /, *args: _T1) -> map[_T2]:
    """
    >>> assert list(squish_map(lambda x: x, "foo", "bar", "baz")) == ["foo", "bar", "baz"]
    >>> assert list(squish_map(str, range(3), 4, 5)) == ["0", "1", "2", "4", "5"]
    >>> assert list(squish_map("hello {}".format, (x for x in ("foo", "bar")), "spam")) == ["hello foo", "hello bar", "hello spam"]
    """

    return map(__func, squish_chain(__iterable, *args))


def iter_pair(x: Pair[Any] | Iterable[Pair[Any]]) -> Iterator[Pair[Any]]:
    if is_pair(x):
        yield x
    else:
        yield from x


@overload
def items(x: tuple[_T1, _T2], /, *args: tuple[_T1, _T2]) -> itertools.chain[tuple[_T1, _T2]]:
    ...


@overload
def items(x: CanBeItems[_T1, _T2], /, *args: tuple[_T1, _T2]) -> itertools.chain[tuple[_T1, _T2]]:
    ...


@overload
def items(x: CanBeItems[str, _T2], /, *args: tuple[str, _T2], **kwargs: _T2) -> itertools.chain[tuple[str, _T2]]:
    ...


def items(
    x: tuple[_T1 | str, _T2] | CanBeItems[_T1 | str, _T2], /, *args: tuple[_T1 | str, _T2], **kwargs: _T2
) -> itertools.chain[tuple[_T1, _T2]] | itertools.chain[tuple[str, _T2]]:
    """
    >>> assert (
        list(utils.items({"a": 1, "b": 2}))
        == list(utils.items([("a", 1), ("b", 2)]))
        == list(utils.items(("a", 1), ("b", 2)))
        == list(utils.items(zip("ab", (1, 2))))
        == [("a", 1), ("b", 2)]
    )
    """
    if isinstance(x, tuple):
        x = iter_pair(x)

    return itertools.chain((x.items() if isinstance(x, Mapping) else x), args, kwargs.items())


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
