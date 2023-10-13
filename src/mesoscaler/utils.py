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
import collections
import datetime
import enum
import functools
import itertools
import json
import types
import urllib.parse
from collections.abc import Sequence

import numpy as np
import pandas as pd
import toml
from pandas._typing import DtypeObj

try:
    get_ipython  # type: ignore
    import tqdm.notebook as tqdm
except (NameError, ImportError):
    try:
        import tqdm  # type: ignore
    except ImportError:
        tqdm = None  # type: ignore
from ._torch_compat import Tensor
from ._typing import (
    Any,
    AnyArrayLike,
    Array,
    Callable,
    Final,
    Hashable,
    Iterable,
    Iterator,
    ListLike,
    Literal,
    Mapping,
    N,
    NDArray,
    Pair,
    Sequence,
    Sized,
    StrPath,
    TypeGuard,
    TypeVar,
    overload,
)

_T1 = TypeVar("_T1", bound=Any)
_T2 = TypeVar("_T2")

_NoDefault = enum.Enum("_NoDefault", "NoDefault")
NO_DEFAULT = _NoDefault.NoDefault
NoDefault = Literal[_NoDefault.NoDefault]
del _NoDefault


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


# =====================================================================================================================
# - array/tensor utils
# =====================================================================================================================
TensorT = TypeVar("TensorT", Tensor, Array[..., Any])


def normalize(x: TensorT) -> TensorT:
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
    if not isinstance(x, (np.ndarray, Tensor)):
        raise TypeError("Input tensor must be a numpy array or a PyTorch tensor.")
    return (x - x.min()) / (x.max() - x.min())  # pyright: ignore


def normalized_scale(x: TensorT, rate: float = 1.0) -> TensorT:
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


def log_scale(x: NDArray[np.number], rate: float = 1.0) -> NDArray[np.float_]:
    return normalized_scale(np.log(x), rate=rate)


def sort_unique(x: ListLike[_T1], descending=False) -> NDArray[_T1]:
    """
    Sorts the elements of the input array `x` in ascending order and removes any duplicates.

    Parameters
    ----------
    x : ListLike[_T1]
        The input array to be sorted and made unique.

    Returns
    -------
    NDArray[_T1]
        A new array containing the sorted, unique elements of `x`.

    Examples
    --------
    >>> sort_unique([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    array([1, 2, 3, 4, 5, 6, 9])
    """
    a = np.sort(np.unique(np.asanyarray(x)))
    if descending:
        a = a[::-1]
    return a


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
    arr: Array[[N, N, ...], _T1],
    *,
    img_size: int = 256,
    method: str = "linear",
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
    from scipy.interpolate import RegularGridInterpolator

    x, y = arr.shape[:2]
    if x != y:  # first two dimensions must be equal
        raise ValueError(f"array must be square, but got shape: {arr.shape}")
    elif x == img_size == y:  # no interpolation needed
        return arr  # pyright: ignore
    points, values = square_space(x, img_size)
    interp = RegularGridInterpolator(points, arr, method=method)
    return interp(values).astype(arr.dtype)


# =====================================================================================================================
# - repr utils
# =====================================================================================================================
_array2string = functools.partial(
    np.array2string,
    max_line_width=100,
    precision=2,
    separator=" ",
    floatmode="fixed",
)


def _repr_generator(*args: tuple[str, Any]):
    prefix = "- "
    k, _ = zip(*args)
    width = max(map(len, k))
    for key, value in args:
        key = f"{prefix}{key.rjust(width)}: "
        if isinstance(value, np.ndarray):
            value = _array2string(value, prefix=key)

        else:
            value = repr(value)
        yield f"{key}{value}"


def join_kv(head: tuple[Hashable, Any] | str | type, *args: tuple[Hashable, Any], sep="\n") -> str:
    if isinstance(head, tuple):
        args = (head, *args)
        head = ""
    elif isinstance(head, type):
        head = f"{head.__name__}:"

    text = sep.join(_repr_generator(*((str(k), v) for k, v in args)))
    return sep.join([head, text])


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
max_len: Final[Callable[[Iterable[Sized]], int]] = lambda x: max(map(len, x))


@overload
def find(__func: Callable[[_T1], bool], __x: Iterable[_T1]) -> _T1:
    ...


@overload
def find(__func: Callable[[_T1], bool], __x: Iterable[_T1], /, *, default: _T2) -> _T1 | _T2:
    ...


def find(__func: Callable[[_T1], bool], __x: Iterable[_T1], /, *, default: _T2 | NoDefault = NO_DEFAULT) -> _T1 | _T2:
    try:
        return next(filter(__func, __x))
    except StopIteration as e:
        if default is not NO_DEFAULT:
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


# =====================================================================================================================
# - mapping utils
# =====================================================================================================================


def nested_proxy(data: Mapping[str, Any]) -> types.MappingProxyType[str, Any]:
    return types.MappingProxyType({k: nested_proxy(v) if isinstance(v, Mapping) else v for k, v in data.items()})


dtype_map = types.MappingProxyType(
    collections.defaultdict(
        lambda: pd.api.types.pandas_dtype("object"),
        {
            str: pd.api.types.pandas_dtype("string[pyarrow]"),
            datetime.datetime: pd.api.types.pandas_dtype("datetime64[ns]"),
            datetime.date: pd.api.types.pandas_dtype("datetime64[ns]"),
            datetime.time: pd.api.types.pandas_dtype("datetime64[ns]"),
            datetime.timedelta: pd.api.types.pandas_dtype("timedelta64[ns]"),
        }
        | {x: pd.api.types.pandas_dtype(x) for x in (int, float, bool, complex, bytes, memoryview)},
    )
)


def get_enum_dtype(x: enum.Enum | type[enum.Enum]) -> DtypeObj:
    """Resolve the dtype of an enum.

    Args:
        x (type[enum.Enum]): The enum to resolve.

    Returns:
        DtypeObj: The dtype of the enum.
    """
    if not isinstance(x, type):
        x = type(x)
    keys_ = dtype_map.keys()
    if type_ := find(lambda t: t in keys_, x.mro(), default=None):
        return dtype_map[type_]

    return pd.api.types.pandas_dtype("category")


def get_pandas_dtype(x) -> DtypeObj:
    if isinstance(x, enum.Enum) or (isinstance(x, type) and issubclass(x, enum.Enum)):
        return get_enum_dtype(x)

    return pd.api.types.pandas_dtype(x)


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
