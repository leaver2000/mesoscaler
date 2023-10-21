"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

__all__ = ["auto_field", "VariableEnum", "IndependentVariables", "DependentVariables"]
import collections
import enum
import types
from typing import MutableMapping, NamedTuple, TypeAlias

import numpy as np
import pandas as pd
import pyproj

from ._typing import (
    Any,
    AnyArrayLike,
    Callable,
    Generic,
    Hashable,
    HashableT,
    Iterable,
    Mapping,
    MutableMapping,
    NamedTuple,
    NumpyDType_T,
    PandasDType_T,
    TypeAlias,
    TypeVar,
    overload,
)

_ENUM_DICT_RESERVED_KEYS = (
    "__doc__",
    "__module__",
    "__qualname__",
    "_order_",
    "_create_pseudo_member_",
    "_generate_next_value_",
    "_missing_",
    "_ignore_",
)

CLASS_LOC = "__mesometa_loc__"
CLASS_METADATA = "__mesometa_cls_data__"
MEMBER_METADATA = "__mesometa_member_data__"
MEMBER_ALIASES = "__mesometa_member_aliases__"
MEMBER_SERIES = "__mesometa_series__"

_T = TypeVar("_T")
_Item: TypeAlias = np.generic | bool | int | float | complex | str | bytes | memoryview | enum.Enum | Hashable
MemberMetadata: TypeAlias = MutableMapping[str, Any]


# =====================================================================================================================
#
# =====================================================================================================================
def auto_field(value: _T | Any = None, *, aliases: list[_T] | None = None, **metadata: Any) -> Any:
    """
    A factory function that creates a new field with the given value and metadata. The EnumMetaCls
    will unpack all of the metadata from the field and store it in the class metadata. Only the
    value is included in the enum_dict.

    Args:
        value: The value of the field. If None, an auto-generated value will be used.
        aliases: A list of aliases for the field.
        **metadata: Additional metadata to attach to the field.

    Returns:
        A new field object with the given value and metadata.
    """

    if value is None:
        value = enum.auto()
    if MEMBER_ALIASES in metadata and aliases is None:
        assert isinstance(metadata[MEMBER_ALIASES], list)
    elif MEMBER_ALIASES in metadata and aliases is not None:
        raise ValueError("Field metadata contains aliases and aliases were passed as an argument.")
    else:
        metadata[MEMBER_ALIASES] = aliases or []

    return _Field(value, metadata)


def get_metadata() -> list[dict[str, Any]]:
    metadata = getattr(_EnumMetaCls.__metadata__, "_data")  # type: Mapping[int, types.MappingProxyType[str, Any]]
    items = ((key, dict(value)) for key, value in metadata.items())
    return [
        {
            "hash": key,
            CLASS_METADATA: dict(value[CLASS_METADATA]),
            MEMBER_SERIES: dict(value[MEMBER_SERIES]),
            MEMBER_METADATA: dict(value[MEMBER_METADATA]),
            MEMBER_ALIASES: value[MEMBER_ALIASES].to_dict(orient="records"),
        }
        for key, value in items
    ]


class _Field(NamedTuple):
    """Temporary container for field metadata."""

    value: Any
    metadata: Mapping[str, Any]


def _unpack_info(old: enum._EnumDict) -> tuple[enum._EnumDict, dict[str, Any]]:
    """Unpacks the enum_dict into a new dict and a metadata dict."""
    new = enum._EnumDict()
    new._cls_name = old._cls_name  # type: ignore
    meta = {}  # type: dict[str, Any]
    for key, value in old.items():
        if isinstance(value, _Field):
            new[key], meta[key] = value
        else:
            new[key] = value
            if key not in _ENUM_DICT_RESERVED_KEYS:
                meta[key] = {}

    return new, meta


def _repack_info(
    name: str,
    member_map: dict[str, enum.Enum],
    metadata: dict[str, dict[str, Any]],
    class_metadata: dict[str, Any],
) -> types.MappingProxyType[str, Any]:
    index = pd.Index(list(member_map.keys()), name="member_names", dtype="string[pyarrow]")  # type: pd.Index[str]
    aliases = pd.DataFrame.from_dict(
        {name: list(set(metadata[name].pop(MEMBER_ALIASES, []))) for name in index}, orient="index"
    ).T

    member_metadata = types.MappingProxyType(collections.defaultdict(dict, metadata))
    member_series = pd.Series(
        list(member_map.values()),
        index=index,
        dtype=pd.CategoricalDtype(),
        name=name,
    )
    return types.MappingProxyType(
        {
            "name": name,
            CLASS_METADATA: class_metadata,
            CLASS_LOC: _Loc(list, member_series),
            MEMBER_METADATA: member_metadata,
            MEMBER_ALIASES: aliases,
            MEMBER_SERIES: member_series,
        }
    )


# =====================================================================================================================
#
# =====================================================================================================================
_AnyArrayLikeT = TypeVar("_AnyArrayLikeT", bound=AnyArrayLike)


# TODO: this need work
class _Loc(Generic[_AnyArrayLikeT]):
    def __init__(
        self,
        hook: Callable[[AnyArrayLike[NumpyDType_T, PandasDType_T]], _AnyArrayLikeT],
        data: AnyArrayLike[NumpyDType_T, PandasDType_T],
    ) -> None:
        self._hook = hook
        self._data = data

    @overload
    def __getitem__(self, item: list) -> PandasDType_T | NumpyDType_T:  # pyright: ignore
        ...

    @overload
    def __getitem__(self, item: Any) -> _AnyArrayLikeT:
        ...

    def __getitem__(self, item: Any) -> _AnyArrayLikeT | PandasDType_T | NumpyDType_T:  # type: ignore
        from .utils import is_array_like

        x = self._data[item]
        return self._hook(x) if is_array_like(x) else x


class _MetaDataDescriptor:
    __getitem__: Any
    _data: collections.defaultdict[int, Mapping[str, Any]]

    def __init__(self) -> None:
        self._data = collections.defaultdict(dict, {})

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._data[hash(instance)]

    def __set__(self, instance, value):
        if instance is None:
            raise TypeError("Cannot set a class attribute.")
        self._data[hash(instance)] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"


class _EnumMetaCls(enum.EnumMeta):
    __metadata__ = _MetaDataDescriptor()

    def __new__(cls, name: str, bases: tuple[Any, ...], cls_dict: enum._EnumDict, **kwargs: Any) -> _EnumMetaCls:
        cls_dict, member_metadata = _unpack_info(cls_dict)
        obj = super().__new__(cls, name, bases, cls_dict)
        if obj._member_names_:
            obj.__metadata__ = _repack_info(name, obj._member_map_, member_metadata, kwargs)

        return obj

    def __repr__(cls) -> str:
        from .utils import join_kv

        return join_kv(cls, *cls._member_map_.items())

    # =================================================================================================================
    @property
    def metadata(cls) -> MutableMapping[str, Any]:
        return cls.__metadata__[CLASS_METADATA]

    @property
    def loc(cls) -> _Loc[list[VariableEnum]]:
        return cls.__metadata__[CLASS_LOC]

    @property
    def _series(cls) -> pd.Series[Any]:
        return cls.__metadata__[MEMBER_SERIES]

    @property
    def _names(cls) -> pd.Index[str]:
        return cls._series.index

    @property
    def _member_metadata(cls) -> types.MappingProxyType[str, MemberMetadata]:
        return cls.__metadata__[MEMBER_METADATA]

    @property
    def _aliases(cls) -> pd.DataFrame:
        return cls.__metadata__[MEMBER_ALIASES]

    # =================================================================================================================
    # - metadata properties
    def to_frame(cls) -> pd.DataFrame:
        df = cls._aliases.copy()
        return df

    def to_series(cls) -> pd.Series[Any]:
        return pd.Series(cls._member_map_, name=cls.__name__)

    # =================================================================================================================
    def __call__(cls, item: Iterable[_Item] | _Item) -> Any | list[Any]:  # type: ignore[override]
        """It is possible to return multiple members if the members share an alias."""
        from .utils import is_scalar

        if is_scalar(item):
            return cls._scalar_lookup(item)
        return cls.loc[cls.is_in(item)]  # type: ignore

    def _scalar_lookup(cls, item: _Item, /) -> Any:
        s = cls._series
        if isinstance(item, str) and (member := s.get(item, None)):
            return member
        elif (mask := s == item).any():
            return s[mask].item()

        return s[cls.is_in(item)].item()

    def to_list(cls, /) -> list[Any]:
        return cls._series.to_list()

    def is_in(cls, item: _Item | Iterable[_Item], /) -> pd.Series[bool]:
        if isinstance(item, str) or not isinstance(item, Iterable):
            item = [item]

        items = list([item] if type(item) in cls.mro() else item)

        return cls._series.isin(items) | cls._names.isin(items) | cls._aliases.isin(items).any(axis=0, skipna=True)

    def difference(cls, item: _Item | Iterable[_Item], /) -> set[Any]:
        return set(cls).difference(cls(item))

    def intersection(cls, item: _Item | Iterable[_Item], /) -> set[Any]:
        return set(cls).intersection(cls(item))

    def remap(cls, item: Iterable[HashableT], /):
        return {x: cls(x) for x in item}

    def add_alias(cls, col: str, alias: list[str]) -> None:
        df = cls._aliases
        if col not in df.columns:
            raise ValueError(f"Column {col} not in aliases")

        mask = df.stack(dropna=False).isin(alias).unstack().fillna(False).any(axis=0)
        if df.columns[mask].size > 1:  # type: ignore
            raise ValueError(f"Alias {alias} already in use")

        x = list(set(df[col].dropna()).union(alias))
        if len(x) >= df.index.size:
            for i in range(len(x) - df.index.size):
                cls._aliases.loc[len(x) - i, :] = None

        cls._aliases.loc[:, col] = x


class VariableEnum(enum.Enum, metaclass=_EnumMetaCls):
    @property
    def aliases(self) -> list[Any]:
        return self.__class__._aliases[self.name].dropna().to_list()

    @property
    def metadata(self) -> MemberMetadata:
        return self.__class__._member_metadata[self.name]

    def add_aliases(self, *aliases: Any) -> None:
        self.__class__._aliases[self.name].extend(aliases)


class IndependentVariables(str, VariableEnum):
    @staticmethod
    def _generate_next_value_(name: str, *_):
        return name

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


def get_crs(name: str) -> pyproj.CRS:
    # TODO: move this to disk
    if name == "ERA5":
        cf = {
            "crs_wkt": 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
            "geographic_crs_name": "WGS 84",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245179,
            "inverse_flattening": 298.257223563,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "horizontal_datum_name": "World Geodetic System 1984 ensemble",
            "grid_mapping_name": "latitude_longitude",
        }
    elif name == "URMA":
        cf = {
            "geographic_crs_name": "NDFD CONUS 2.5km Lambert Conformal Conic",
            "projected_crs_name": "NDFD",
            "grid_mapping_name": "lambert_conformal_conic",
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31424518,
            "inverse_flattening": 298.25722356301,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "horizontal_datum_name": "WGS84",
            "latitude_of_projection_origin": 20.191999,
            "longitude_of_projection_origin": 238.445999,
            "standard_parallel": 25,
            "false_easting": 0.0,
            "false_northing": 0.0,
            "units": "m",
        }
    else:
        raise ValueError(f"Unknown CRS {name!r}")
    return pyproj.CRS.from_cf(cf)


class DependentVariables(IndependentVariables):
    @classmethod  # type: ignore
    @property
    def crs(cls) -> pyproj.CRS:
        md = cls.metadata  # type: MutableMapping[str, Any] # type: ignore
        if not (crs := md.get("crs", None)):
            crs = md["crs"] = get_crs(cls.__name__)

        return crs
