import enum
import types

import pandas as pd
import pyproj

from ._typing import (
    Any,
    ClassVar,
    Generic,
    Hashable,
    HashableT,
    Iterable,
    MutableMapping,
    Self,
    TypeVar,
    overload,
)

_T = TypeVar("_T")
_CLASS_LOC = "__metadata_loc__"
_CLASS_METADATA = "__metadata_cls_data__"
_MEMBER_METADATA = "__metadata_member_data__"
_MEMBER_ALIASES = "__metadata_member_aliases__"
_MEMBER_SERIES = "__metadata_series__"

class _EnumMetaCls(Generic[_T], enum.EnumMeta):
    __metadata__: ClassVar[MutableMapping[str, Any]]
    @property
    def metadata(self) -> MutableMapping[str, Any]: ...
    @property
    def _series(self) -> pd.Series[enum.Enum]: ...  # type: ignore
    @property
    def _names(self) -> pd.Index[str]: ...
    @property
    def _member_metadata(self) -> types.MappingProxyType[str, Any]: ...
    @property
    def _aliases(self) -> pd.DataFrame: ...

class _Loc(Generic[_T]):
    @overload
    def __getitem__(self, item: str) -> _T: ...
    @overload
    def __getitem__(self, item: list[str] | list[bool]) -> list[_T]: ...

class VariableEnum(enum.Enum, metaclass=_EnumMetaCls):
    @classmethod
    @property  # type: ignore
    def loc(cls) -> _Loc[Self]: ...
    @property
    def metadata(self) -> MutableMapping[str, Any]: ...
    @property
    def aliases(self) -> list[Any]: ...
    @overload
    @classmethod
    def __call__(cls, item: Hashable, /) -> Self: ...
    @overload
    @classmethod
    def __call__(cls, item: Iterable[Hashable], /) -> list[Self]: ...
    # - methods[search]
    @classmethod
    def is_in(cls, item: Hashable | Iterable[Hashable], /) -> pd.Series[bool]: ...
    @classmethod
    def difference(cls, item: Hashable | Iterable[Hashable], /) -> set[Self]: ...
    @classmethod
    def intersection(cls, item: Hashable | Iterable[Hashable], /) -> set[Self]: ...
    @classmethod
    def add_alias(cls, name: str, alias: list[Hashable], /, *, sort: bool = False) -> None: ...
    @classmethod
    def remap(cls, item: Iterable[HashableT], /) -> dict[HashableT, Self]: ...
    # - methods[transfer]
    @classmethod
    def to_series(cls) -> pd.Series[Self]: ...  # type: ignore
    @classmethod
    def to_frame(cls) -> pd.DataFrame: ...
    @classmethod
    def to_list(cls) -> list[Self]: ...

class IndependentVariables(str, VariableEnum): ...

class _CRSMixin:
    @classmethod  # type: ignore
    @property
    def crs(cls) -> pyproj.CRS: ...

class DependentVariables(_CRSMixin, IndependentVariables): ...

def auto_field(value: _T | Any = None, /, *, aliases: list[_T] | None = None, **metadata: Any) -> Any: ...
def get_metadata() -> list[dict[str, Any]]: ...
