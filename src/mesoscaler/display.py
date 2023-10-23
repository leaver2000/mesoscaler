from __future__ import annotations

try:
    import matplotlib.pyplot as _plt
except ImportError:
    _plt = None  # type: ignore
from ._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Mesoscale


class PlotAccessor:
    def __init__(self, scale: Mesoscale) -> None:
        if _plt is None:
            raise ImportError("matplotlib is not installed")
        self.plt = _plt
        self._scale = scale

    def table(self):
        X1, X2, Y = self._scale.dx, self._scale.dy, self._scale.levels
        fig = self.plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(121)
        ax.invert_yaxis()
        ax.plot(X1, Y, linestyle="-", linewidth=0.75, marker=".", markersize=2.5)
        ax.plot(X2, Y, linestyle="-", linewidth=0.75, marker=".", markersize=2.5)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_xlabel("Extent (km)")
        ax.legend(["dx", "dy"])

        df = self._scale.to_pandas()
        ax2 = fig.add_subplot(122)
        ax2.axis("off")
        mpl_table = ax2.table(
            cellText=df.to_numpy().round(2).astype(str).tolist(),
            rowLabels=df.index.tolist(),
            bbox=[0, 0, 1, 1],  # type: ignore
            colLabels=df.columns.tolist(),
        )
        return fig, ax, ax2, mpl_table
