"""
Workaround for coffea's parquet reader corrupting nullable integer columns.

coffea.nanoevents.mapping.parquet (seen in coffea 2026.5.0) builds the column form from the arrow
type (e.g. int32), but converts an integer column that holds nulls to float64 with NaN; the int form
then reinterprets that float64 buffer, giving values like 1072693248 (= high word of 1.0). In the
compacted stage-1 ntuples every jetN_* integer column (hf strip sizes, jet-ID multiplicities, flavours)
is null when the jet is missing, so anything read through processor.Runner(format="parquet") gets
garbage for them. patch_parquet_nullable_ints() fills those nulls with the dtype's max instead, so the
buffer stays int; callers must drop rows where the object is missing (e.g. jetN_pt is NaN).
Details: docs/known_issues.md (2026-10-31).

How to run: not a script; call patch_parquet_nullable_ints() in the process() of a coffea processor run
with format="parquet" (it must run on the Dask worker; repeated calls are no-ops).
Example:
    from modules.coffea_parquet_utils import patch_parquet_nullable_ints
    def process(self, events):
        patch_parquet_nullable_ints()
        ...
"""

from types import SimpleNamespace

_PATCHED = False


class _PreRead:
    """Source stand-in that returns an already-read table, so the column is read only once."""

    def __init__(self, table):
        self._table = table

    def read(self, column_name):
        return self._table


def patch_parquet_nullable_ints():
    global _PATCHED
    if _PATCHED:
        return
    import awkward
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    from coffea.nanoevents.mapping.parquet import ParquetSourceMapping

    shim = ParquetSourceMapping.UprootLikeShim
    original_array = shim.array

    def array(self, entry_start, entry_stop):
        table = self.source.read(self.column)
        aspa = table[entry_start:entry_stop][0].chunk(0)
        if isinstance(aspa, awkward._connect.pyarrow.AwkwardArrowArray):
            aspa = aspa.storage
        if pa.types.is_integer(aspa.type) and aspa.null_count > 0:
            fill = np.iinfo(aspa.type.to_pandas_dtype()).max
            filled = pc.fill_null(aspa, pa.scalar(fill, type=aspa.type)).to_numpy()
            return awkward.Array(awkward.contents.NumpyArray(filled))
        return original_array(SimpleNamespace(source=_PreRead(table), column=self.column), entry_start, entry_stop)

    shim.array = array
    _PATCHED = True
