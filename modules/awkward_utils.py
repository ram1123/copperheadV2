import awkward as ak
import numpy as np

from modules.utils import logger


def ensure_event_axis(collection, n_events: int, label: str = "collection"):
    """
    Restore a missing outer event axis for single-event or empty collections.

    Some Awkward operations on filtered single-event chunks can squeeze the
    outer event dimension, leaving a flat collection that later axis=1
    operations cannot handle. This helper rebuilds the event axis when needed.
    """
    if collection.ndim >= 2:
        return collection

    n_items = len(collection)

    if n_events == 1:
        # Re-add the squeezed length-1 outer axis -> (1 * n_items * record).
        # np.newaxis just wraps the layout; it avoids ak.unflatten's
        # offset-fitting check on some sliced layouts.
        rebuilt = collection[np.newaxis]
    elif n_items == 0:
        rebuilt = ak.unflatten(collection, np.zeros(n_events, dtype="int64"))
    else:
        raise RuntimeError(
            f"Unexpected {label} collapse: ndim={collection.ndim}, "
            f"n_items={n_items}, n_events={n_events}"
        )

    logger.info(
        f"[{label}-norm] restored event axis: ndim {collection.ndim} -> "
        f"{rebuilt.ndim}, n_events={n_events}, n_items={n_items}"
    )
    return rebuilt
