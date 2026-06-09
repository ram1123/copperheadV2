import os
import uuid
import awkward as ak
from coffea.processor import ProcessorABC

from src.stage1.runner_processor import write_cutflow_outputs



class AkArray:
    def __init__(self, arr=None):
        self.arr = ak.Array([]) if arr is None else arr

    def __iadd__(self, other):
        if len(self.arr) == 0:
            self.arr = other.arr
        elif len(other.arr) == 0:
            pass
        else:
            self.arr = ak.concatenate([self.arr, other.arr])
        return self

    def __add__(self, other):
        result = AkArray(self.arr)
        result += other
        return result

    def identity(self):
        return AkArray()


class CopperheadRunnerAdapter(ProcessorABC):
    """
    Adapter for coffea.processor.Runner.
    Stores only JSON-serialisable config — no EventProcessor instance,
    no awkward behaviors — so dask can pickle this class cleanly.
    EventProcessor is reconstructed fresh on each worker call.
    """

    def __init__(
        self,
        config,
        dataset_yaml_file,
        save_path,
        test_mode=False,
        isCutflow=False,
    ):
        # Store only plain picklable data — NO EventProcessor instance
        self._config = self._normalize_config(config)
        self._dataset_yaml_file = dataset_yaml_file
        self._save_path = save_path
        self._test_mode = test_mode
        self._isCutflow = isCutflow

    @staticmethod
    def _normalize_config(config):
        """
        Accept either the plain config dict or an EventProcessor-like object
        carrying a `.config` dict.
        """
        if isinstance(config, dict):
            return dict(config)

        cfg = getattr(config, "config", None)
        if isinstance(cfg, dict):
            return dict(cfg)

        raise TypeError(
            "CopperheadRunnerAdapter expected a config dict or an object with "
            f"a dict-like .config attribute, got {type(config).__name__}"
        )

    def process(self, events):
        # Reconstruct EventProcessor fresh on the worker — avoids pickling behaviors
        from src.copperhead_processor import EventProcessor

        processor = EventProcessor(
            self._config,
            test_mode=self._test_mode,
            isCutflow=self._isCutflow,
        )

        out_collections, n_processed = processor.process(
            events, dataset_yaml_file=self._dataset_yaml_file
        )

        os.makedirs(self._save_path, exist_ok=True)

        out_collections["fraction"] = (
            events.metadata["fraction"] * ak.ones_like(out_collections["event"])
        )
        skim_zip = ak.zip(out_collections, depth_limit=1)

        shard_id = uuid.uuid4().hex
        parquet_path = os.path.join(self._save_path, f"part_{shard_id}.parquet")
        ak.to_parquet(skim_zip, parquet_path)

        if self._isCutflow and hasattr(processor, "cutflow"):
            write_cutflow_outputs(
                processor.cutflow,
                self._save_path,
                events.metadata["dataset"],
                shard_id,
            )

        result = {
            "__n_processed__": AkArray(ak.Array([int(n_processed)])),
            "__written_files__": AkArray(ak.Array([1])),
        }
        return result

    def postprocess(self, accumulator):
        return accumulator
