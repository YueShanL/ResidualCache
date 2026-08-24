from __future__ import annotations

from typing import Hashable, Iterable, Mapping, Sequence

from .contracts import ClusterSelection, KVRecordPayload, LayerKVView, RecordRef


class KVPayloadStore:
    """Sidecar payload registry keyed by stable layer-local record identity."""

    def __init__(self) -> None:
        self._payloads: dict[RecordRef, KVRecordPayload] = {}

    def register(self, record: RecordRef, payload: KVRecordPayload) -> None:
        if record in self._payloads:
            raise ValueError(f"K/V payload is already registered for {record!r}")
        self._payloads[record] = payload

    def get(self, record: RecordRef) -> KVRecordPayload:
        try:
            return self._payloads[record]
        except KeyError as error:
            raise KeyError(f"no K/V payload is registered for {record!r}") from error

    def retain(self, active_records: Iterable[RecordRef]) -> None:
        active = set(active_records)
        self._payloads = {
            record: payload
            for record, payload in self._payloads.items()
            if record in active
        }

    def __len__(self) -> int:
        return len(self._payloads)


class LayerKVCacheBuilder:
    """Pack selected cluster records into one variable-length K/V per layer."""

    def __init__(self, payload_store: KVPayloadStore) -> None:
        self.payload_store = payload_store

    def build(
        self,
        records_by_layer: Mapping[int, Sequence[RecordRef]],
        *,
        selected_clusters: Mapping[int, Sequence[ClusterSelection]] | None = None,
        device=None,
    ) -> dict[int, LayerKVView]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - project runtime has torch.
            raise RuntimeError("dynamic K/V packing requires torch") from error

        views: dict[int, LayerKVView] = {}
        for layer, record_refs in sorted(records_by_layer.items()):
            tokens: list[tuple[int, RecordRef, object, object]] = []
            expected_shape: tuple[int, int, int] | None = None
            expected_dtype = None
            for record in record_refs:
                if record.layer != layer:
                    raise ValueError("record is stored under the wrong layer")
                payload = self.payload_store.get(record)
                key = payload.key
                value = payload.value
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                    raise TypeError("K/V payloads must be torch tensors")
                if key.ndim != 4 or value.shape != key.shape:
                    raise ValueError("K/V tensors must have shape [batch, heads, tokens, dim]")
                if key.shape[2] != len(payload.logical_positions):
                    raise ValueError("K/V token count does not match logical positions")
                shape = (int(key.shape[0]), int(key.shape[1]), int(key.shape[3]))
                if expected_shape is None:
                    expected_shape = shape
                    expected_dtype = (key.dtype, value.dtype)
                elif shape != expected_shape:
                    raise ValueError("layer K/V payload shapes are incompatible")
                elif (key.dtype, value.dtype) != expected_dtype:
                    raise ValueError("layer K/V payload dtypes are incompatible")
                for token_index, logical_position in enumerate(payload.logical_positions):
                    tokens.append(
                        (
                            logical_position,
                            record,
                            key[:, :, token_index : token_index + 1, :],
                            value[:, :, token_index : token_index + 1, :],
                        )
                    )
            if not tokens:
                continue
            tokens.sort(key=lambda item: item[0])
            positions = tuple(item[0] for item in tokens)
            if len(set(positions)) != len(positions):
                raise ValueError("selected layer memory contains duplicate logical positions")
            keys = [item[2].to(device) if device is not None else item[2] for item in tokens]
            values = [item[3].to(device) if device is not None else item[3] for item in tokens]
            cluster_ids: tuple[Hashable, ...] = ()
            if selected_clusters is not None:
                cluster_ids = tuple(
                    selection.cluster_id for selection in selected_clusters.get(layer, ())
                )
            views[layer] = LayerKVView(
                layer=layer,
                key=torch.cat(keys, dim=2),
                value=torch.cat(values, dim=2),
                logical_positions=positions,
                record_refs=tuple(item[1] for item in tokens),
                selected_cluster_ids=cluster_ids,
            )
        return views
