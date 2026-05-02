from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Sequence,
)

import torch

from diffusionrl.utils.batched import (
    Batched,
    FieldKind,
    _concat_value,
    _field_kind,
    _slice_value,
    concat_field,
)

if TYPE_CHECKING:
    from transfer_queue import BatchMeta


@dataclass
class TqMeta(Batched):
    data: torch.Tensor | list = concat_field()
    batch_meta: "BatchMeta" = concat_field()
    _shape: torch.Size = concat_field()
    _data_key: str = None

    @classmethod
    def concat(cls: "TqMeta", items: Sequence["TqMeta"]):
        from transfer_queue import BatchMeta

        if not items:
            raise ValueError(f"Cannot concat empty sequence of {cls.__name__}")
        if len(items) == 1:
            return items[0]
        assert all(items[0]._data_key == item._data_key for item in items)

        batch_sizes = [item.batch_size for item in items]
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(items[0]):  # type: ignore[arg-type]
            values = [getattr(item, f.name) for item in items]
            if _field_kind(f) is FieldKind.CONCAT:
                if all(isinstance(v, BatchMeta) for v in values):
                    kwargs[f.name] = BatchMeta.concat(values)
                elif all(isinstance(v, torch.Size) for v in values):
                    assert all(s[1:] == values[0][1:] for s in values)
                    kwargs[f.name] = torch.Size([sum(s[0] for s in values), *values[0][1:]])
                else:
                    kwargs[f.name] = _concat_value(values, batch_sizes)
            else:
                kwargs[f.name] = values[0]
        return cls(**kwargs)

    @property
    def shape(self) -> list[int]:
        return self._shape

    @property
    def batch_size(self) -> int:
        return self._shape[0]

    def slice(self: "TqMeta", start: int, end: int):
        from transfer_queue import BatchMeta

        bs = self.batch_size
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if _field_kind(f) is FieldKind.CONCAT:
                if isinstance(val, BatchMeta):
                    kwargs[f.name] = val.select_samples(list(range(start, end)))
                elif isinstance(val, torch.Size):
                    kwargs[f.name] = torch.Size([end - start, *val[1:]])
                else:
                    kwargs[f.name] = _slice_value(val, start, end, bs)
            else:
                kwargs[f.name] = val
        return type(self)(**kwargs)

    def reset_data(self):
        self.data = None

    def reset_batch_meta(self):
        self.batch_meta = None
