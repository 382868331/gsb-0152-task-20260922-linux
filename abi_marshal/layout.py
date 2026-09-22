"""Byte layout and eightbyte classification.

Layout rules (fixed ABI contract):
  * scalar alignment == its size (1/2/4/8), so alignment never exceeds 8
  * record fields sit at the next offset that is a multiple of their
    alignment; the record's alignment is the max field alignment; its
    size is the end offset rounded up to that alignment (tail padding)
  * an empty record has size 0 and alignment 1
  * arrays lay elements contiguously at the full element size (stride ==
    element size, which is already a multiple of the element alignment)

Classification rules:
  * aggregates of size > 16 bytes are MEMORY (passed on the stack)
  * otherwise the aggregate is split into 8-byte eightbytes; an
    eightbyte is INTEGER if any integer/pointer leaf overlaps it,
    otherwise FLOAT
  * size-0 aggregates classify to the empty tuple: no registers, no stack
"""

from dataclasses import dataclass

from .errors import CycleError, SizeLimitError, TypeDefinitionError
from .model import ArrayType, FloatType, IntType, PtrType, RecordType

#: Maximum size in bytes of any laid-out type (interface bound).
MAX_TYPE_SIZE = 1 << 24  # 16 MiB

# Eightbyte classes.
INTEGER = "integer"
FLOAT = "float"
#: classify() result for aggregates larger than 16 bytes.
MEMORY = "memory"


def align_up(n, align):
    """Round ``n`` up to a multiple of ``align`` (a power of two)."""
    return (n + align - 1) // align * align


@dataclass(frozen=True)
class FieldLayout:
    name: str
    offset: int
    type: object


@dataclass(frozen=True)
class Layout:
    size: int
    align: int
    #: records only: tuple of FieldLayout in declaration order
    fields: tuple = ()
    #: arrays only: element type and count (stride == element size)
    element: object = None
    count: object = None


def layout(typ):
    """Compute the Layout of an ABI type.  Raises CycleError on cycles."""
    return _layout(typ, {}, (), ())


def _check_size(size, path):
    if size > MAX_TYPE_SIZE:
        raise SizeLimitError(
            f"type size {size} bytes exceeds the {MAX_TYPE_SIZE}-byte limit", path
        )


def _layout(typ, memo, stack, path):
    key = id(typ)
    hit = memo.get(key)
    if hit is not None:
        return hit

    if isinstance(typ, IntType):
        lay = Layout(typ.size, typ.size)
    elif isinstance(typ, FloatType):
        lay = Layout(8, 8)
    elif isinstance(typ, PtrType):
        lay = Layout(8, 8)
    elif isinstance(typ, ArrayType):
        el = _layout(typ.element, memo, stack, path + ("[]",))
        size = el.size * typ.count
        _check_size(size, path)
        lay = Layout(size, el.align, element=typ.element, count=typ.count)
    elif isinstance(typ, RecordType):
        if key in stack:
            raise CycleError(f"record {typ.name!r} contains itself by value", path)
        stack = stack + (key,)
        offset = 0
        max_align = 1
        fields = []
        for fname, ftype in typ.fields:
            fl = _layout(ftype, memo, stack, path + (fname,))
            offset = align_up(offset, fl.align)
            fields.append(FieldLayout(fname, offset, ftype))
            offset += fl.size
            _check_size(offset, path + (fname,))
            if fl.align > max_align:
                max_align = fl.align
        if fields:
            lay = Layout(align_up(offset, max_align), max_align, fields=tuple(fields))
        else:
            # Empty record: size 0, alignment 1, consumes nothing.
            lay = Layout(0, 1, fields=())
    else:
        raise TypeDefinitionError(f"not an ABI type: {typ!r}", path)

    memo[key] = lay
    return lay


def _collect_leaves(typ, lay, base, out):
    """Append (offset, size, class) for every scalar leaf under ``typ``."""
    if isinstance(typ, (IntType, PtrType)):
        out.append((base, lay.size, INTEGER))
    elif isinstance(typ, FloatType):
        out.append((base, 8, FLOAT))
    elif isinstance(typ, RecordType):
        for f in lay.fields:
            _collect_leaves(f.type, layout(f.type), base + f.offset, out)
    elif isinstance(typ, ArrayType):
        el_lay = layout(typ.element)
        if el_lay.size:
            for i in range(typ.count):
                _collect_leaves(typ.element, el_lay, base + i * el_lay.size, out)
    else:  # pragma: no cover - layout() already rejected non-types
        raise TypeDefinitionError(f"not an ABI type: {typ!r}")


def classify(typ):
    """Classify an aggregate (or scalar) into eightbyte classes.

    Returns the string ``MEMORY`` for types larger than 16 bytes, the
    empty tuple for size-0 types, otherwise a tuple of INTEGER/FLOAT
    with one entry per 8-byte eightbyte.
    """
    lay = layout(typ)
    if lay.size == 0:
        return ()
    if lay.size > 16:
        return MEMORY
    leaves = []
    _collect_leaves(typ, lay, 0, leaves)
    chunks = []
    for c in range((lay.size + 7) // 8):
        lo, hi = c * 8, min(c * 8 + 8, lay.size)
        kinds = {k for off, sz, k in leaves if off < hi and off + sz > lo}
        # An eightbyte covered only by padding defaults to INTEGER.
        chunks.append(FLOAT if kinds == {FLOAT} else INTEGER)
    return tuple(chunks)
