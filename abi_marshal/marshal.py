"""Call marshalling: register/stack planning, packing and unpacking.

Register file (fixed ABI contract):
  * 4 integer registers, 2 float registers, each 8 bytes wide
  * scalar integers/pointers take one integer register, scalar floats
    one float register
  * aggregates classified as eightbytes take one register per eightbyte
    (INTEGER -> integer register, FLOAT -> float register)
  * if an aggregate's registers are not all available, the WHOLE
    aggregate goes to a 16-byte-aligned stack slot and consumes NO
    registers; the freed registers stay available for later arguments
  * MEMORY aggregates (>16 bytes) always go to a 16-byte-aligned slot
  * spilled scalars go to 8-byte-aligned stack slots
  * size-0 aggregates consume neither registers nor stack

Registers and stack carry raw little-endian bytes; padding is zero.
pack_call is all-or-nothing: on any validation error it raises and no
partial image is returned (the function is pure, nothing is mutated).
"""

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass

from .errors import (
    ABIError,
    ArityError,
    ImageError,
    IntegerOverflowError,
    NonFiniteFloatError,
    ValueTypeError,
)
from .layout import FLOAT, INTEGER, MEMORY, align_up, classify, layout
from .model import ArrayType, FloatType, IntType, PtrType, RecordType, Type

INT_REG_COUNT = 4
FLOAT_REG_COUNT = 2
STACK_SLOT_ALIGN = 16  # aggregates on the stack
STACK_SCALAR_ALIGN = 8  # spilled scalars on the stack
REG_WIDTH = 8


@dataclass(frozen=True)
class ArgPlacement:
    """Where one argument's 8-byte chunks landed.

    ``chunks`` holds one ``(where, index)`` entry per 8-byte chunk of the
    encoded argument: ``("int", i)`` / ``("float", i)`` for registers,
    ``("stack", offset)`` for stack bytes.
    """

    index: int
    type: object
    size: int
    chunks: tuple


@dataclass(frozen=True)
class CallPlan:
    placements: tuple
    stack_size: int


@dataclass(frozen=True)
class CallImage:
    """The marshalled form of one call.

    ``int_regs``/``float_regs`` are tuples of 4/2 entries, each either
    ``None`` (register unused) or 8 raw bytes.  ``stack`` is the stack
    image; ``placements`` describes each argument's location.
    """

    int_regs: tuple
    float_regs: tuple
    stack: bytes
    placements: tuple


def _scalar_class(typ):
    if isinstance(typ, (IntType, PtrType)):
        return INTEGER
    if isinstance(typ, FloatType):
        return FLOAT
    return None


def plan_call(types):
    """Compute the register/stack placement for a call signature."""
    types = _check_types(types)
    int_used = 0
    float_used = 0
    stack_len = 0
    placements = []
    for i, typ in enumerate(types):
        lay = layout(typ)
        scalar = _scalar_class(typ)
        if scalar is not None:
            classes = (scalar,)
            stack_align = STACK_SCALAR_ALIGN
        else:
            classes = classify(typ)
            stack_align = STACK_SLOT_ALIGN
        if classes is not MEMORY and classes:
            need_int = classes.count(INTEGER)
            need_float = classes.count(FLOAT)
            if (
                int_used + need_int <= INT_REG_COUNT
                and float_used + need_float <= FLOAT_REG_COUNT
            ):
                chunks = []
                for cls in classes:
                    if cls == INTEGER:
                        chunks.append(("int", int_used))
                        int_used += 1
                    else:
                        chunks.append(("float", float_used))
                        float_used += 1
                placements.append(ArgPlacement(i, typ, lay.size, tuple(chunks)))
                continue
        # Stack path: MEMORY aggregates, register shortfall (whole
        # aggregate, no registers consumed), and spilled scalars.
        if lay.size == 0:
            placements.append(ArgPlacement(i, typ, 0, ()))
            continue
        offset = align_up(stack_len, stack_align)
        nchunks = (lay.size + 7) // 8
        chunks = tuple(("stack", offset + 8 * k) for k in range(nchunks))
        stack_len = offset + 8 * nchunks
        placements.append(ArgPlacement(i, typ, lay.size, chunks))
    return CallPlan(tuple(placements), stack_len)


def _check_types(types):
    if not isinstance(types, (list, tuple)):
        raise ValueTypeError(f"types must be a list/tuple, got {type(types).__name__}")
    types = tuple(types)
    for i, t in enumerate(types):
        if not isinstance(t, Type):
            raise ValueTypeError(f"not an ABI type: {t!r}", (f"arg{i}",))
    return types


def _check_values(values, ntypes):
    if not isinstance(values, (list, tuple)):
        raise ValueTypeError(f"values must be a list/tuple, got {type(values).__name__}")
    if len(values) != ntypes:
        raise ArityError(f"call expects {ntypes} arguments, got {len(values)}")
    return tuple(values)


# ---------------------------------------------------------------------------
# value encoding / decoding


def _int_range(typ):
    return typ.min_value, typ.max_value


def encode_value(typ, value, path=()):
    """Encode one Python value to exactly ``layout(typ).size`` bytes."""
    if isinstance(typ, IntType):
        if type(value) is bool or not isinstance(value, int):
            raise ValueTypeError(
                f"expected int for {typ.name}, got {type(value).__name__}", path
            )
        lo, hi = _int_range(typ)
        if not lo <= value <= hi:
            raise IntegerOverflowError(
                f"value {value} out of range [{lo}, {hi}] for {typ.name}", path
            )
        return value.to_bytes(typ.size, "little", signed=typ.signed)
    if isinstance(typ, PtrType):
        if type(value) is bool or not isinstance(value, int):
            raise ValueTypeError(
                f"expected int for pointer, got {type(value).__name__}", path
            )
        if not 0 <= value < (1 << 64):
            raise IntegerOverflowError(
                f"pointer value {value} out of range [0, 2**64)", path
            )
        return value.to_bytes(8, "little")
    if isinstance(typ, FloatType):
        if type(value) is bool or not isinstance(value, (int, float)):
            raise ValueTypeError(
                f"expected int or float for f64, got {type(value).__name__}", path
            )
        if isinstance(value, int):
            try:
                f = float(value)
            except OverflowError:
                raise IntegerOverflowError(
                    f"integer {value} too large to represent as f64", path
                ) from None
            if not math.isfinite(f) or int(f) != value:
                raise IntegerOverflowError(
                    f"integer {value} is not exactly representable as f64", path
                )
        else:
            f = value
        if not math.isfinite(f):
            raise NonFiniteFloatError(f"non-finite float {f!r} is not representable", path)
        return struct.pack("<d", f)
    if isinstance(typ, RecordType):
        if not isinstance(value, Mapping):
            raise ValueTypeError(
                f"record {typ.name!r} expects a mapping, got {type(value).__name__}", path
            )
        lay = layout(typ)
        names = {f.name for f in lay.fields}
        missing = [f.name for f in lay.fields if f.name not in value]
        extra = [k for k in value if k not in names]
        if missing or extra:
            raise ValueTypeError(
                f"record {typ.name!r}: missing fields {missing}, unexpected fields {extra}",
                path,
            )
        buf = bytearray(lay.size)
        for f in lay.fields:
            chunk = encode_value(f.type, value[f.name], path + (f.name,))
            buf[f.offset : f.offset + len(chunk)] = chunk
        return bytes(buf)
    if isinstance(typ, ArrayType):
        if not isinstance(value, (list, tuple)):
            raise ValueTypeError(
                f"array expects a list/tuple, got {type(value).__name__}", path
            )
        if len(value) != typ.count:
            raise ArityError(
                f"array expects exactly {typ.count} elements, got {len(value)}", path
            )
        el_lay = layout(typ.element)
        buf = bytearray(el_lay.size * typ.count)
        for i, item in enumerate(value):
            chunk = encode_value(typ.element, item, path + (i,))
            off = i * el_lay.size
            buf[off : off + len(chunk)] = chunk
        return bytes(buf)
    raise ABIError(f"not an ABI type: {typ!r}", path)  # pragma: no cover


def decode_value(typ, buf, off=0, path=()):
    """Decode one value from ``buf`` at ``off``; inverse of encode_value."""
    if isinstance(typ, IntType):
        return int.from_bytes(buf[off : off + typ.size], "little", signed=typ.signed)
    if isinstance(typ, PtrType):
        return int.from_bytes(buf[off : off + 8], "little")
    if isinstance(typ, FloatType):
        return struct.unpack_from("<d", buf, off)[0]
    if isinstance(typ, RecordType):
        return {
            f.name: decode_value(f.type, buf, off + f.offset, path + (f.name,))
            for f in layout(typ).fields
        }
    if isinstance(typ, ArrayType):
        el_lay = layout(typ.element)
        return [
            decode_value(typ.element, buf, off + i * el_lay.size, path + (i,))
            for i in range(typ.count)
        ]
    raise ABIError(f"not an ABI type: {typ!r}", path)  # pragma: no cover


# ---------------------------------------------------------------------------
# pack / unpack


def pack_call(types, values):
    """Marshal one call.  Returns a CallImage; raises on any bad input."""
    types = _check_types(types)
    values = _check_values(values, len(types))
    plan = plan_call(types)
    int_regs = [None] * INT_REG_COUNT
    float_regs = [None] * FLOAT_REG_COUNT
    stack = bytearray(plan.stack_size)
    for placement, typ, val in zip(plan.placements, types, values):
        data = encode_value(typ, val, (f"arg{placement.index}",))
        padded = data + b"\x00" * (-len(data) % REG_WIDTH)
        for ci, (where, idx) in enumerate(placement.chunks):
            chunk = padded[ci * REG_WIDTH : (ci + 1) * REG_WIDTH]
            if where == "int":
                int_regs[idx] = chunk
            elif where == "float":
                float_regs[idx] = chunk
            else:
                stack[idx : idx + REG_WIDTH] = chunk
    return CallImage(tuple(int_regs), tuple(float_regs), bytes(stack), plan.placements)


def _check_reg(regs, expected, label):
    if not isinstance(regs, (list, tuple)) or len(regs) != expected:
        raise ImageError(
            f"{label} must be a sequence of {expected} entries, got "
            f"{None if not isinstance(regs, (list, tuple)) else len(regs)}"
        )
    for i, reg in enumerate(regs):
        if reg is not None and not (isinstance(reg, (bytes, bytearray)) and len(reg) == REG_WIDTH):
            raise ImageError(f"{label}[{i}] must be None or {REG_WIDTH} bytes")


def unpack_call(types, image):
    """Rebuild the argument values from a CallImage produced by pack_call.

    The placement plan is recomputed from ``types`` alone; the image must
    match it exactly (register count, 8-byte registers, exact stack size).
    """
    types = _check_types(types)
    if not isinstance(image, CallImage):
        raise ImageError(f"expected a CallImage, got {type(image).__name__}")
    plan = plan_call(types)
    _check_reg(image.int_regs, INT_REG_COUNT, "int_regs")
    _check_reg(image.float_regs, FLOAT_REG_COUNT, "float_regs")
    if not isinstance(image.stack, (bytes, bytearray)):
        raise ImageError("stack must be bytes")
    if len(image.stack) != plan.stack_size:
        raise ImageError(
            f"stack image is {len(image.stack)} bytes, expected {plan.stack_size}"
        )
    values = []
    for placement, typ in zip(plan.placements, types):
        padded = bytearray(REG_WIDTH * len(placement.chunks))
        for ci, (where, idx) in enumerate(placement.chunks):
            if where == "int":
                chunk = image.int_regs[idx]
                if chunk is None:
                    raise ImageError(f"integer register {idx} is empty", (f"arg{placement.index}",))
            elif where == "float":
                chunk = image.float_regs[idx]
                if chunk is None:
                    raise ImageError(f"float register {idx} is empty", (f"arg{placement.index}",))
            else:
                chunk = image.stack[idx : idx + REG_WIDTH]
            padded[ci * REG_WIDTH : (ci + 1) * REG_WIDTH] = chunk
        values.append(decode_value(typ, bytes(padded), 0, (f"arg{placement.index}",)))
    return values
