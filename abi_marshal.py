"""abi_marshal — 聚合参数的合成 ABI 编组（离线外部函数接口工程）。

本模块实现且仅实现以下明示 ABI 契约（不宣称兼容任何真实平台 ABI）：

* 小端字节序；
* 指针 8 字节（无符号）；整数 1/2/4/8 字节（有符号或无符号）；浮点 8 字节
  （IEEE-754 binary64）；
* 记录（record）按自然对齐布局，对齐上限 8 字节；空记录 size=0、align=1；
* 数组为显式长度的同类元素序列，元素按完整 size 连续排列；
* 聚合按 8 字节块（eightbyte）分类：块内含任意整数/指针字节则该块为
  INTEGER 类，否则为 FLOAT 类；聚合超过 16 字节整体走栈（MEMORY）；
* 整数寄存器 4 个、浮点寄存器 2 个；一个聚合所需寄存器不足时，整体走
  16 字节对齐的栈，且不消耗任何寄存器；
* 空记录不消耗寄存器也不消耗栈。
* 记录可含无符号位域字段：底层存储单元仅 8/16/32 位无符号整数，小端、
  低位先分配；字段不横跨单元，放不下即开新单元；零宽字段结束当前单元，
  下一字段重新对齐。未用位与结构体填充一律清零。

公开接口：类型构造（IntType/FloatType/PtrType/ArrayType/RecordType 及常量）、
layout()、classify()、encode()、decode()、pack_call()、unpack_call()。

错误语义：所有非法输入抛出 ABIError 子类，并尽量携带可定位的 path
（如 ``arg[2].samples[3]``）。不做静默纠正：整数溢出、bool 冒充整数、
NaN/Infinity、字段缺失/多余、数组长度不符、循环类型一律拒绝。
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    # 常量
    "INTEGER", "FLOAT", "MEMORY",
    "INT_REG_COUNT", "FLOAT_REG_COUNT", "STACK_ALIGN", "MAX_EIGHTBYTE_SIZE",
    "MAX_ARGS", "MAX_ARRAY_COUNT", "MAX_RECORD_FIELDS", "MAX_TYPE_SIZE",
    # 错误
    "ABIError", "InvalidTypeError", "RecursiveTypeError",
    "InvalidValueError", "IntegerOverflowError", "NonFiniteFloatError",
    # 类型
    "Type", "IntType", "FloatType", "PtrType", "ArrayType", "RecordType",
    "BitfieldType",
    "INT8", "INT16", "INT32", "INT64",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "DOUBLE", "PTR",
    # 布局 / 分类
    "Layout", "FieldLayout", "layout", "classify",
    # 编组
    "Placement", "CallImage",
    "encode", "decode", "pack_call", "unpack_call",
]

# ---------------------------------------------------------------------------
# ABI 常量与输入上限
# ---------------------------------------------------------------------------

INTEGER = "INTEGER"
FLOAT = "FLOAT"
MEMORY = "MEMORY"

INT_REG_COUNT = 4
FLOAT_REG_COUNT = 2
STACK_ALIGN = 16
MAX_EIGHTBYTE_SIZE = 16  # 聚合超过 16 字节整体走栈

MAX_ARGS = 1024                 # 单次调用最多参数个数
MAX_ARRAY_COUNT = 65536         # 数组最大元素数
MAX_RECORD_FIELDS = 256         # 记录最大字段数
MAX_TYPE_SIZE = 1 << 20         # 任一类型完整布局最大字节数（1 MiB）

_INT_SIZES = (1, 2, 4, 8)
_BITFIELD_UNIT_SIZES = (1, 2, 4)  # 位域存储单元仅 8/16/32 位


# ---------------------------------------------------------------------------
# 错误层级
# ---------------------------------------------------------------------------

class ABIError(Exception):
    """本模块所有错误的基类。``path`` 给出可定位的出错位置。"""

    def __init__(self, message: str, path: str | None = None):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


class InvalidTypeError(ABIError):
    """类型定义本身非法（未定义的记录、字段重复、尺寸超限等）。"""


class RecursiveTypeError(InvalidTypeError):
    """记录类型直接/间接包含自身，布局无限大，拒绝。"""


class InvalidValueError(ABIError):
    """值与类型不匹配（类型错误、长度不符、字段缺失/多余等）。"""


class IntegerOverflowError(InvalidValueError):
    """整数/指针值超出目标类型的可表示范围。"""


class NonFiniteFloatError(InvalidValueError):
    """浮点值为 NaN 或 Infinity，拒绝。"""


# ---------------------------------------------------------------------------
# 类型描述
# ---------------------------------------------------------------------------

class Type:
    """所有 ABI 类型描述的基类。"""

    __slots__ = ()

    @property
    def name(self) -> str:  # pragma: no cover - 子类覆盖
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class IntType(Type):
    """1/2/4/8 字节整数。"""

    size: int
    signed: bool

    def __post_init__(self):
        if self.size not in _INT_SIZES:
            raise InvalidTypeError(
                f"integer size must be one of {_INT_SIZES}, got {self.size!r}")
        if not isinstance(self.signed, bool):
            raise InvalidTypeError(f"signed must be bool, got {self.signed!r}")

    @property
    def name(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.size * 8}"

    @property
    def range(self) -> tuple[int, int]:
        bits = self.size * 8
        if self.signed:
            return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        return (0, (1 << bits) - 1)


@dataclass(frozen=True)
class FloatType(Type):
    """8 字节 IEEE-754 binary64 浮点。"""

    @property
    def name(self) -> str:
        return "f64"


@dataclass(frozen=True)
class PtrType(Type):
    """8 字节无符号指针。"""

    @property
    def name(self) -> str:
        return "ptr"


@dataclass(frozen=True)
class ArrayType(Type):
    """显式长度的同类元素序列；元素按完整 size 连续排列。"""

    elem: Type
    count: int

    def __post_init__(self):
        _check_type(self.elem)
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise InvalidTypeError(
                f"array count must be an int, got {self.count!r}")
        if not 0 <= self.count <= MAX_ARRAY_COUNT:
            raise InvalidTypeError(
                f"array count {self.count} out of range [0, {MAX_ARRAY_COUNT}]")

    @property
    def name(self) -> str:
        return f"{self.elem.name}[{self.count}]"


@dataclass(frozen=True)
class BitfieldType(Type):
    """无符号位域字段：底层存储单元为 8/16/32 位无符号整数。

    ``width`` 为占用位数；``width == 0`` 表示零宽字段（结束当前存储单元，
    下一字段重新对齐），不持有值，也不出现在记录值字典中。
    """

    base: IntType
    width: int

    def __post_init__(self):
        if not isinstance(self.base, IntType):
            raise InvalidTypeError(
                f"bitfield base must be an IntType, got {self.base!r}")
        if self.base.signed:
            raise InvalidTypeError("bitfield base must be unsigned")
        if self.base.size not in _BITFIELD_UNIT_SIZES:
            raise InvalidTypeError(
                f"bitfield unit size must be one of {_BITFIELD_UNIT_SIZES} "
                f"bytes, got {self.base.size}")
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise InvalidTypeError(
                f"bitfield width must be an int, got {self.width!r}")
        if not 0 <= self.width <= self.base.size * 8:
            raise InvalidTypeError(
                f"bitfield width {self.width} out of range "
                f"[0, {self.base.size * 8}]")

    @property
    def name(self) -> str:
        return f"u{self.base.size * 8}:{self.width}"

    @property
    def range(self) -> tuple[int, int]:
        return (0, (1 << self.width) - 1)


class RecordType(Type):
    """命名字段记录。支持先建壳后 ``define()``，以便表达（并被拒绝的）循环。"""

    __slots__ = ("_name", "_fields")

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise InvalidTypeError(f"record name must be a non-empty str, got {name!r}")
        self._name = name
        self._fields: tuple[tuple[str, Type], ...] | None = None

    @property
    def name(self) -> str:
        return self._name

    def define(self, fields) -> None:
        """一次性定义字段序列 ``[(field_name, Type), ...]``。只能定义一次。"""
        if self._fields is not None:
            raise InvalidTypeError(f"record {self._name!r} is already defined")
        items = tuple(fields)
        if len(items) > MAX_RECORD_FIELDS:
            raise InvalidTypeError(
                f"record {self._name!r} has {len(items)} fields, "
                f"limit is {MAX_RECORD_FIELDS}")
        seen: set[str] = set()
        for item in items:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise InvalidTypeError(
                    f"record {self._name!r}: field must be a (name, Type) pair, "
                    f"got {item!r}")
            fname, ftype = item
            if not isinstance(fname, str) or not fname:
                raise InvalidTypeError(
                    f"record {self._name!r}: field name must be a non-empty str, "
                    f"got {fname!r}")
            if fname in seen:
                raise InvalidTypeError(
                    f"record {self._name!r}: duplicate field name {fname!r}")
            seen.add(fname)
            _check_type(ftype)
        self._fields = items

    @property
    def fields(self) -> tuple[tuple[str, Type], ...]:
        if self._fields is None:
            raise InvalidTypeError(f"record {self._name!r} is not defined yet")
        return self._fields


INT8 = IntType(1, True)
INT16 = IntType(2, True)
INT32 = IntType(4, True)
INT64 = IntType(8, True)
UINT8 = IntType(1, False)
UINT16 = IntType(2, False)
UINT32 = IntType(4, False)
UINT64 = IntType(8, False)
DOUBLE = FloatType()
PTR = PtrType()


def _check_type(t) -> None:
    if not isinstance(t, Type):
        raise InvalidTypeError(f"expected an ABI Type, got {t!r}")


def _align_up(offset: int, align: int) -> int:
    return (offset + align - 1) // align * align


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldLayout:
    """记录中一个字段的布局。"""

    name: str
    offset: int
    type: Type
    layout: "Layout"
    # 位域字段专用：单元内位偏移与（已就位的）掩码；非位域为 None
    bit_offset: int | None = None
    mask: int | None = None


@dataclass(frozen=True)
class Layout:
    """一个类型的完整布局：size、align，以及（记录的）逐字段布局。"""

    type: Type
    size: int
    align: int
    fields: tuple[FieldLayout, ...] = ()


def layout(t: Type, _visiting: frozenset[int] | None = None) -> Layout:
    """计算类型 ``t`` 的字节布局。循环记录类型抛 RecursiveTypeError。"""
    _check_type(t)
    if _visiting is None:
        _visiting = frozenset()

    if isinstance(t, IntType):
        return Layout(t, t.size, t.size)
    if isinstance(t, FloatType):
        return Layout(t, 8, 8)
    if isinstance(t, PtrType):
        return Layout(t, 8, 8)

    if isinstance(t, ArrayType):
        elem_layout = layout(t.elem, _visiting)
        size = elem_layout.size * t.count
        if size > MAX_TYPE_SIZE:
            raise InvalidTypeError(
                f"type size {size} exceeds limit {MAX_TYPE_SIZE}")
        align = elem_layout.align if t.count > 0 else 1
        return Layout(t, size, align)

    if isinstance(t, BitfieldType):
        return Layout(t, t.base.size, t.base.size)

    if isinstance(t, RecordType):
        if id(t) in _visiting:
            raise RecursiveTypeError(
                f"recursive record type {t.name!r} has infinite layout")
        fields = t.fields  # 未定义时抛 InvalidTypeError
        visiting = _visiting | {id(t)}
        offset = 0
        align = 1
        field_layouts: list[FieldLayout] = []
        unit_open = False
        open_unit_bits = 0
        unit_start = 0
        unit_used = 0
        for fname, ftype in fields:
            if isinstance(ftype, BitfieldType):
                flay = layout(ftype, visiting)
                unit_bits = ftype.base.size * 8
                if ftype.width == 0:
                    # 零宽字段：结束当前单元，下一字段重新对齐
                    unit_open = False
                    field_layouts.append(
                        FieldLayout(fname, offset, ftype, flay))
                    continue
                if (not unit_open or open_unit_bits != unit_bits
                        or unit_used + ftype.width > unit_bits):
                    # 当前单元放不下（或无单元）：对齐后开新单元
                    offset = _align_up(offset, ftype.base.size)
                    unit_open = True
                    open_unit_bits = unit_bits
                    unit_start = offset
                    unit_used = 0
                    offset += ftype.base.size
                align = max(align, flay.align)
                mask = ((1 << ftype.width) - 1) << unit_used
                field_layouts.append(
                    FieldLayout(fname, unit_start, ftype, flay,
                                unit_used, mask))
                unit_used += ftype.width
                continue
            unit_open = False
            flay = layout(ftype, visiting)
            align = max(align, flay.align)
            offset = _align_up(offset, flay.align)
            field_layouts.append(FieldLayout(fname, offset, ftype, flay))
            offset += flay.size
        size = _align_up(offset, align)
        if size > MAX_TYPE_SIZE:
            raise InvalidTypeError(
                f"type size {size} exceeds limit {MAX_TYPE_SIZE}")
        return Layout(t, size, align, tuple(field_layouts))

    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# 聚合分类（按 8 字节块）
# ---------------------------------------------------------------------------

def _spans(t: Type, lay: Layout, base: int, out: list[tuple[int, int, str]]) -> None:
    """把类型的所有标量叶子展开为 (偏移, 长度, 类) 区间。"""
    if isinstance(t, (IntType, PtrType)):
        out.append((base, lay.size, INTEGER))
    elif isinstance(t, BitfieldType):
        if t.width > 0:
            out.append((base, lay.size, INTEGER))
    elif isinstance(t, FloatType):
        out.append((base, lay.size, FLOAT))
    elif isinstance(t, ArrayType):
        elem_layout = layout(t.elem)
        step = elem_layout.size
        for i in range(t.count):
            _spans(t.elem, elem_layout, base + i * step, out)
    elif isinstance(t, RecordType):
        for f in lay.fields:
            _spans(f.type, f.layout, base + f.offset, out)
    else:  # pragma: no cover
        raise InvalidTypeError(f"unsupported type {t!r}")


def classify(t: Type) -> tuple[str, ...]:
    """按 8 字节块分类聚合。

    返回逐块的 ``INTEGER``/``FLOAT`` 元组；size 为 0 返回空元组；
    超过 16 字节返回 ``(MEMORY,)``。块内含任意整数/指针字节即为 INTEGER。
    """
    lay = layout(t)
    if lay.size == 0:
        return ()
    if lay.size > MAX_EIGHTBYTE_SIZE:
        return (MEMORY,)
    spans: list[tuple[int, int, str]] = []
    _spans(t, lay, 0, spans)
    classes: list[str] = []
    for chunk_start in range(0, lay.size, 8):
        cls = FLOAT
        for off, size, c in spans:
            if c == INTEGER and off < chunk_start + 8 and off + size > chunk_start:
                cls = INTEGER
                break
        classes.append(cls)
    return tuple(classes)


# ---------------------------------------------------------------------------
# 值编码 / 解码（类型化字节序列）
# ---------------------------------------------------------------------------

def _encode_int(value, lay: Layout, signed: bool, path: str) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidValueError(
            f"expected an int (bool is not accepted), got {value!r}", path)
    lo, hi = lay.type.range
    if not lo <= value <= hi:
        raise IntegerOverflowError(
            f"value {value} out of range [{lo}, {hi}] for {lay.type.name}", path)
    return value.to_bytes(lay.size, "little", signed=signed)


def encode(t: Type, value, lay: Layout | None = None, path: str = "value") -> bytes:
    """把 Python 值按类型 ``t`` 的布局编码为小端字节串（长度恰为 size）。"""
    _check_type(t)
    if lay is None:
        lay = layout(t)

    if isinstance(t, IntType):
        return _encode_int(value, lay, t.signed, path)

    if isinstance(t, PtrType):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidValueError(
                f"expected an int address (bool is not accepted), got {value!r}",
                path)
        if not 0 <= value <= (1 << 64) - 1:
            raise IntegerOverflowError(
                f"pointer value {value} out of range [0, 2**64 - 1]", path)
        return value.to_bytes(8, "little")

    if isinstance(t, FloatType):
        # 只接受 float：接受 int 会在 |v| >= 2**53 时静默丢精度，拒绝。
        if isinstance(value, bool) or not isinstance(value, float):
            raise InvalidValueError(
                f"expected a float, got {value!r}", path)
        if not math.isfinite(value):
            raise NonFiniteFloatError(
                f"non-finite float {value!r} is rejected", path)
        return struct.pack("<d", value)

    if isinstance(t, ArrayType):
        if not isinstance(value, (list, tuple)):
            raise InvalidValueError(
                f"expected a list/tuple of {t.count} elements, got {value!r}",
                path)
        if len(value) != t.count:
            raise InvalidValueError(
                f"array length {len(value)} != declared count {t.count}", path)
        elem_layout = layout(t.elem)
        return b"".join(
            encode(t.elem, v, elem_layout, f"{path}[{i}]")
            for i, v in enumerate(value)
        )

    if isinstance(t, RecordType):
        if not isinstance(value, Mapping):
            raise InvalidValueError(
                f"expected a mapping of record fields, got {value!r}", path)
        expected = {f.name for f in lay.fields
                    if not (isinstance(f.type, BitfieldType)
                            and f.type.width == 0)}
        actual = set(value.keys())
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise InvalidValueError(
                f"record {t.name!r} missing fields {sorted(missing)}", path)
        if extra:
            raise InvalidValueError(
                f"record {t.name!r} has unexpected fields {sorted(extra)}", path)
        buf = bytearray(lay.size)
        for f in lay.fields:
            if isinstance(f.type, BitfieldType):
                if f.type.width == 0:
                    continue
                v = value[f.name]
                if isinstance(v, bool) or not isinstance(v, int):
                    raise InvalidValueError(
                        f"expected an int (bool is not accepted), got {v!r}",
                        f"{path}.{f.name}")
                lo, hi = f.type.range
                if not lo <= v <= hi:
                    raise IntegerOverflowError(
                        f"value {v} out of range [{lo}, {hi}] for "
                        f"bitfield {f.type.name}", f"{path}.{f.name}")
                # 只写本字段掩码内的位，未用位与相邻字段不受影响
                unit_size = f.layout.size
                unit = int.from_bytes(
                    buf[f.offset:f.offset + unit_size], 'little')
                unit |= (v << f.bit_offset) & f.mask
                buf[f.offset:f.offset + unit_size] = (
                    unit.to_bytes(unit_size, 'little'))
                continue
            chunk = encode(f.type, value[f.name], f.layout, f"{path}.{f.name}")
            buf[f.offset:f.offset + f.layout.size] = chunk
        return bytes(buf)

    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def decode(t: Type, data: bytes, lay: Layout | None = None, path: str = "value"):
    """把 ``encode`` 的产物按类型 ``t`` 解码回 Python 值。"""
    _check_type(t)
    if lay is None:
        lay = layout(t)
    if not isinstance(data, (bytes, bytearray)):
        raise InvalidValueError(f"expected bytes, got {data!r}", path)
    if len(data) != lay.size:
        raise InvalidValueError(
            f"byte length {len(data)} != layout size {lay.size} for {t.name}",
            path)

    if isinstance(t, IntType):
        return int.from_bytes(data, "little", signed=t.signed)
    if isinstance(t, PtrType):
        return int.from_bytes(data, "little")
    if isinstance(t, FloatType):
        return struct.unpack("<d", data)[0]
    if isinstance(t, ArrayType):
        elem_layout = layout(t.elem)
        step = elem_layout.size
        return [
            decode(t.elem, data[i * step:(i + 1) * step], elem_layout,
                   f"{path}[{i}]")
            for i in range(t.count)
        ]
    if isinstance(t, RecordType):
        return {
            f.name: _decode_field(f, data, path)
            for f in lay.fields
            if not (isinstance(f.type, BitfieldType) and f.type.width == 0)
        }
    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def _decode_field(f: FieldLayout, data: bytes, path: str):
    """解码记录的一个字段；位域按位偏移/掩码从存储单元中取出。"""
    if isinstance(f.type, BitfieldType):
        unit = int.from_bytes(
            data[f.offset:f.offset + f.layout.size], 'little')
        return (unit & f.mask) >> f.bit_offset
    return decode(f.type, data[f.offset:f.offset + f.layout.size],
                  f.layout, f"{path}.{f.name}")


# ---------------------------------------------------------------------------
# 调用编组：寄存器分配 + 栈映像
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """一个实参的安放结果。

    kind: ``"registers"`` / ``"stack"`` / ``"none"``（空聚合，不占任何资源）。
    """

    kind: str
    eightbytes: tuple[str, ...]
    int_regs: tuple[int, ...] = ()
    float_regs: tuple[int, ...] = ()
    stack_offset: int | None = None
    size: int = 0


@dataclass(frozen=True)
class CallImage:
    """一次调用的完整编组映像。

    int_regs: 长度 4，未用槽位为 None，已用槽位为该 8 字节块的小端整数值。
    float_regs: 长度 2，未用槽位为 None，已用槽位为该 8 字节块解释的 double。
    stack: 16 字节对齐的栈映像字节串。
    placements: 与实参一一对应的安放结果。
    """

    int_regs: tuple
    float_regs: tuple
    stack: bytes
    placements: tuple[Placement, ...]


def pack_call(args) -> CallImage:
    """把 ``[(Type, value), ...]`` 编组为寄存器 + 栈映像。

    逐参数：size 为 0 的聚合不占资源；超过 16 字节走栈；否则按块申请
    寄存器，寄存器不足时整体走 16 字节对齐的栈且不消耗寄存器。
    任一参数非法时整体抛错，不返回部分结果。
    """
    if not isinstance(args, (list, tuple)):
        raise InvalidValueError(
            f"args must be a list/tuple of (Type, value) pairs, got {args!r}")
    if len(args) > MAX_ARGS:
        raise InvalidValueError(
            f"too many arguments: {len(args)} > limit {MAX_ARGS}")

    int_regs: list = [None] * INT_REG_COUNT
    float_regs: list = [None] * FLOAT_REG_COUNT
    stack = bytearray()
    placements: list[Placement] = []
    next_int = 0
    next_float = 0

    def to_stack(data: bytes, cls: tuple[str, ...], size: int) -> Placement:
        off = _align_up(len(stack), STACK_ALIGN)
        stack.extend(b"\x00" * (off - len(stack)))
        stack.extend(data)
        return Placement("stack", cls, (), (), off, size)

    for i, item in enumerate(args):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise InvalidValueError(
                f"each argument must be a (Type, value) pair, got {item!r}",
                f"arg[{i}]")
        t, value = item
        try:
            _check_type(t)
            lay = layout(t)
        except ABIError as exc:
            raise type(exc)(exc.message, f"arg[{i}]") from None
        data = encode(t, value, lay, path=f"arg[{i}]")
        cls = classify(t)

        if not cls:
            placements.append(Placement("none", (), (), (), None, 0))
            continue
        if cls == (MEMORY,):
            placements.append(to_stack(data, cls, lay.size))
            continue

        need_int = cls.count(INTEGER)
        need_float = cls.count(FLOAT)
        if (next_int + need_int <= INT_REG_COUNT
                and next_float + need_float <= FLOAT_REG_COUNT):
            iregs: list[int] = []
            fregs: list[int] = []
            for k, c in enumerate(cls):
                chunk = data[8 * k:8 * k + 8].ljust(8, b"\x00")
                if c == INTEGER:
                    int_regs[next_int] = int.from_bytes(chunk, "little")
                    iregs.append(next_int)
                    next_int += 1
                else:
                    float_regs[next_float] = struct.unpack("<d", chunk)[0]
                    fregs.append(next_float)
                    next_float += 1
            placements.append(
                Placement("registers", cls, tuple(iregs), tuple(fregs),
                          None, lay.size))
        else:
            # 寄存器不足：整体走栈，不消耗任何寄存器。
            placements.append(to_stack(data, cls, lay.size))

    stack.extend(b"\x00" * (-len(stack) % STACK_ALIGN))
    return CallImage(tuple(int_regs), tuple(float_regs), bytes(stack),
                     tuple(placements))


def unpack_call(types, image: CallImage) -> list:
    """按 ``types`` 从 ``image`` 重建实参值列表（``pack_call`` 的逆）。"""
    if not isinstance(image, CallImage):
        raise InvalidValueError(f"expected a CallImage, got {image!r}")
    if not isinstance(types, (list, tuple)):
        raise InvalidValueError(
            f"types must be a list/tuple of Type, got {types!r}")
    if len(types) != len(image.placements):
        raise InvalidValueError(
            f"{len(types)} types but image has {len(image.placements)} "
            "placements")

    values = []
    for i, (t, pl) in enumerate(zip(types, image.placements)):
        _check_type(t)
        lay = layout(t)
        path = f"arg[{i}]"
        if pl.kind == "none":
            data = b""
        elif pl.kind == "registers":
            chunks: list[bytes] = []
            int_iter = iter(pl.int_regs)
            float_iter = iter(pl.float_regs)
            for c in pl.eightbytes:
                if c == INTEGER:
                    v = image.int_regs[next(int_iter)]
                    chunks.append(v.to_bytes(8, "little"))
                else:
                    chunks.append(struct.pack("<d", image.float_regs[next(float_iter)]))
            data = b"".join(chunks)[:lay.size]
        elif pl.kind == "stack":
            data = image.stack[pl.stack_offset:pl.stack_offset + pl.size]
            if len(data) != pl.size:
                raise InvalidValueError(
                    f"stack image truncated: need {pl.size} bytes at offset "
                    f"{pl.stack_offset}, have {len(data)}", path)
        else:
            raise InvalidValueError(f"unknown placement kind {pl.kind!r}", path)
        values.append(decode(t, data, lay, path))
    return values
