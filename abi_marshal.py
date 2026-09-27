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

返回值沿用同一套分类/对齐规则：小结果（<=16 字节的非空标量或聚合）
逐块放入调用结果寄存器；大结果（MEMORY，>16 字节）由调用方提供一块
自然对齐的缓冲区，其地址作为隐式指针实参占用**第一个参数位置**，随后
的显式实参依次后移；空记录没有返回值，不占任何资源。本模块只模拟该
合成调用模型，不对接真实硬件：缓冲区由 CallImage 自带的 result_buffer
字节串表示，地址是合成地址空间中的确定性常量。

公开接口：类型构造（IntType/FloatType/PtrType/ArrayType/RecordType 及常量）、
layout()、classify()、encode()、decode()、pack_call()、unpack_call()、
pack_callsite()、write_result()、unpack_result()。

错误语义：所有非法输入抛出 ABIError 子类，并尽量携带可定位的 path
（如 ``arg[2].samples[3]``）。不做静默纠正：整数溢出、bool 冒充整数、
NaN/Infinity、字段缺失/多余、数组长度不符、循环类型一律拒绝。
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace as dataclasses_replace

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
    "INT8", "INT16", "INT32", "INT64",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "DOUBLE", "PTR",
    # 布局 / 分类
    "Layout", "FieldLayout", "layout", "classify",
    # 编组
    "Placement", "CallImage",
    "ResultPlacement", "RESULT_BUFFER_BASE",
    "encode", "decode", "pack_call", "unpack_call",
    "pack_callsite", "write_result", "unpack_result",
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

# 合成调用模型中调用方结果缓冲区所在的合成地址空间基址。仅用于在调用
# 映像中以 8 字节无符号指针表示缓冲区地址，不对应任何真实内存。
RESULT_BUFFER_BASE = 0x7F00_0000_0000

_INT_SIZES = (1, 2, 4, 8)


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

    if isinstance(t, RecordType):
        if id(t) in _visiting:
            raise RecursiveTypeError(
                f"recursive record type {t.name!r} has infinite layout")
        fields = t.fields  # 未定义时抛 InvalidTypeError
        visiting = _visiting | {id(t)}
        offset = 0
        align = 1
        field_layouts: list[FieldLayout] = []
        for fname, ftype in fields:
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
        expected = {f.name for f in lay.fields}
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
            f.name: decode(f.type, data[f.offset:f.offset + f.layout.size],
                           f.layout, f"{path}.{f.name}")
            for f in lay.fields
        }
    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


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
    sret: bool = False  # 大返回值缓冲区地址的隐式首参


@dataclass(frozen=True)
class ResultPlacement:
    """返回值的安放规则（在调用点即确定，与返回值本身无关）。

    kind: ``"registers"`` / ``"memory"`` / ``"none"``（空记录）。
    memory 时 address 是调用方缓冲区的合成地址，align 是缓冲区对齐；
    registers 时 int_regs/float_regs 是结果占用的寄存器槽下标（从 0 起）。
    """

    kind: str
    eightbytes: tuple[str, ...]
    int_regs: tuple[int, ...] = ()
    float_regs: tuple[int, ...] = ()
    address: int | None = None
    size: int = 0
    align: int = 1


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
    # 以下字段仅在 pack_callsite 产出的调用点映像中出现；pack_call 产物中
    # result_placement 为 None、result_buffer 为空、arg_base 为 0。
    result_placement: ResultPlacement | None = None
    result_buffer: bytes = b""
    arg_base: int = 0  # 隐式 sret 指针实参的个数（0 或 1）


def _validate_arg_list(args) -> None:
    if not isinstance(args, (list, tuple)):
        raise InvalidValueError(
            f"args must be a list/tuple of (Type, value) pairs, got {args!r}")
    if len(args) > MAX_ARGS:
        raise InvalidValueError(
            f"too many arguments: {len(args)} > limit {MAX_ARGS}")


def _pack_image(args, return_type) -> CallImage:
    """编组核心：可选地预置大返回值的隐式缓冲区指针首参。"""
    _validate_arg_list(args)

    result_placement: ResultPlacement | None = None
    result_buffer = b""
    arg_base = 0
    packed_args: list = list(args)

    if return_type is not None:
        _check_type(return_type)
        rlay = layout(return_type)
        rcls = classify(return_type)
        if not rcls:
            result_placement = ResultPlacement("none", (), (), (), None, 0,
                                               rlay.align)
        elif rcls == (MEMORY,):
            # 调用方提供一块自然对齐的结果空间；地址占用第一个参数位置，
            # 随后的显式实参统一后移一位（arg_base=1）。
            result_buffer = b"\x00" * rlay.size
            result_placement = ResultPlacement(
                "memory", rcls, (), (), RESULT_BUFFER_BASE, rlay.size,
                rlay.align)
            packed_args = [(PTR, RESULT_BUFFER_BASE)] + packed_args
            arg_base = 1
        else:
            # 小结果总是从 0 号结果寄存器起安放（与实参寄存器池共享编号，
            # 但在返回点单独观察），<=16 字节必然放得下。
            ri = tuple(range(rcls.count(INTEGER)))
            rf = tuple(range(rcls.count(FLOAT)))
            result_placement = ResultPlacement(
                "registers", rcls, ri, rf, None, rlay.size, rlay.align)

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

    for i, item in enumerate(packed_args):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise InvalidValueError(
                f"each argument must be a (Type, value) pair, got {item!r}",
                f"arg[{i}]")
        t, value = item
        is_hidden_sret = arg_base == 1 and i == 0
        try:
            _check_type(t)
            lay = layout(t)
        except ABIError as exc:
            raise type(exc)(exc.message, f"arg[{i}]") from None
        sret_path = "__sret__" if is_hidden_sret else f"arg[{i - arg_base}]"
        data = encode(t, value, lay, path=sret_path)
        cls = classify(t)

        if not cls:
            placements.append(Placement("none", (), (), (), None, 0,
                                        is_hidden_sret))
            continue
        if cls == (MEMORY,):
            pl = to_stack(data, cls, lay.size)
            placements.append(
                Placement(pl.kind, pl.eightbytes, pl.int_regs, pl.float_regs,
                          pl.stack_offset, pl.size, is_hidden_sret))
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
                          None, lay.size, is_hidden_sret))
        else:
            # 寄存器不足：整体走栈，不消耗任何寄存器。
            pl = to_stack(data, cls, lay.size)
            placements.append(
                Placement(pl.kind, pl.eightbytes, pl.int_regs, pl.float_regs,
                          pl.stack_offset, pl.size, is_hidden_sret))

    stack.extend(b"\x00" * (-len(stack) % STACK_ALIGN))
    return CallImage(tuple(int_regs), tuple(float_regs), bytes(stack),
                     tuple(placements), result_placement,
                     bytes(result_buffer), arg_base)


def pack_call(args) -> CallImage:
    """把 ``[(Type, value), ...]`` 编组为寄存器 + 栈映像（无返回值槽）。

    逐参数：size 为 0 的聚合不占资源；超过 16 字节走栈；否则按块申请
    寄存器，寄存器不足时整体走 16 字节对齐的栈且不消耗寄存器。
    任一参数非法时整体抛错，不返回部分结果。
    """
    return _pack_image(args, None)


def pack_callsite(return_type, args) -> CallImage:
    """调用点编组：在参数映像之外再安排返回值的接收位置。

    返回值按与实参相同的规则分类：

    * 空记录：无返回值，不占寄存器/缓冲区，显式实参位置不变；
    * <=16 字节的标量或聚合：逐块进入从 0 起的结果寄存器（记录于
      ``image.result_placement``），显式实参位置不变；
    * >16 字节（MEMORY）：调用方提供一块按该类型自然对齐的缓冲区，
      其 8 字节地址作为隐式指针实参占据第一个参数位置，其后的每个
      显式实参顺延一个寄存器/栈位置（``arg_base=1``）。

    缓冲区是合成模型自带的 ``result_buffer`` 字节串（初值全零），其
    地址是合成地址空间常量，不涉及真实硬件内存。
    """
    return _pack_image(args, return_type)


def unpack_call(types, image: CallImage) -> list:
    """按 ``types`` 从 ``image`` 重建显式实参值列表。

    若映像由 ``pack_callsite`` 为大返回值预置了隐式 sret 指针首参
    （``arg_base=1``），该槽会被自动跳过，调用方传入的 ``types`` 只列
    显式实参。注意：寄存器映像反映的是进入函数时的现场，若随后用
    ``write_result`` 覆写了结果寄存器，应先解出参数再写返回值。
    """
    if not isinstance(image, CallImage):
        raise InvalidValueError(f"expected a CallImage, got {image!r}")
    if not isinstance(types, (list, tuple)):
        raise InvalidValueError(
            f"types must be a list/tuple of Type, got {types!r}")
    visible = image.placements[image.arg_base:]
    if len(types) != len(visible):
        raise InvalidValueError(
            f"{len(types)} types but image has {len(visible)} visible "
            f"placements (arg_base={image.arg_base})")

    values = []
    for i, (t, pl) in enumerate(zip(types, visible)):
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


def write_result(return_type, image: CallImage, value) -> CallImage:
    """被调方侧：按调用点映像的安放规则写入返回值，返回返回点新映像。

    * 小结果（registers）：按 8 字节块写入 0 号起的整数/浮点结果寄存器，
      其余寄存器槽保持进入时的值（合成模型不模拟调用方保存寄存器）；
    * 大结果（memory）：把按 ``return_type`` 编码（含嵌套记录/数组的
      对齐与尾填充）的完整字节写入调用方缓冲区 ``result_buffer``，
      寄存器与栈映像不变；
    * 空记录（none）：映像原样返回，值必须是与空记录匹配的 ``{}``。

    值校验失败时整体抛错，返回的映像不存在部分写入。
    """
    if not isinstance(image, CallImage):
        raise InvalidValueError(f"expected a CallImage, got {image!r}")
    rp = image.result_placement
    if rp is None:
        raise InvalidValueError(
            "image has no result placement; use pack_callsite() to model "
            "a return value")
    _check_type(return_type)
    lay = layout(return_type)
    if classify(return_type) != rp.eightbytes:
        # 安放规则必须与调用点声明的返回类型逐块一致（含 () 空记录）。
        raise InvalidTypeError(
            f"return type {return_type.name!r} does not match the callsite "
            f"result placement {rp.eightbytes}")
    if lay.size != rp.size:
        raise InvalidTypeError(
            f"return type {return_type.name!r} size {lay.size} does not "
            f"match the callsite result size {rp.size}")
    data = encode(return_type, value, lay, path="result")

    if rp.kind == "none":
        return image

    if rp.kind == "memory":
        if len(data) != rp.size:
            raise InvalidTypeError(
                f"encoded result size {len(data)} != buffer size {rp.size}")
        return dataclasses_replace(image, result_buffer=bytes(data))

    if rp.kind == "registers":
        int_regs = list(image.int_regs)
        float_regs = list(image.float_regs)
        int_iter = iter(rp.int_regs)
        float_iter = iter(rp.float_regs)
        for k, c in enumerate(rp.eightbytes):
            chunk = data[8 * k:8 * k + 8].ljust(8, b"\x00")
            if c == INTEGER:
                int_regs[next(int_iter)] = int.from_bytes(chunk, "little")
            else:
                float_regs[next(float_iter)] = struct.unpack("<d", chunk)[0]
        return dataclasses_replace(image, int_regs=tuple(int_regs),
                                   float_regs=tuple(float_regs))

    raise InvalidTypeError(f"unknown result placement kind {rp.kind!r}")


def unpack_result(return_type, image: CallImage):
    """调用方侧：按同一套布局/对齐规则从返回点映像重建返回值。

    registers 从结果 8 字节块拼回（裁掉最后一块的寄存器零填充，记录
    内部的对齐填充字节重建为零），memory 从 ``result_buffer`` 整体解码。
    返回类型是数组时得到 ``list``，记录得到 ``dict``。
    """
    if not isinstance(image, CallImage):
        raise InvalidValueError(f"expected a CallImage, got {image!r}")
    rp = image.result_placement
    if rp is None:
        raise InvalidValueError(
            "image has no result placement; use pack_callsite()")
    _check_type(return_type)
    lay = layout(return_type)

    if rp.kind == "none":
        if lay.size != 0:
            raise InvalidTypeError(
                f"non-empty return type {return_type.name!r} cannot be read "
                "from a 'none' placement")
        return decode(return_type, b"", lay, "result")

    if rp.kind == "memory":
        if rp.eightbytes != classify(return_type):
            raise InvalidTypeError(
                f"return type {return_type.name!r} does not match the "
                "memory result placement")
        if len(image.result_buffer) != rp.size:
            raise InvalidValueError(
                f"result buffer length {len(image.result_buffer)} != "
                f"expected {rp.size}")
        return decode(return_type, image.result_buffer, lay, "result")

    if rp.kind == "registers":
        if rp.eightbytes != classify(return_type):
            raise InvalidTypeError(
                f"return type {return_type.name!r} does not match the "
                f"register result placement {rp.eightbytes}")
        chunks: list[bytes] = []
        int_iter = iter(rp.int_regs)
        float_iter = iter(rp.float_regs)
        for c in rp.eightbytes:
            if c == INTEGER:
                idx = next(int_iter)
                v = image.int_regs[idx]
                if not isinstance(v, int) or isinstance(v, bool):
                    raise InvalidValueError(
                        f"integer result register {idx} not populated: {v!r}",
                        "result")
                chunks.append(v.to_bytes(8, "little"))
            else:
                idx = next(float_iter)
                v = image.float_regs[idx]
                if not isinstance(v, float):
                    raise InvalidValueError(
                        f"float result register {idx} not populated: {v!r}",
                        "result")
                chunks.append(struct.pack("<d", v))
        data = b"".join(chunks)[:lay.size]
        return decode(return_type, data, lay, "result")

    raise InvalidTypeError(f"unknown result placement kind {rp.kind!r}")
