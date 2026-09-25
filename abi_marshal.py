"""abi_marshal — 聚合参数的合成 ABI 编组（离线外部函数接口工程）。

本模块实现且仅实现以下明示 ABI 契约（不宣称兼容任何真实平台 ABI）：

* 小端字节序；
* 指针 8 字节（无符号）；整数 1/2/4/8 字节（有符号或无符号）；浮点 8 字节
  （IEEE-754 binary64）；
* 记录（record）按自然对齐布局，对齐上限 8 字节；空记录 size=0、align=1；
* 数组为显式长度的同类元素序列，元素按完整 size 连续排列；
* 聚合按 8 字节块（eightbyte）分类：块内含任意整数/指针字节则该块为
  INTEGER 类，否则为 FLOAT 类；聚合超过 16 字节整体走栈（MEMORY）；
* 【参数】整数寄存器 4 个、浮点寄存器 2 个；一个聚合所需寄存器不足时，
  整体走 16 字节对齐的栈，且不消耗任何寄存器；
* 【参数】空记录不消耗寄存器也不消耗栈；
* 【返回】标量及不超过 16 字节的聚合沿用同一块分类；整数返回寄存器 2 个、
  浮点返回寄存器 2 个，逐块分配；空记录不占任何返回寄存器；
* 【返回】超过 16 字节的聚合由调用方传入有界返回缓冲区间接返回：被调方把
  完整布局（padding 一律写零）一次性写入该缓冲区；缓冲区的容量、对齐与
  地址范围必须先验证通过，任何失败都不得写入部分结果；
* 【交互】间接返回的隐藏指针先占一个整数**参数**寄存器，再安排原参数，
  因此它可能使后面原本恰好放得下的整个聚合改走栈。

公开接口：类型构造（IntType/FloatType/PtrType/ArrayType/RecordType 及常量）、
layout()、classify()、encode()、decode()、pack_call()、unpack_call()；
返回侧 return_convention()、prepare_return_buffer()、encode_return()、
decode_return()、pack_invoke()。

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
    "INT_REG_COUNT", "FLOAT_REG_COUNT",
    "RET_INT_REG_COUNT", "RET_FLOAT_REG_COUNT",
    "STACK_ALIGN", "MAX_EIGHTBYTE_SIZE",
    "MAX_ARGS", "MAX_ARRAY_COUNT", "MAX_RECORD_FIELDS", "MAX_TYPE_SIZE",
    # 错误
    "ABIError", "InvalidTypeError", "RecursiveTypeError",
    "InvalidValueError", "IntegerOverflowError", "NonFiniteFloatError",
    "BufferError",
    # 类型
    "Type", "IntType", "FloatType", "PtrType", "ArrayType", "RecordType",
    "INT8", "INT16", "INT32", "INT64",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "DOUBLE", "PTR",
    # 布局 / 分类
    "Layout", "FieldLayout", "layout", "classify",
    # 参数编组
    "Placement", "CallImage",
    "encode", "decode", "pack_call", "unpack_call",
    # 返回编组
    "ReturnConvention", "ReturnImage", "InvokeImage", "Memory",
    "return_convention", "prepare_return_buffer",
    "encode_return", "decode_return", "pack_invoke",
]

# ---------------------------------------------------------------------------
# ABI 常量与输入上限
# ---------------------------------------------------------------------------

INTEGER = "INTEGER"
FLOAT = "FLOAT"
MEMORY = "MEMORY"

INT_REG_COUNT = 4
FLOAT_REG_COUNT = 2
RET_INT_REG_COUNT = 2      # 整数返回寄存器
RET_FLOAT_REG_COUNT = 2    # 浮点返回寄存器
STACK_ALIGN = 16
MAX_EIGHTBYTE_SIZE = 16  # 聚合超过 16 字节整体走栈（参数）/ 间接返回（返回）

MAX_ARGS = 1024                 # 单次调用最多参数个数
MAX_ARRAY_COUNT = 65536         # 数组最大元素数
MAX_RECORD_FIELDS = 256         # 记录最大字段数
MAX_TYPE_SIZE = 1 << 20         # 任一类型完整布局最大字节数（1 MiB）
MAX_ADDRESS = (1 << 64) - 1     # 合成地址空间上限

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


class BufferError(ABIError):
    """间接返回缓冲区非法：容量不足、未对齐或地址范围越界。

    缓冲区在任何写入之前完成校验，抛出此错误保证未发生部分写入。
    """


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


def _pack(args, hidden_ptr_values) -> CallImage:
    """编组核心。``hidden_ptr_values`` 中的指针按顺序先占用整数参数寄存器，
    再安排普通实参（间接返回的隐藏指针即由此注入）。"""
    int_regs: list = [None] * INT_REG_COUNT
    float_regs: list = [None] * FLOAT_REG_COUNT
    stack = bytearray()
    placements: list[Placement] = []
    next_int = 0
    next_float = 0

    for addr in hidden_ptr_values:
        int_regs[next_int] = addr
        next_int += 1

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
    return _pack(args, ())


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


# ---------------------------------------------------------------------------
# 合成内存：间接返回的有界缓冲区
# ---------------------------------------------------------------------------

class Memory:
    """一块合成地址空间，由 ``bytearray`` 支撑，基地址为 0。

    仅用于模拟“调用方传入有界返回缓冲区”：``allocate`` 按对齐 bump 分配，
    返回合成地址；所有读写都经过边界校验，绝不允许越过已分配字节。
    """

    __slots__ = ("_ram", "_brk")

    def __init__(self, capacity: int = MAX_TYPE_SIZE):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise InvalidTypeError(
                f"memory capacity must be an int, got {capacity!r}")
        if not 0 <= capacity <= MAX_ADDRESS:
            raise InvalidTypeError(
                f"memory capacity {capacity} out of range [0, 2**64 - 1]")
        self._ram = bytearray(capacity)
        self._brk = 0

    @property
    def capacity(self) -> int:
        return len(self._ram)

    def __len__(self) -> int:
        return len(self._ram)

    def allocate(self, size: int, align: int = 1) -> int:
        """在合成内存中 bump 分配 ``size`` 字节（按 ``align`` 对齐）。"""
        if isinstance(size, bool) or not isinstance(size, int):
            raise InvalidTypeError(f"size must be an int, got {size!r}")
        if isinstance(align, bool) or not isinstance(align, int) or align < 1:
            raise InvalidTypeError(f"align must be a positive int, got {align!r}")
        if size < 0 or size > MAX_TYPE_SIZE:
            raise BufferError(
                f"allocation size {size} out of range [0, {MAX_TYPE_SIZE}]")
        addr = _align_up(self._brk, align)
        if addr + size > len(self._ram) or addr + size > MAX_ADDRESS + 1:
            raise BufferError(
                f"out of memory: cannot allocate {size} bytes at {addr:#x} "
                f"(capacity {len(self._ram)})")
        self._brk = addr + size
        return addr

    def _check_range(self, addr: int, size: int) -> None:
        if isinstance(addr, bool) or not isinstance(addr, int):
            raise BufferError(f"address must be an int, got {addr!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BufferError(f"size must be a non-negative int, got {size!r}")
        if not 0 <= addr <= MAX_ADDRESS:
            raise BufferError(f"address {addr:#x} out of address space")
        # 先做 64 位范围检查（addr + size 不回绕），再做后备内存检查。
        if size > MAX_ADDRESS + 1 or addr > MAX_ADDRESS + 1 - size:
            raise BufferError(
                f"range [{addr:#x}, {addr + size:#x}) exceeds 64-bit "
                "address space")
        if addr + size > len(self._ram):
            raise BufferError(
                f"range [{addr:#x}, {addr + size:#x}) exceeds allocated "
                f"memory of {len(self._ram)} bytes")

    def write(self, addr: int, data) -> None:
        """一次性写入（单次切片赋值，不存在部分写入）。"""
        if not isinstance(data, (bytes, bytearray)):
            raise BufferError(f"data must be bytes, got {data!r}")
        self._check_range(addr, len(data))
        self._ram[addr:addr + len(data)] = data

    def read(self, addr: int, size: int) -> bytes:
        self._check_range(addr, size)
        return bytes(self._ram[addr:addr + size])

    def fill(self, addr: int, size: int, value: int) -> None:
        """把 ``[addr, addr+size)`` 全部填成 ``value``（供测试预置哨兵字节）。"""
        self._check_range(addr, size)
        if isinstance(value, bool) or not isinstance(value, int):
            raise BufferError(f"fill value must be an int, got {value!r}")
        if not 0 <= value <= 255:
            raise BufferError(f"fill value {value} out of byte range")
        self._ram[addr:addr + size] = bytes((value,)) * size


def prepare_return_buffer(t: Type, memory: Memory, address: int,
                          capacity: int) -> tuple[Memory, int]:
    """验证调用方提供的返回缓冲区：容量、对齐、地址范围。

    通过则原样返回 ``(memory, address)``；任一条件不满足抛 BufferError，
    且不触碰缓冲区任何字节。
    """
    _check_type(t)
    if not isinstance(memory, Memory):
        raise BufferError(f"expected a Memory, got {memory!r}")
    if isinstance(address, bool) or not isinstance(address, int):
        raise BufferError(f"address must be an int, got {address!r}")
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise BufferError(f"capacity must be an int, got {capacity!r}")
    if not 0 <= capacity <= MAX_ADDRESS + 1:
        raise BufferError(
            f"capacity {capacity} out of range [0, 2**64]")
    lay = layout(t)
    if not 0 <= address <= MAX_ADDRESS:
        raise BufferError(f"buffer address {address:#x} out of address space")
    if address % lay.align != 0:
        raise BufferError(
            f"buffer address {address:#x} not aligned to {lay.align} "
            f"(required by {t.name})")
    if capacity < lay.size:
        raise BufferError(
            f"buffer capacity {capacity} < return size {lay.size} "
            f"for {t.name}")
    # 范围：64 位地址空间 + 后备内存。两项检查均在任何写入之前。
    if address > MAX_ADDRESS + 1 - lay.size:
        raise BufferError(
            f"buffer range [{address:#x}, {address + lay.size:#x}) exceeds "
            "64-bit address space")
    if address + lay.size > len(memory):
        raise BufferError(
            f"buffer range [{address:#x}, {address + lay.size:#x}) exceeds "
            f"backing memory of {len(memory)} bytes")
    return memory, address


# ---------------------------------------------------------------------------
# 返回约定
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReturnConvention:
    """一个返回类型的安放约定。

    kind: ``"registers"``（逐块进返回寄存器）/ ``"memory"``（有界缓冲区
    间接返回）/ ``"none"``（空记录，不占任何返回资源）。
    address: 仅 ``memory`` 有效，调用方缓冲区地址（同时作为隐藏指针值）。
    """

    kind: str
    eightbytes: tuple[str, ...]
    size: int
    align: int
    address: int | None = None


def return_convention(t: Type, address: int | None = None) -> ReturnConvention:
    """决定返回类型的安放：空记录 none；超 16 字节 memory；其余 registers。

    ``memory`` 约定必须给出缓冲区 ``address``（调用方负责先用
    :func:`prepare_return_buffer` 验证）。
    """
    _check_type(t)
    lay = layout(t)
    cls = classify(t)
    if not cls:
        return ReturnConvention("none", (), 0, lay.align)
    if cls == (MEMORY,):
        if isinstance(address, bool) or not isinstance(address, int):
            raise BufferError(
                f"indirect return of {t.name} requires an integer buffer "
                f"address, got {address!r}")
        if not 0 <= address <= MAX_ADDRESS:
            raise BufferError(
                f"return buffer address {address:#x} out of address space")
        return ReturnConvention("memory", cls, lay.size, lay.align, address)
    # 不超过 16 字节 => 至多两个 8 字节块，返回寄存器（每类各 2 个）必够。
    need_int = cls.count(INTEGER)
    need_float = cls.count(FLOAT)
    if (need_int > RET_INT_REG_COUNT
            or need_float > RET_FLOAT_REG_COUNT):  # pragma: no cover - 不可达
        raise InvalidTypeError(
            f"return of {t.name} needs {need_int} int / {need_float} float "
            "return registers")
    return ReturnConvention("registers", cls, lay.size, lay.align)


# ---------------------------------------------------------------------------
# 返回映像与一次完整调用的映像
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReturnImage:
    """被调方产生的返回值映像。

    kind ``registers``：int_regs/float_regs 各长度 2，未用槽位 None；
    kind ``memory``：返回值已写入 InvokeImage 所指缓冲区，寄存器槽全 None；
    kind ``none``：空记录。
    """

    kind: str
    eightbytes: tuple[str, ...] = ()
    int_regs: tuple = (None, None)
    float_regs: tuple = (None, None)


@dataclass(frozen=True)
class InvokeImage:
    """一次“带返回值的调用”的完整编组计划与调用映像。

    return_type: 声明的返回类型；
    convention: 返回安放约定；
    call_image: 参数编组映像（间接返回时隐藏指针已先占一个整数寄存器）；
    memory/address: 间接返回的有界缓冲区（registers/none 时为 None）。
    """

    return_type: Type
    convention: ReturnConvention
    call_image: CallImage
    memory: Memory | None
    address: int | None

    @property
    def hidden_pointer(self) -> int | None:
        """间接返回时占用第一个整数参数寄存器的隐藏指针值。"""
        return self.address if self.convention.kind == "memory" else None


def pack_invoke(return_type: Type, args,
                memory: Memory | None = None,
                address: int = 0, capacity: int | None = None) -> InvokeImage:
    """调用方视角：编组一次带返回值的调用。

    * 返回类型走寄存器/为空：直接编组原参数；
    * 返回类型为大聚合：先在（给定或新建的）合成内存上落实有界返回缓冲区，
      再把其地址作为隐藏指针编入**第一个整数参数寄存器**，然后才安排原
      参数——因此后续聚合可能因寄存器被占而整体改走栈。

    缓冲区的容量、对齐与范围在校验全部通过前不发生任何写入。
    """
    _check_type(return_type)
    if not isinstance(args, (list, tuple)):
        raise InvalidValueError(
            f"args must be a list/tuple of (Type, value) pairs, got {args!r}")
    if len(args) > MAX_ARGS:
        raise InvalidValueError(
            f"too many arguments: {len(args)} > limit {MAX_ARGS}")
    lay = layout(return_type)
    cls = classify(return_type)

    hidden: tuple[int, ...] = ()
    mem: Memory | None = None
    addr: int | None = None
    if cls == (MEMORY,):
        if memory is None:
            mem = Memory()
            addr = mem.allocate(lay.size, lay.align)
        else:
            if capacity is None:
                raise BufferError(
                    "a caller-provided memory requires an explicit capacity")
            mem, addr = prepare_return_buffer(
                return_type, memory, address, capacity)
        hidden = (addr,)
    elif memory is not None:
        raise InvalidValueError(
            f"return type {return_type.name} is returned in registers; "
            "a buffer must not be supplied")

    conv = return_convention(return_type, addr)
    image = _pack(args, hidden)
    return InvokeImage(return_type, conv, image, mem, addr)


# ---------------------------------------------------------------------------
# 返回值编码 / 解码（被调方写返回、调用方读返回）
# ---------------------------------------------------------------------------

def encode_return(invoke: InvokeImage, value) -> ReturnImage:
    """被调方视角：把返回值按 InvokeImage 的约定编码。

    registers：逐块写入 2 个整数 / 2 个浮点返回寄存器；
    memory：先复核缓冲区大小、对齐与范围，再把完整布局（padding 全零）
    一次性写入，任何失败都不写部分结果；
    none：空记录。
    """
    if not isinstance(invoke, InvokeImage):
        raise InvalidValueError(f"expected an InvokeImage, got {invoke!r}")
    t = invoke.return_type
    conv = invoke.convention
    lay = layout(t)

    if conv.kind == "none":
        # 仍然校验值（必须是空记录对应的 {}）。
        encode(t, value, lay, path="ret")
        return ReturnImage("none")

    if conv.kind == "memory":
        if not isinstance(invoke.memory, Memory) or invoke.address is None:
            raise BufferError("indirect return has no valid buffer")
        # 被调方不信任调用方：写入前再次验证容量/对齐/范围。
        prepare_return_buffer(t, invoke.memory, invoke.address,
                              len(invoke.memory) - invoke.address)
        data = encode(t, value, lay, path="ret")  # padding 已全零
        assert len(data) == lay.size
        invoke.memory.write(invoke.address, data)  # 单次切片写入
        return ReturnImage("memory", conv.eightbytes)

    # registers
    data = encode(t, value, lay, path="ret")
    int_regs: list = [None, None]
    float_regs: list = [None, None]
    ni = nf = 0
    for k, c in enumerate(conv.eightbytes):
        chunk = data[8 * k:8 * k + 8].ljust(8, b"\x00")
        if c == INTEGER:
            int_regs[ni] = int.from_bytes(chunk, "little")
            ni += 1
        else:
            float_regs[nf] = struct.unpack("<d", chunk)[0]
            nf += 1
    return ReturnImage("registers", conv.eightbytes,
                       tuple(int_regs), tuple(float_regs))


def decode_return(invoke: InvokeImage, ret: ReturnImage):
    """调用方视角：按 InvokeImage 的约定从返回映像重建 Python 值。

    只按类型化叶子读取，聚合中的 padding 字节永不被当作值。
    """
    if not isinstance(invoke, InvokeImage):
        raise InvalidValueError(f"expected an InvokeImage, got {invoke!r}")
    if not isinstance(ret, ReturnImage):
        raise InvalidValueError(f"expected a ReturnImage, got {ret!r}")
    t = invoke.return_type
    conv = invoke.convention
    lay = layout(t)

    if ret.kind != conv.kind:
        raise InvalidValueError(
            f"return image kind {ret.kind!r} does not match convention "
            f"{conv.kind!r} for {t.name}", "ret")

    if conv.kind == "none":
        return decode(t, b"", lay, "ret")

    if conv.kind == "memory":
        if not isinstance(invoke.memory, Memory) or invoke.address is None:
            raise BufferError("indirect return has no valid buffer")
        # read 内部做 64 位范围与后备内存边界校验；只按类型化叶子解码。
        data = invoke.memory.read(invoke.address, lay.size)
        return decode(t, data, lay, "ret")

    chunks: list[bytes] = []
    int_iter = iter(range(RET_INT_REG_COUNT))
    float_iter = iter(range(RET_FLOAT_REG_COUNT))
    for c in conv.eightbytes:
        if c == INTEGER:
            idx = next(int_iter)
            v = ret.int_regs[idx]
            if v is None:
                raise InvalidValueError(
                    f"missing int return register {idx} for {t.name}", "ret")
            chunks.append(v.to_bytes(8, "little"))
        else:
            idx = next(float_iter)
            v = ret.float_regs[idx]
            if v is None:
                raise InvalidValueError(
                    f"missing float return register {idx} for {t.name}", "ret")
            chunks.append(struct.pack("<d", v))
    data = b"".join(chunks)[:lay.size]
    return decode(t, data, lay, "ret")
