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
    "OverlappingBitFieldError", "InvalidUnionTagError",
    "VariadicPromotionError", "OverlappingStorageError", "BridgeError",
    # 类型
    "Type", "IntType", "FloatType", "PtrType", "ArrayType", "RecordType",
    "Int128Type", "BitField", "UnionType", "Float32Type",
    "INT8", "INT16", "INT32", "INT64",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "INT128", "UINT128",
    "DOUBLE", "PTR", "FLOAT32",
    # 布局 / 分类
    "Layout", "FieldLayout", "layout", "classify",
    # 编组
    "Placement", "CallImage",
    "encode", "decode", "pack_call", "unpack_call",
    # ---- v2：显式描述符双布局 / 联合 / 位域 / 变参 / 桥接 / 证书 ----
    "LITTLE", "BIG",
    "RegisterRules", "ABIDescriptor", "LAYOUT_A", "LAYOUT_B",
    "UnionValue", "FloatBits", "Param", "SharedUnit",
    "BitSeg", "LayoutV2", "FieldV2", "FrameRegion",
    "PlacementV2", "ReturnInfo", "SaveEntry", "PlannedCall",
    "CallImageV2", "LayoutCertificate", "FrameCertificate",
    "ImageCertificate", "verify_image",
    "layout_v2", "plan_call", "pack_call_v2", "unpack_call_v2",
    "unpack_return_v2", "encode_v2", "decode_v2",
    "classify_v2", "certify_layout", "certify_frame",
    "bridge_call", "bridge_call_into",
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
    """浮点值为 NaN 或 Infinity，v1 入口拒绝（v2 中 NaN 按原始载荷位保留）。"""


class OverlappingBitFieldError(InvalidTypeError):
    """同一存储组中两个活动位域占据了重叠的位（联合内的合法重叠除外）。"""


class InvalidUnionTagError(InvalidValueError):
    """联合体活动成员标签缺失、越界，或无标签解码时活动成员不可辨识。"""


class VariadicPromotionError(InvalidValueError):
    """可变实参无法按默认提升规则表示（如 128 位整数/聚合走变参）。"""


class OverlappingStorageError(InvalidValueError):
    """共享存储单元的多个视图在重叠位上写入不一致的值。"""


class BridgeError(ABIError):
    """A→B 调用映像桥接失败。``reason`` 为机器可读原因码，错误带参数路径。

    reason 取值：``"overlap-conflict"``、``"tag-out-of-range"``、
    ``"unrepresentable-promotion"``、``"indistinguishable-union"``、
    ``"value-range"``。桥接失败前不写入目标缓冲区任何字节。
    """

    def __init__(self, message: str, path: str | None = None,
                 reason: str = "bridge-failed"):
        self.reason = reason
        super().__init__(message, path)


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
            if isinstance(ftype, BitField):
                # 位域仅 v2 描述符布局支持；外层名必须与位域名一致。
                if fname != ftype.name:
                    raise InvalidTypeError(
                        f"record {self._name!r}: bitfield tuple name {fname!r} "
                        f"!= BitField name {ftype.name!r}")
                check_name = ftype.name
            else:
                _check_type(ftype)
                check_name = fname
            if check_name in seen:
                raise InvalidTypeError(
                    f"record {self._name!r}: duplicate field name {check_name!r}")
            seen.add(check_name)
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


# ===========================================================================
# v2：显式双布局描述符 / 联合 / 位域 / 128 位整数 / 变参 /
#     返回值间接区 / 变参寄存器保存区 / 按位共享存储单元 / 桥接 / 位证书
#
# v2 与 v1 的关系：v1（本文件上半部分）行为完全不变；v2 不根据主机 struct
# 布局推导任何东西，所有字节序/对齐/寄存器规则都来自显式 ABIDescriptor。
# 规划器（layout_v2/plan_call，只产生几何信息）与编解码器
# （encode_v2/pack_call_v2，接触值）严格分离。
# ===========================================================================

LITTLE = "little"
BIG = "big"


# ---------------------------------------------------------------------------
# 描述符
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegisterRules:
    """一个目标布局的寄存器槽规则。"""

    int_count: int          # 整数寄存器个数（每槽 8 字节）
    float_count: int        # 浮点寄存器个数
    max_aggregate_bytes: int  # 聚合可走寄存器的最大尺寸（超过整体走栈）

    def __post_init__(self):
        if self.int_count < 0 or self.float_count < 0:
            raise InvalidTypeError("register counts must be non-negative")
        if self.max_aggregate_bytes <= 0:
            raise InvalidTypeError("max_aggregate_bytes must be positive")


@dataclass(frozen=True)
class ABIDescriptor:
    """目标布局的显式声明。新布局只能由这里描述，禁止从主机布局推导。"""

    name: str
    endian: str                       # LITTLE / BIG
    max_align: int                    # 聚合对齐上限（标量仍自然对齐）
    int_regs: RegisterRules
    stack_align: int = 16
    pointer_size: int = 8
    save_int_slots: int = 0           # 变参寄存器保存区每库容量（字）
    save_float_slots: int = 0

    def __post_init__(self):
        if self.endian not in (LITTLE, BIG):
            raise InvalidTypeError(
                f"endian must be LITTLE or BIG, got {self.endian!r}")
        if self.pointer_size != 8:
            raise InvalidTypeError("only 8-byte pointers are described")
        if self.max_align not in (1, 2, 4, 8, 16):
            raise InvalidTypeError(f"bad max_align {self.max_align}")

    @property
    def struct_prefix(self) -> str:
        return "<" if self.endian == LITTLE else ">"

    def flipped(self) -> "ABIDescriptor":
        """返回仅字节序相反的描述符（证书重放用）。"""
        endian = BIG if self.endian == LITTLE else LITTLE
        return ABIDescriptor(
            f"{self.name}!{endian}", endian, self.max_align, self.int_regs,
            self.stack_align, self.pointer_size,
            self.save_int_slots, self.save_float_slots)


# 布局 A：与 v1 明示契约同构（小端、对齐上限 8、4 整数 + 2 浮点、16 字节界），
# 另装备变参保存区（仅在出现可变参数时占用帧空间，固定调用与 v1 完全一致）。
LAYOUT_A = ABIDescriptor(
    name="A", endian=LITTLE, max_align=8,
    int_regs=RegisterRules(int_count=4, float_count=2,
                           max_aggregate_bytes=16),
    save_int_slots=4, save_float_slots=2)

# 布局 B：相反字节序、对齐上限 16、6 整数 + 4 浮点、32 字节寄存器聚合界。
LAYOUT_B = ABIDescriptor(
    name="B", endian=BIG, max_align=16,
    int_regs=RegisterRules(int_count=6, float_count=4,
                           max_aggregate_bytes=32),
    save_int_slots=6, save_float_slots=4)


# ---------------------------------------------------------------------------
# v2 类型：128 位整数、位域、联合
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Int128Type(Type):
    """16 字节整数（有/无符号）。两端口布局均支持。"""

    signed: bool

    @property
    def name(self) -> str:
        return f"{'i' if self.signed else 'u'}128"

    @property
    def range(self) -> tuple[int, int]:
        if self.signed:
            return (-(1 << 127), (1 << 127) - 1)
        return (0, (1 << 128) - 1)


INT128 = Int128Type(True)
UINT128 = Int128Type(False)


@dataclass(frozen=True)
class Float32Type(Type):
    """4 字节 IEEE-754 binary32 浮点（仅 v2 描述符布局支持；v1 入口拒绝）。

    变参默认提升时 f32 -> f64（binary64）。
    """

    @property
    def name(self) -> str:
        return "f32"


FLOAT32 = Float32Type()


@dataclass(frozen=True)
class BitField:
    """记录内的位域声明（不是独立 Type，只能出现在记录/存储单元中）。

    base 取 1/2/4/8 字节整数类型，决定存储单元；width 为位宽（1..128 限：
    1..64）。lsb 给定时为单元内显式起始位（从存储整数的 LSB 起算），与已
    占用位重叠即 OverlappingBitFieldError；省略时顺序打包，单元装不下则
    自动跨入下一同型存储单元。
    """

    name: str
    base: IntType
    width: int
    lsb: int | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise InvalidTypeError("bitfield name must be a non-empty str")
        if not isinstance(self.base, IntType) or self.base.size not in (1, 2, 4, 8):
            raise InvalidTypeError(
                f"bitfield {self.name!r}: base must be a 1/2/4/8-byte IntType")
        if isinstance(self.width, bool) or not 1 <= self.width <= self.base.size * 8:
            raise InvalidTypeError(
                f"bitfield {self.name!r}: width 1..{self.base.size * 8} "
                f"expected, got {self.width!r}")
        if self.lsb is not None:
            if isinstance(self.lsb, bool) or not 0 <= self.lsb < self.base.size * 8:
                raise InvalidTypeError(
                    f"bitfield {self.name!r}: explicit lsb out of unit")


class UnionType(Type):
    """带活动成员标签的联合。

    members 为 [(name, Type), ...]；默认在偏移 0 放置 1 字节无符号标签，
    成员体从标签之后开始、全部重叠安放。tag_type=None 得到无标签联合
    （C 语义；解码时必须能唯一辨识，否则 InvalidUnionTagError）。
    标签值为成员下标或成员名；越界标签 InvalidUnionTagError。
    """

    __slots__ = ("_name", "_members", "_tag_type", "_defined")

    def __init__(self, name: str, tag_type=UINT8):
        if not isinstance(name, str) or not name:
            raise InvalidTypeError("union name must be a non-empty str")
        if tag_type is not None and not isinstance(tag_type, IntType):
            raise InvalidTypeError("union tag_type must be an IntType or None")
        self._name = name
        self._tag_type = tag_type
        self._members: tuple[tuple[str, Type], ...] | None = None
        self._defined = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def tag_type(self):
        return self._tag_type

    def define(self, members) -> None:
        if self._defined:
            raise InvalidTypeError(f"union {self._name!r} is already defined")
        items = tuple(members)
        if not items:
            raise InvalidTypeError(f"union {self._name!r} needs >= 1 member")
        if len(items) > MAX_RECORD_FIELDS:
            raise InvalidTypeError(
                f"union {self._name!r} has {len(items)} members, "
                f"limit is {MAX_RECORD_FIELDS}")
        seen: set[str] = set()
        for item in items:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise InvalidTypeError(
                    f"union {self._name!r}: member must be (name, Type)")
            mname, mtype = item
            if not isinstance(mname, str) or not mname:
                raise InvalidTypeError("union member name must be non-empty str")
            if mname in seen:
                raise InvalidTypeError(
                    f"union {self._name!r}: duplicate member {mname!r}")
            seen.add(mname)
            _check_type(mtype)
        self._members = items
        self._defined = True

    @property
    def members(self) -> tuple[tuple[str, Type], ...]:
        if self._members is None:
            raise InvalidTypeError(f"union {self._name!r} is not defined yet")
        return self._members


@dataclass(frozen=True)
class UnionValue:
    """联合体的带标签值：``tag`` 为成员下标（int）或成员名（str）。"""

    tag: object
    value: object


@dataclass(frozen=True)
class FloatBits:
    """浮点的原始 NaN 位载荷（用于逐位保留 NaN payload）。仅接受 NaN 位型。

    width=64 对应 binary64，width=32 对应 binary32。
    """

    pattern: int
    width: int = 64

    def __post_init__(self):
        if self.width not in (32, 64):
            raise InvalidValueError("FloatBits width must be 32 or 64")
        if not 0 <= self.pattern <= (1 << self.width) - 1:
            raise InvalidValueError(
                f"FloatBits pattern out of {self.width}-bit range")
        if self.width == 64:
            exp, mbits, exp_bits = (self.pattern >> 52) & 0x7FF, 52, 11
        else:
            exp, mbits, exp_bits = (self.pattern >> 23) & 0xFF, 23, 8
        mant = self.pattern & ((1 << mbits) - 1)
        if exp != (1 << exp_bits) - 1 or mant == 0:
            raise InvalidValueError(
                "FloatBits only carries NaN patterns (use float for finite "
                "values; infinity is rejected at the codec boundary)")

    @property
    def sign(self) -> int:
        return (self.pattern >> (self.width - 1)) & 1

    @property
    def quiet(self) -> int:
        # quiet NaN 位是尾数最高位：f64 的 bit51、f32 的 bit22
        return (self.pattern >> (51 if self.width == 64 else 22)) & 1

    @property
    def payload(self) -> int:
        mbits = 51 if self.width == 64 else 22
        return self.pattern & ((1 << mbits) - 1)

    def widen_to_64(self) -> "FloatBits":
        """binary32 NaN -> binary64 NaN：符号保留、指数展宽、尾数载荷零扩展。"""
        if self.width == 64:
            return self
        sign = self.sign
        quiet = (self.pattern >> 22) & 1
        payload = self.pattern & ((1 << 22) - 1)
        pat = (sign << 63) | (0x7FF << 52) | (quiet << 51) | payload
        return FloatBits(pat, 64)

    def __eq__(self, other) -> bool:
        return (isinstance(other, FloatBits) and other.pattern == self.pattern
                and other.width == self.width)

    def __repr__(self) -> str:
        return f"FloatBits(0x{self.pattern:0{self.width // 4}x}, width={self.width})"


@dataclass(frozen=True)
class Param:
    """实参规约：类型 + 变参标记 + 按位共享存储单元归属。

    shared 为单元 id，shared_size 为该单元总字节数（同 id 必须一致）；
    tag_hint 供无标签联合解码/桥接时指定活动成员（下标或成员名）。
    """

    type: Type
    variadic: bool = False
    shared: str | None = None
    shared_size: int | None = None
    tag_hint: object = None

    def __post_init__(self):
        _check_type(self.type)
        if self.shared is not None:
            if not isinstance(self.shared, str) or not self.shared:
                raise InvalidTypeError("shared unit id must be a non-empty str")
            if not isinstance(self.shared_size, int) or self.shared_size <= 0:
                raise InvalidTypeError("shared unit requires a positive size")


@dataclass(frozen=True)
class SharedUnit:
    """一个按位共享存储单元的运行时值：id、总字节数、各视图值。

    views: ``{视图名: 值}``；视图名必须与该实参 Param 的记录类型同名。
    """

    unit_id: str
    size: int
    views: dict

    def __post_init__(self):
        if not isinstance(self.unit_id, str) or not self.unit_id:
            raise InvalidValueError("SharedUnit id must be a non-empty str")
        if not isinstance(self.size, int) or self.size <= 0:
            raise InvalidValueError("SharedUnit size must be positive")
        if not isinstance(self.views, dict) or not self.views:
            raise InvalidValueError("SharedUnit needs a non-empty views dict")


# ---------------------------------------------------------------------------
# v2 布局几何（规划器，不接触任何值）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BitSeg:
    """一段连续位的静态/证据位置。

    byte/lsb：相对所属缓冲的字节与字节内位（自该字节 LSB 起算）；
    role: "data" 标量整词 / "float" 浮点整词 / "bits" 位域 / "tag" 联合标签；
    cls: 寄存器分类 INTEGER/FLOAT；value 为该段实际承载的无符号位型
    （静态几何中为 None）。
    """

    owner: str
    name: str
    byte: int
    lsb: int
    width: int
    role: str
    cls: str
    value: int | None = None


@dataclass(frozen=True)
class FieldV2:
    """记录/联合中的一个成员或一个位域存储组。"""

    name: str
    offset: int
    size: int
    type: Type | None
    kind: str                 # "member" / "bits"
    segs: tuple[BitSeg, ...] = ()
    base_size: int = 0        # bits 组：存储单元字节数
    unit_count: int = 0       # bits 组：占用单元数


@dataclass(frozen=True)
class LayoutV2:
    type: Type
    size: int
    align: int
    fields: tuple[FieldV2, ...] = ()
    segs: tuple[BitSeg, ...] = ()
    member_offset: int = 0    # union：成员体起始偏移
    tag_size: int = 0         # union：标签字节数（0 表示无标签）


def _cap_align(align: int, desc: ABIDescriptor) -> int:
    return min(align, desc.max_align)


def _shift_segs(segs, delta: int, owner_prefix: str = ""):
    out = []
    for s in segs:
        owner = f"{owner_prefix}{s.owner}" if owner_prefix else s.owner
        out.append(BitSeg(owner, s.name, s.byte + delta, s.lsb, s.width,
                          s.role, s.cls, s.value))
    return tuple(out)


def _layout_v2(t: Type, desc: ABIDescriptor, visiting: frozenset[int]) -> LayoutV2:
    if isinstance(t, IntType):
        return LayoutV2(t, t.size, t.size, (),
                        (BitSeg("", "value", 0, 0, t.size * 8,
                                "data", INTEGER),))
    if isinstance(t, Int128Type):
        return LayoutV2(t, 16, _cap_align(16, desc), (),
                        (BitSeg("", "value", 0, 0, 128, "data", INTEGER),))
    if isinstance(t, PtrType):
        return LayoutV2(t, 8, 8, (),
                        (BitSeg("", "value", 0, 0, 64, "data", INTEGER),))
    if isinstance(t, FloatType):
        return LayoutV2(t, 8, 8, (),
                        (BitSeg("", "value", 0, 0, 64, "float", FLOAT),))
    if isinstance(t, Float32Type):
        return LayoutV2(t, 4, 4, (),
                        (BitSeg("", "value", 0, 0, 32, "float", FLOAT),))

    if isinstance(t, ArrayType):
        el = _layout_v2(t.elem, desc, visiting)
        size = el.size * t.count
        if size > MAX_TYPE_SIZE:
            raise InvalidTypeError(
                f"type size {size} exceeds limit {MAX_TYPE_SIZE}")
        align = _cap_align(el.align, desc) if t.count else 1
        segs = []
        for i in range(t.count):
            segs.extend(_shift_segs(el.segs, i * el.size))
        fields = tuple(
            FieldV2(f"[{i}]", i * el.size, el.size, t.elem, "member",
                    _shift_segs(el.segs, i * el.size))
            for i in range(t.count))
        return LayoutV2(t, size, align, fields, tuple(segs))

    if isinstance(t, RecordType):
        if id(t) in visiting:
            raise RecursiveTypeError(
                f"recursive record type {t.name!r} has infinite layout")
        visiting = visiting | {id(t)}
        return _layout_record(t, desc, visiting)

    if isinstance(t, UnionType):
        if id(t) in visiting:
            raise RecursiveTypeError(
                f"recursive union type {t.name!r} has infinite layout")
        visiting = visiting | {id(t)}
        return _layout_union(t, desc, visiting)

    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def _bf_allocate_group(run, group_start: int, path_name: str):
    """把一个同 base 的连续位域序列分配到若干存储单元。

    返回 (组字节数, [BitSeg...])，seg 位置相对组起点；显式 lsb 与已分配
    位重叠即 OverlappingBitFieldError；顺序位域装不下时自动跨入下一单元。
    """
    base = run[0].base
    unit_bits = base.size * 8
    segs: list[BitSeg] = []
    units = 1
    cursor = 0           # 顺序位域：当前单元内已用到的位
    occupied: list[tuple[int, int]] = []  # 已占用绝对位区间（半开）

    def claim(abs_lsb: int, width: int, name: str):
        for lo, hi in occupied:
            if abs_lsb < hi and lo < abs_lsb + width:
                raise OverlappingBitFieldError(
                    f"bitfield {name!r} at bit {abs_lsb} overlaps allocated "
                    f"bits [{lo}, {hi})", path_name)
        occupied.append((abs_lsb, abs_lsb + width))

    for bf in run:
        if bf.base.size != base.size:
            raise InvalidTypeError(
                f"bitfield group at {path_name} mixes storage unit widths")
        if bf.lsb is not None:
            abs_lsb = bf.lsb
            claim(abs_lsb, bf.width, bf.name)
            start_unit, local = divmod(abs_lsb, unit_bits)
            remain = unit_bits - local
            if bf.width <= remain:
                segs.append(BitSeg("", bf.name, start_unit * base.size, local,
                                   bf.width, "bits", INTEGER))
            else:
                segs.append(BitSeg("", bf.name, start_unit * base.size, local,
                                   remain, "bits", INTEGER))
                segs.append(BitSeg("", bf.name, (start_unit + 1) * base.size, 0,
                                   bf.width - remain, "bits", INTEGER))
            units = max(units, -(-(abs_lsb + bf.width) // unit_bits))
            # 顺序游标推进到显式声明之后（同单元连续时）
            cursor = (abs_lsb + bf.width) % unit_bits
        else:
            abs_lsb = (units - 1) * unit_bits + cursor
            claim(abs_lsb, bf.width, bf.name)
            if cursor + bf.width <= unit_bits:
                segs.append(BitSeg("", bf.name, (units - 1) * base.size, cursor,
                                   bf.width, "bits", INTEGER))
                cursor += bf.width
            else:
                # 跨存储单元：当前单元剩余 + 下一单元前部
                remain = unit_bits - cursor
                segs.append(BitSeg("", bf.name, (units - 1) * base.size, cursor,
                                   remain, "bits", INTEGER))
                units += 1
                segs.append(BitSeg("", bf.name, (units - 1) * base.size, 0,
                                   bf.width - remain, "bits", INTEGER))
                cursor = bf.width - remain
    return units * base.size, tuple(segs)


def _layout_record(t: RecordType, desc: ABIDescriptor, visiting) -> LayoutV2:
    offset = 0
    align = 1
    fields: list[FieldV2] = []
    all_segs: list[BitSeg] = []
    items = t.fields
    i = 0
    while i < len(items):
        fname, ftype = items[i]
        if isinstance(ftype, BitField):
            # 收集连续位域序列：存储单元由 base 宽度决定（signed 只影响
            # 值解释，不影响分组），故 base 尺寸相同即同组。
            run = [ftype]
            j = i + 1
            while (j < len(items)
                   and isinstance(items[j][1], BitField)
                   and items[j][1].base.size == ftype.base.size):
                run.append(items[j][1])
                j += 1
            unit_align = _cap_align(ftype.base.size, desc)
            group_start = _align_up(offset, unit_align)
            gsize, gsegs = _bf_allocate_group(
                run, group_start, f"record {t.name!r}")
            shifted = _shift_segs(gsegs, group_start)
            # FieldV2.segs 统一为绝对位段（成员/位域组一致）；编解码位域时
            # 用 s.byte - f.offset 还原组内相对字节。
            fields.append(FieldV2(
                f"@bf[{run[0].name}]", group_start, gsize, None, "bits",
                shifted, ftype.base.size, gsize // ftype.base.size))
            all_segs.extend(shifted)
            align = max(align, unit_align)
            offset = group_start + gsize
            i = j
            continue
        fl = _layout_v2(ftype, desc, visiting)
        al = _cap_align(fl.align, desc)
        offset = _align_up(offset, al)
        shifted = _shift_segs(fl.segs, offset, f"{fname}.")
        fields.append(FieldV2(fname, offset, fl.size, ftype, "member",
                              shifted))
        all_segs.extend(shifted)
        align = max(align, al)
        offset += fl.size
        i += 1
    size = _align_up(offset, align)
    if size > MAX_TYPE_SIZE:
        raise InvalidTypeError(f"type size {size} exceeds limit {MAX_TYPE_SIZE}")
    return LayoutV2(t, size, align, tuple(fields), tuple(all_segs))


def _layout_union(t: UnionType, desc: ABIDescriptor, visiting) -> LayoutV2:
    tag = t.tag_type
    tag_size = tag.size if tag is not None else 0
    member_layouts = [(n, mt, _layout_v2(mt, desc, visiting))
                      for n, mt in t.members]
    max_align = 1
    max_member_size = 0
    for _n, _mt, ml in member_layouts:
        max_align = max(max_align, _cap_align(ml.align, desc))
        max_member_size = max(max_member_size, ml.size)
    member_offset = _align_up(tag_size, max_align) if member_layouts else tag_size
    size = _align_up(member_offset + max_member_size,
                     max(max_align, _cap_align(tag_size, desc) if tag else 1))
    if size > MAX_TYPE_SIZE:
        raise InvalidTypeError(f"type size {size} exceeds limit {MAX_TYPE_SIZE}")

    fields: list[FieldV2] = []
    segs: list[BitSeg] = []
    if tag is not None:
        tag_seg = BitSeg("", "$tag", 0, 0, tag_size * 8, "tag", INTEGER)
        fields.append(FieldV2("$tag", 0, tag_size, tag, "member", (tag_seg,)))
        segs.append(tag_seg)
    for n, mt, ml in member_layouts:
        shifted = _shift_segs(ml.segs, member_offset, f"{n}.")
        fields.append(FieldV2(n, member_offset, ml.size, mt, "member", shifted))
        segs.extend(shifted)
    return LayoutV2(t, size, max(max_align, 1), tuple(fields), tuple(segs),
                    member_offset, tag_size)


def layout_v2(t: Type, desc: ABIDescriptor) -> LayoutV2:
    """在描述符 ``desc`` 下计算类型的显式字节布局与逐位静态证据。"""
    _check_type(t)
    if not isinstance(desc, ABIDescriptor):
        raise InvalidTypeError("desc must be an ABIDescriptor")
    return _layout_v2(t, desc, frozenset())


def classify_v2(t: Type, desc: ABIDescriptor) -> tuple[str, ...]:
    """描述符下的逐 8 字节块分类（几何判定，不接触值）。

    联合按所有成员的最坏情况分类；空类型返回 ()；超界返回 (MEMORY,)。
    """
    lay = layout_v2(t, desc)
    if lay.size == 0:
        return ()
    if lay.size > desc.int_regs.max_aggregate_bytes:
        return (MEMORY,)
    classes = []
    for k in range(0, lay.size, 8):
        lo, hi = k, k + 8
        cls = None
        for s in lay.segs:
            if s.byte < hi and s.byte + (s.width + s.lsb + 7) // 8 > lo:
                if s.cls == INTEGER:
                    cls = INTEGER
                    break
                cls = FLOAT
        classes.append(cls or INTEGER)   # 纯填充块按整数类
    return tuple(classes)



# ---------------------------------------------------------------------------
# 调用帧规划：返回值间接区 + 实参 + 变参寄存器保存区
#
# 帧几何（字节偏移自调用者帧基起算）：
#   [ return_area | arg slots (regs 无字节 / stack 槽) | variadic save area ]
# 仅在出现 variadic 实参时分配保存区，尺寸由描述符 save_*_slots 固定。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameRegion:
    name: str
    offset: int
    size: int
    kind: str            # "return-area" / "arg-slot" / "save-int" / "save-float"
    arg_index: int = -1


@dataclass(frozen=True)
class PlacementV2:
    arg_index: int
    kind: str                       # "registers"/"stack"/"none"/"alias"
    eightbytes: tuple[str, ...]
    int_regs: tuple[int, ...]
    float_regs: tuple[int, ...]
    stack_offset: int | None
    size: int
    variadic: bool
    promoted_from: Type | None = None
    shared: str | None = None
    alias_of: int = -1              # kind=="alias" 时指向单元属主实参下标


@dataclass(frozen=True)
class ReturnInfo:
    ret_type: Type
    indirect: bool          # 是否经返回值间接区（> 寄存器聚合界或 128 位整数）
    area_offset: int        # 恒为 0
    area_size: int
    eightbytes: tuple[str, ...]
    int_regs: tuple[int, ...]
    float_regs: tuple[int, ...]


@dataclass(frozen=True)
class SaveEntry:
    """保存区中的一个槽：来自哪个变参实参、寄存器号、保存字节偏移。"""

    kind: str               # "int" / "float"
    slot: int
    offset: int             # 相对帧基（保存区内）
    arg_index: int
    eightbyte: int          # 该实参内第几块


@dataclass(frozen=True)
class SharedPlan:
    """一个按位共享存储单元的几何规划。"""

    unit_id: str
    owner: int                  # 首个实参下标
    aliases: tuple[int, ...]    # 其余视图实参下标
    size: int
    eightbytes: tuple[str, ...]
    align: int


@dataclass(frozen=True)
class PlannedCall:
    desc: ABIDescriptor
    return_info: ReturnInfo | None
    placements: tuple[PlacementV2, ...]
    regions: tuple[FrameRegion, ...]
    save_entries: tuple[SaveEntry, ...]
    frame_size: int
    int_regs: tuple[int | None, ...]
    float_regs: tuple[int | None, ...]
    shared_units: tuple[SharedPlan, ...] = ()


def _promote(t: Type):
    """变参默认提升（合成 C 规则）：

    * 1/2 字节整数（有/无符号）→ 4 字节同符号整数；
    * f32 → f64（binary64）；
    * 4/8 字节整数、指针、f64 原样；
    * 其余（128 位整数、聚合、数组）无法表示 → VariadicPromotionError。
    返回 (提升后类型, 原类型 or None)。
    """
    if isinstance(t, Float32Type):
        return DOUBLE, t
    if isinstance(t, FloatType):
        return t, None
    if isinstance(t, IntType):
        if t.size in (1, 2):
            return IntType(4, t.signed), t
        return t, None
    if isinstance(t, PtrType):
        return t, None
    raise VariadicPromotionError(
        f"variadic argument of type {t.name} cannot undergo default "
        "integer/float promotion (only 1/2/4/8-byte integers, pointers and "
        "f64 are representable)")


def _return_info(ret_type: Type | None, desc: ABIDescriptor) -> ReturnInfo | None:
    if ret_type is None:
        return None
    lay = layout_v2(ret_type, desc)
    rules = desc.int_regs
    if lay.size == 0:
        return ReturnInfo(ret_type, False, 0, 0, (), (), ())
    indirect = lay.size > rules.max_aggregate_bytes or isinstance(ret_type, Int128Type)
    if indirect:
        area = _align_up(lay.size, desc.stack_align)
        return ReturnInfo(ret_type, True, 0, area, (), (), ())
    cls = classify_v2(ret_type, desc)
    iregs: list[int] = []
    fregs: list[int] = []
    for k, c in enumerate(cls):
        (iregs if c == INTEGER else fregs).append(k)
    return ReturnInfo(ret_type, False, 0, 0, cls, tuple(iregs), tuple(fregs))


def _shared_classify(plan_size: int, view_layouts, desc: ABIDescriptor):
    """按共享单元全部视图的位段并集逐 8 字节分类（纯填充块按整数类）。"""
    if plan_size > desc.int_regs.max_aggregate_bytes:
        return (MEMORY,)
    marks = bytearray(plan_size)  # 每字节：0 无位 / 1 FLOAT / 2 INTEGER
    for lay in view_layouts:
        for s in lay.segs:
            for bit in range(s.byte * 8 + s.lsb,
                             s.byte * 8 + s.lsb + s.width):
                byte = bit // 8
                level = 2 if s.cls == INTEGER else 1
                if marks[byte] < level:
                    marks[byte] = level
    classes = []
    for k in range(0, plan_size, 8):
        block = marks[k:k + 8]
        if 2 in block:
            classes.append(INTEGER)
        elif 1 in block:
            classes.append(FLOAT)
        else:
            classes.append(INTEGER)
    return tuple(classes)


def plan_call(params, desc: ABIDescriptor,
              ret_type: Type | None = None) -> PlannedCall:
    """纯几何规划：寄存器分配、栈槽、返回区、保存区、共享单元、帧尺寸。

    不接触任何值。params 为 Param 列表（最多 MAX_ARGS 个）。同一 ``shared``
    id 的实参构成一个按位共享存储单元：首个实参是属主、占用一份资源，
    其余实参为别名（``kind="alias"``），分类取所有视图位段的并集。
    """
    if not isinstance(desc, ABIDescriptor):
        raise InvalidTypeError("desc must be an ABIDescriptor")
    if not isinstance(params, (list, tuple)):
        raise InvalidTypeError("params must be a list/tuple of Param")
    if len(params) > MAX_ARGS:
        raise InvalidValueError(
            f"too many arguments: {len(params)} > limit {MAX_ARGS}")

    # 预分组共享单元（校验尺寸一致、视图不越界、别名不可变参）
    shared_order: list[str] = []
    shared_meta: dict[str, dict] = {}
    for i, p in enumerate(params):
        if not isinstance(p, Param):
            raise InvalidTypeError(f"param[{i}] must be a Param", f"arg[{i}]")
        if p.shared is None:
            continue
        if p.variadic:
            raise VariadicPromotionError(
                "a shared-storage view cannot be a variadic argument",
                f"arg[{i}]")
        meta = shared_meta.get(p.shared)
        if meta is None:
            meta = {"owner": i, "aliases": [], "size": p.shared_size,
                    "layouts": []}
            shared_meta[p.shared] = meta
            shared_order.append(p.shared)
        elif meta["size"] != p.shared_size:
            raise InvalidTypeError(
                f"shared unit {p.shared!r} declared with conflicting sizes "
                f"{meta['size']} and {p.shared_size}", f"arg[{i}]")
        try:
            lay = layout_v2(p.type, desc)
        except OverlappingBitFieldError as exc:
            raise type(exc)(exc.message, f"arg[{i}]") from None
        if lay.size > p.shared_size:
            raise InvalidTypeError(
                f"shared view {p.type.name} size {lay.size} exceeds unit "
                f"size {p.shared_size}", f"arg[{i}]")
        meta["layouts"].append(lay)
        if i != meta["owner"]:
            meta["aliases"].append(i)

    shared_plans: dict[str, SharedPlan] = {}
    for uid in shared_order:
        meta = shared_meta[uid]
        align = _cap_align(max((l.align for l in meta["layouts"]), default=1),
                           desc)
        cls = _shared_classify(meta["size"], meta["layouts"], desc)
        shared_plans[uid] = SharedPlan(
            uid, meta["owner"], tuple(meta["aliases"]), meta["size"], cls,
            align)

    rinfo = _return_info(ret_type, desc)
    regions: list[FrameRegion] = []
    cursor = 0
    if rinfo is not None and rinfo.indirect:
        regions.append(FrameRegion("return_area", 0, rinfo.area_size,
                                   "return-area"))
        cursor = rinfo.area_size

    rules = desc.int_regs
    next_int = 0
    next_float = 0
    placements: list[PlacementV2 | None] = [None] * len(params)
    save_entries: list[SaveEntry] = []
    save_needed = any(p.variadic for p in params)

    def emit_save(variadic: bool, kind: str, slot: int, i: int, eb: int):
        if not variadic:
            return
        cap = desc.save_int_slots if kind == "int" else desc.save_float_slots
        if slot >= cap:
            raise VariadicPromotionError(
                f"variadic {kind} register save area exhausted at slot {slot}",
                f"arg[{i}]")
        save_entries.append(SaveEntry(kind, slot, -1, i, eb))

    for i, p in enumerate(params):
        # 共享别名：不占任何资源
        if p.shared is not None and shared_plans[p.shared].owner != i:
            owner = shared_plans[p.shared].owner
            placements[i] = PlacementV2(
                i, "alias", (), (), (), None,
                layout_v2(p.type, desc).size, False, None, p.shared, owner)
            continue

        eff_type = p.type
        promoted = None
        if p.variadic:
            try:
                eff_type, promoted = _promote(p.type)
            except VariadicPromotionError as exc:
                raise type(exc)(exc.message, f"arg[{i}]") from None
        if p.shared is not None:
            sp = shared_plans[p.shared]
            lay_size, cls, unit_align = sp.size, sp.eightbytes, sp.align
        else:
            try:
                lay = layout_v2(eff_type, desc)
            except OverlappingBitFieldError as exc:
                raise type(exc)(exc.message, f"arg[{i}]") from None
            lay_size, unit_align = lay.size, lay.align
            cls = classify_v2(eff_type, desc)

        if not cls:
            placements[i] = PlacementV2(
                i, "none", (), (), (), None, 0, p.variadic, promoted, p.shared)
            continue

        if cls == (MEMORY,):
            off = _align_up(cursor, desc.stack_align)
            slot_size = _align_up(lay_size, desc.stack_align)
            regions.append(FrameRegion(
                f"arg[{i}]" + (f" shared:{p.shared}" if p.shared else ""),
                off, slot_size, "arg-slot", i))
            cursor = off + slot_size
            placements[i] = PlacementV2(
                i, "stack", cls, (), (), off, lay_size, p.variadic,
                promoted, p.shared)
            continue

        need_int = cls.count(INTEGER)
        need_float = cls.count(FLOAT)
        if (next_int + need_int <= rules.int_count
                and next_float + need_float <= rules.float_count):
            iregs: list[int] = []
            fregs: list[int] = []
            eb = 0
            for c in cls:
                if c == INTEGER:
                    slot = next_int
                    iregs.append(slot)
                    emit_save(p.variadic, "int", slot, i, eb)
                    next_int += 1
                else:
                    slot = next_float
                    fregs.append(slot)
                    emit_save(p.variadic, "float", slot, i, eb)
                    next_float += 1
                eb += 1
            placements[i] = PlacementV2(
                i, "registers", cls, tuple(iregs), tuple(fregs), None,
                lay_size, p.variadic, promoted, p.shared)
        else:
            off = _align_up(cursor, desc.stack_align)
            slot_size = _align_up(lay_size, desc.stack_align)
            regions.append(FrameRegion(
                f"arg[{i}]" + (f" shared:{p.shared}" if p.shared else ""),
                off, slot_size, "arg-slot", i))
            cursor = off + slot_size
            placements[i] = PlacementV2(
                i, "stack", cls, (), (), off, lay_size, p.variadic,
                promoted, p.shared)

    # 变参寄存器保存区（整数库 + 浮点库，槽号即寄存器号，基址在参数区之后）
    if save_needed:
        save_base = _align_up(cursor, desc.stack_align)
        save_int_size = desc.save_int_slots * 8
        ri = save_base
        regions.append(FrameRegion("save_int", ri, save_int_size, "save-int"))
        rf = ri + save_int_size
        save_float_size = desc.save_float_slots * 8
        regions.append(FrameRegion("save_float", rf, save_float_size,
                                   "save-float"))
        final_entries = [
            SaveEntry(e.kind, e.slot,
                      ri + e.slot * 8 if e.kind == "int" else rf + e.slot * 8,
                      e.arg_index, e.eightbyte)
            for e in sorted(save_entries, key=lambda e: (e.kind != "int",
                                                         e.slot))
        ]
        save_entries = final_entries
        cursor = rf + save_float_size

    frame_size = _align_up(cursor, desc.stack_align)
    return PlannedCall(
        desc, rinfo, tuple(placements), tuple(regions), tuple(save_entries),
        frame_size,
        tuple([None] * rules.int_count), tuple([None] * rules.float_count),
        tuple(shared_plans[u] for u in shared_order))


# ---------------------------------------------------------------------------
# v2 编解码器（接触值；与几何规划器分离）
#
# 约定（缺省固定）：
# * 填充字节编码时固定写 0（不读取、不比较填充字节）；
# * 联合未激活成员字节固定写 0；
# * 多字节标量按描述符字节序存放；位域位号相对“存储单元整数”的 LSB，
#   与字节序无关（单元整数本身按描述符字节序存放）；
# * 浮点叶子的 NaN 一律以 FloatBits(原始 64 位载荷) 表达，有限值为 float；
# * 变参实参先按原类型校验，再按提升类型编码。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 位/字节原语
# ---------------------------------------------------------------------------

def _check_int_value(value, size: int, signed: bool, path: str):
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidValueError(
            f"expected an int (bool is not accepted), got {value!r}", path)
    bits = size * 8
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    if not lo <= value <= hi:
        raise IntegerOverflowError(
            f"value {value} out of range [{lo}, {hi}] for "
            f"{'i' if signed else 'u'}{bits}", path)


def _unit_get(buf, byte: int, unit: int, endian: str) -> int:
    return int.from_bytes(bytes(buf[byte:byte + unit]), endian)


def _unit_set(buf, byte: int, unit: int, endian: str, value: int) -> None:
    buf[byte:byte + unit] = value.to_bytes(unit, endian)


def _put_field_bits(buf, byte: int, unit: int, endian: str,
                    lsb: int, width: int, pattern: int) -> None:
    """把 pattern 的低 width 位写入存储单元整数的 [lsb, lsb+width) 位。"""
    mask = (1 << width) - 1
    v = _unit_get(buf, byte, unit, endian)
    v = (v & ~(mask << lsb)) | ((pattern & mask) << lsb)
    _unit_set(buf, byte, unit, endian, v)


def _get_field_bits(buf, byte: int, unit: int, endian: str,
                    lsb: int, width: int) -> int:
    return (_unit_get(buf, byte, unit, endian) >> lsb) & ((1 << width) - 1)


def _scalar_bytes(t: Type, value, desc: ABIDescriptor, path: str) -> bytes:
    endian = desc.endian
    if isinstance(t, Int128Type):
        _check_int_value(value, 16, t.signed, path)
        return value.to_bytes(16, endian, signed=t.signed)
    if isinstance(t, IntType):
        _check_int_value(value, t.size, t.signed, path)
        return value.to_bytes(t.size, endian, signed=t.signed)
    if isinstance(t, PtrType):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidValueError(
                f"expected an int address (bool is not accepted), got {value!r}",
                path)
        if not 0 <= value <= (1 << 64) - 1:
            raise IntegerOverflowError(
                f"pointer value {value} out of range [0, 2**64 - 1]", path)
        return value.to_bytes(8, endian)
    if isinstance(t, (FloatType, Float32Type)):
        width = 64 if isinstance(t, FloatType) else 32
        nbytes = width // 8
        fmt_char = "d" if width == 64 else "f"
        if isinstance(value, FloatBits):
            if value.width != width:
                raise InvalidValueError(
                    f"FloatBits width {value.width} != {t.name} width {width}",
                    path)
            return value.pattern.to_bytes(nbytes, desc.endian)
        if isinstance(value, bool) or not isinstance(value, float):
            raise InvalidValueError(
                f"expected a float or FloatBits(NaN), got {value!r}", path)
        if math.isnan(value) or math.isinf(value):
            raise NonFiniteFloatError(
                f"non-finite float {value!r}: wrap NaN payloads in FloatBits; "
                "infinity is rejected", path)
        return struct.pack(desc.struct_prefix + fmt_char, value)
    raise InvalidTypeError(f"not a scalar {t!r}", path)  # pragma: no cover


def encode_v2(t: Type, value, desc: ABIDescriptor, path: str = "value",
              lay: LayoutV2 | None = None) -> bytes:
    """按描述符把规范值编码为字节（填充位固定 0）。"""
    _check_type(t)
    if lay is None:
        lay = layout_v2(t, desc)

    if isinstance(t, (IntType, Int128Type, PtrType, FloatType, Float32Type)):
        return _scalar_bytes(t, value, desc, path)

    if isinstance(t, ArrayType):
        if not isinstance(value, (list, tuple)):
            raise InvalidValueError(
                f"expected a list/tuple of {t.count} elements, got {value!r}",
                path)
        if len(value) != t.count:
            raise InvalidValueError(
                f"array length {len(value)} != declared count {t.count}", path)
        buf = bytearray(lay.size)
        el = layout_v2(t.elem, desc)
        for i, v in enumerate(value):
            chunk = encode_v2(t.elem, v, desc, f"{path}[{i}]", el)
            buf[i * el.size:(i + 1) * el.size] = chunk
        return bytes(buf)

    if isinstance(t, RecordType):
        if not isinstance(value, Mapping):
            raise InvalidValueError(
                f"expected a mapping of record fields, got {value!r}", path)
        expected: set[str] = set()
        for f in lay.fields:
            if f.kind == "member":
                expected.add(f.name)
            else:
                # 位域组：组内每个 BitSeg 名即一个位域名
                names = {s.name for s in f.segs}
                expected.update(names)
        missing = expected - set(value.keys())
        extra = set(value.keys()) - expected
        if missing:
            raise InvalidValueError(
                f"record {t.name!r} missing fields {sorted(missing)}", path)
        if extra:
            raise InvalidValueError(
                f"record {t.name!r} has unexpected fields {sorted(extra)}",
                path)
        buf = bytearray(lay.size)
        for f in lay.fields:
            if f.kind == "member":
                chunk = encode_v2(f.type, value[f.name], desc,
                                  f"{path}.{f.name}",
                                  layout_v2(f.type, desc))
                buf[f.offset:f.offset + f.size] = chunk
            else:
                # 位域组：按 BitField 声明取值并写入存储单元
                bf_decls = [x[1] for x in t.fields
                            if isinstance(x[1], BitField)]
                by_name = {b.name: b for b in bf_decls}
                written: set[str] = set()
                for s in f.segs:
                    bf = by_name[s.name]
                    if s.name in written:
                        continue
                    written.add(s.name)
                    raw = value[s.name]
                    if isinstance(raw, bool) or not isinstance(raw, int):
                        raise InvalidValueError(
                            f"bitfield {s.name!r} expects an int (bool rejected), "
                            f"got {raw!r}", f"{path}.{s.name}")
                    bits = bf.width
                    lo = -(1 << (bits - 1)) if bf.base.signed else 0
                    hi = (1 << (bits - 1)) - 1 if bf.base.signed else (1 << bits) - 1
                    if not lo <= raw <= hi:
                        raise IntegerOverflowError(
                            f"bitfield {s.name!r} value {raw} out of range "
                            f"[{lo}, {hi}] ({bits} bits)", f"{path}.{s.name}")
                    pattern = raw & ((1 << bits) - 1)
                    # 该位域可能被拆成两段（跨单元）
                    parts = [x for x in f.segs if x.name == s.name]
                    for ps in parts:
                        part = (pattern >> _low_offset(parts, ps)) & \
                               ((1 << ps.width) - 1)
                        _put_field_bits(buf, ps.byte, f.base_size,
                                        desc.endian, ps.lsb, ps.width, part)
        return bytes(buf)

    if isinstance(t, UnionType):
        if not isinstance(value, UnionValue):
            raise InvalidValueError(
                f"union {t.name!r} requires a UnionValue(tag, value), got "
                f"{value!r}", path)
        n = len(t.members)
        tag = value.tag
        if isinstance(tag, str):
            names = [m[0] for m in t.members]
            if tag not in names:
                raise InvalidUnionTagError(
                    f"union {t.name!r}: tag {tag!r} is not a member name "
                    f"{names}", path)
            idx = names.index(tag)
        elif isinstance(tag, bool) or not isinstance(tag, int):
            raise InvalidUnionTagError(
                f"union {t.name!r}: tag must be int index or member name, got "
                f"{tag!r}", path)
        elif not 0 <= tag < n:
            raise InvalidUnionTagError(
                f"union {t.name!r}: tag index {tag} out of range [0, {n})",
                path)
        else:
            idx = tag
        buf = bytearray(lay.size)
        mname, mtype = t.members[idx]
        if t.tag_type is not None:
            tag_bytes = _scalar_bytes(t.tag_type, idx, desc, f"{path}.$tag")
            buf[0:len(tag_bytes)] = tag_bytes
        mchunk = encode_v2(mtype, value.value, desc, f"{path}.{mname}")
        buf[lay.member_offset:lay.member_offset + len(mchunk)] = mchunk
        return bytes(buf)

    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def _low_offset(parts, ps) -> int:
    """位域被拆成多段时，段 ps 对应原始位型中的最低位偏移。"""
    ordered = sorted(parts, key=lambda x: (x.byte, x.lsb))
    total = 0
    for q in ordered:
        if q is ps:
            return total
        total += q.width
    return total  # pragma: no cover


def _decode_scalar(t: Type, data: bytes, desc: ABIDescriptor, path: str):
    if isinstance(t, Int128Type):
        return int.from_bytes(data, desc.endian, signed=t.signed)
    if isinstance(t, IntType):
        return int.from_bytes(data, desc.endian, signed=t.signed)
    if isinstance(t, PtrType):
        return int.from_bytes(data, desc.endian)
    if isinstance(t, Float32Type):
        pattern = int.from_bytes(data, desc.endian)
        exp = (pattern >> 23) & 0xFF
        mant = pattern & ((1 << 23) - 1)
        if exp == 0xFF and mant != 0:
            return FloatBits(pattern, 32)
        return struct.unpack(desc.struct_prefix + "f", data)[0]
    if isinstance(t, FloatType):
        pattern = int.from_bytes(data, desc.endian)
        exp = (pattern >> 52) & 0x7FF
        mant = pattern & ((1 << 52) - 1)
        if exp == 0x7FF and mant != 0:
            return FloatBits(pattern)      # NaN：逐位保留载荷
        return struct.unpack(desc.struct_prefix + "d", data)[0]
    raise InvalidTypeError(f"not a scalar {t!r}", path)  # pragma: no cover


def decode_v2(t: Type, data, desc: ABIDescriptor, path: str = "value",
              lay: LayoutV2 | None = None, tag_hint=None):
    """按描述符把字节解码为规范值（NaN 叶子还原为 FloatBits）。"""
    _check_type(t)
    if lay is None:
        lay = layout_v2(t, desc)
    if not isinstance(data, (bytes, bytearray)):
        raise InvalidValueError(f"expected bytes, got {data!r}", path)
    if len(data) != lay.size:
        raise InvalidValueError(
            f"byte length {len(data)} != layout size {lay.size} for {t.name}",
            path)

    if isinstance(t, (IntType, Int128Type, PtrType, FloatType, Float32Type)):
        return _decode_scalar(t, data, desc, path)

    if isinstance(t, ArrayType):
        el = layout_v2(t.elem, desc)
        step = el.size
        return [decode_v2(t.elem, data[i * step:(i + 1) * step], desc,
                          f"{path}[{i}]", el)
                for i in range(t.count)]

    if isinstance(t, RecordType):
        out = {}
        for f in lay.fields:
            if f.kind == "member":
                out[f.name] = decode_v2(
                    f.type, data[f.offset:f.offset + f.size], desc,
                    f"{path}.{f.name}", layout_v2(f.type, desc))
            else:
                bf_decls = [x[1] for x in t.fields
                            if isinstance(x[1], BitField)]
                by_name = {b.name: b for b in bf_decls}
                done: set[str] = set()
                for s in f.segs:
                    if s.name in done:
                        continue
                    done.add(s.name)
                    parts = sorted((x for x in f.segs if x.name == s.name),
                                   key=lambda x: (x.byte, x.lsb))
                    pattern = 0
                    shift = 0
                    for ps in parts:
                        part = _get_field_bits(
                            data, ps.byte, f.base_size,
                            desc.endian, ps.lsb, ps.width)
                        pattern |= part << shift
                        shift += ps.width
                    bf = by_name[s.name]
                    bits = bf.width
                    if bf.base.signed and (pattern >> (bits - 1)) & 1:
                        pattern -= 1 << bits
                    out[s.name] = pattern
        return out

    if isinstance(t, UnionType):
        if t.tag_type is not None:
            idx = int.from_bytes(data[0:t.tag_type.size], desc.endian,
                                 signed=False)
            if not 0 <= idx < len(t.members):
                raise InvalidUnionTagError(
                    f"union {t.name!r}: stored tag {idx} out of range "
                    f"[0, {len(t.members)})", path)
            mname, mtype = t.members[idx]
            mlay = layout_v2(mtype, desc)
            val = decode_v2(mtype,
                            data[lay.member_offset:lay.member_offset + mlay.size],
                            desc, f"{path}.{mname}", mlay)
            return UnionValue(idx, val)
        # 无标签联合：优先 tag_hint；否则要求严格重编码唯一可辨识
        if tag_hint is not None:
            idx = _resolve_union_hint(t, tag_hint, path)
            mname, mtype = t.members[idx]
            mlay = layout_v2(mtype, desc)
            return UnionValue(idx, decode_v2(
                mtype, data[lay.member_offset:lay.member_offset + mlay.size],
                desc, f"{path}.{mname}", mlay))
        candidates = []
        body = bytes(data[lay.member_offset:])
        for idx2, (mname, mtype) in enumerate(t.members):
            mlay = layout_v2(mtype, desc)
            try:
                val = decode_v2(mtype, body[:mlay.size], desc,
                                f"{path}.{mname}", mlay)
                if encode_v2(mtype, val, desc, f"{path}.{mname}", mlay) \
                        == bytes(body[:mlay.size]):
                    candidates.append((idx2, val))
            except ABIError:
                continue
        if len(candidates) == 1:
            return UnionValue(candidates[0][0], candidates[0][1])
        raise InvalidUnionTagError(
            f"untagged union {t.name!r} active member is not uniquely "
            f"identifiable from its bits ({len(candidates)} candidates); "
            "pass tag_hint", path)

    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def _resolve_union_hint(t: UnionType, hint, path: str) -> int:
    names = [m[0] for m in t.members]
    if isinstance(hint, str):
        if hint not in names:
            raise InvalidUnionTagError(
                f"union {t.name!r}: tag_hint {hint!r} not in {names}", path)
        return names.index(hint)
    if isinstance(hint, bool) or not isinstance(hint, int):
        raise InvalidUnionTagError(
            f"union {t.name!r}: tag_hint must be int/str, got {hint!r}", path)
    if not 0 <= hint < len(t.members):
        raise InvalidUnionTagError(
            f"union {t.name!r}: tag_hint {hint} out of range", path)
    return hint


# ---------------------------------------------------------------------------
# 帧编组 / 解组（逐位证据、共享存储合并、返回间接区、变参保存区）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceSeg:
    """一位段的来源证据：映像位置 -> 参数路径。

    loc: "ireg"/"freg"/"frame"；slot 为寄存器号或帧内字节偏移；
    byte/lsb/width 相对该位置（寄存器按 8 字节整词的 LSB 起算位号）。
    """

    loc: str
    slot: int
    byte: int
    lsb: int
    width: int
    path: str
    role: str


@dataclass(frozen=True)
class CallImageV2:
    plan: PlannedCall
    int_regs: tuple       # None 或无符号 64 位位型
    float_regs: tuple     # None、float 或 FloatBits（NaN 载荷逐位保留）
    frame: bytes          # 返回区 + 栈槽 + 保存区
    evidence: tuple[EvidenceSeg, ...]
    return_regs: tuple | None = None     # (int_patterns, float_patterns)
    return_area: bytes | None = None


def _active_segs(t: Type, value, desc: ABIDescriptor, path: str,
                 lay: LayoutV2, base: int = 0):
    """生成某值所有活动逻辑位段。

    产出元组 ``(unit_start, unit_size, lsb, width, role, cls, path)``：
    unit_start/unit_size 为存储单元在实参内的字节起点与大小；lsb 为相对
    单元整数 LSB 的逻辑位号（与字节序无关）。物理位由 _logical_runs 映射。
    """
    if isinstance(t, (IntType, Int128Type, PtrType)):
        n = (16 if isinstance(t, Int128Type)
             else 8 if isinstance(t, PtrType) else t.size)
        yield (base, n, 0, n * 8, "data", INTEGER, path)
    elif isinstance(t, (FloatType, Float32Type)):
        n = 8 if isinstance(t, FloatType) else 4
        yield (base, n, 0, n * 8, "float", FLOAT, path)
    elif isinstance(t, ArrayType):
        el = layout_v2(t.elem, desc)
        for i in range(t.count):
            yield from _active_segs(t.elem, value[i], desc, f"{path}[{i}]",
                                    el, base + i * el.size)
    elif isinstance(t, RecordType):
        for f in lay.fields:
            if f.kind == "member":
                sub = layout_v2(f.type, desc)
                yield from _active_segs(f.type, value[f.name], desc,
                                        f"{path}.{f.name}", sub,
                                        base + f.offset)
            else:
                for s in f.segs:
                    # 位域单元：组起点 + 段相对字节（向下取整到单元）
                    ustart = base + (s.byte // f.base_size) * f.base_size
                    lsb = (s.byte % f.base_size) * 8 + s.lsb
                    yield (ustart, f.base_size, lsb, s.width, "bits",
                           INTEGER, f"{path}.{s.name}")
    elif isinstance(t, UnionType):
        uv = value
        if t.tag_type is not None:
            n = t.tag_type.size
            yield (base, n, 0, n * 8, "tag", INTEGER, f"{path}.$tag")
        idx = uv.tag if isinstance(uv.tag, int) else \
            [m[0] for m in t.members].index(uv.tag)
        mname, mtype = t.members[idx]
        mlay = layout_v2(mtype, desc)
        yield from _active_segs(mtype, uv.value, desc, f"{path}.{mname}",
                                mlay, base + lay.member_offset)


def _logical_runs(unit_start: int, unit_size: int, lsb: int, width: int,
                  endian: str):
    """逻辑位区间 -> 连续物理 (字节偏移, 字节内lsb, 宽) 段（逐位合并）。

    小端：物理字节 = unit_start + lsb//8；大端：物理字节镜像到单元另一端。
    """
    runs: list = []
    for k in range(width):
        logical = lsb + k
        if endian == LITTLE:
            pb = unit_start + logical // 8
        else:
            pb = unit_start + (unit_size - 1 - logical // 8)
        pl = logical % 8
        if runs and runs[-1][0] == pb and runs[-1][1] + runs[-1][2] == pl:
            ob, ol, ow = runs[-1]
            runs[-1] = (ob, ol, ow + 1)
        else:
            runs.append((pb, pl, 1))
    return runs


def _chunk_of_phys(pb: int, pl: int, endian: str):
    """实参内物理 (字节, 位) -> (8字节块号, 寄存器整词内lsb)。"""
    chunk = pb // 8
    word_lsb = (7 - pb % 8) * 8 + pl if endian == BIG else (pb % 8) * 8 + pl
    return chunk, word_lsb



def _promote_value(t: Type, v, desc: ABIDescriptor, path: str):
    """规范值随变参默认提升一起提升（f32 先舍入到 binary32 再展宽）。"""
    if isinstance(t, Float32Type):
        if isinstance(v, FloatBits):
            return v.widen_to_64()
        rounded = struct.unpack(">f", struct.pack(">f", v))[0]
        return rounded
    return v


def _demote_value(t: Type, v, path: str):
    """解组时把提升类型的规范值还原回原变参类型的规范值。"""
    if isinstance(t, IntType) and t.size in (1, 2):
        bits = t.size * 8
        pat = v & ((1 << bits) - 1)
        if t.signed and (pat >> (bits - 1)) & 1:
            return pat - (1 << bits)
        return pat
    if isinstance(t, Float32Type):
        if isinstance(v, FloatBits):
            sign = v.sign
            quiet = (v.pattern >> 51) & 1
            payload = v.pattern & ((1 << 22) - 1)
            pat = (sign << 31) | (0xFF << 23) | (quiet << 22) | payload
            return FloatBits(pat, 32)
        return struct.unpack(">f", struct.pack(">f", v))[0]
    return v


def pack_call_v2(params, values, desc: ABIDescriptor,
                 ret_type: Type | None = None, ret_value=None,
                 shared_units=()) -> CallImageV2:
    """按描述符编组一次调用。

    三阶段、全部成功才产出：1) 每个实参离线编码；2) 共享存储单元逐位合并
    （冲突整体失败）；3) 一次性写入帧与寄存器。values 中共享视图位置传
    None；NaN 叶子用 FloatBits；ret_value 非 None 时同时编返回值。
    """
    if not isinstance(params, (list, tuple)):
        raise InvalidValueError("params must be a list/tuple of Param")
    if not isinstance(values, (list, tuple)):
        raise InvalidValueError("values must be a list/tuple")
    if len(params) != len(values):
        raise InvalidValueError(
            f"params/values length mismatch: {len(params)} vs {len(values)}")
    plan = plan_call(params, desc, ret_type)

    # ---- 阶段 1：离线编码（不接触输出）----
    encoded: dict[int, bytes] = {}
    eff_values: dict[int, object] = {}
    for i, p in enumerate(params):
        path = f"arg[{i}]"
        if p.shared is not None:
            continue
        v = values[i]
        eff_type = p.type
        if p.variadic:
            try:
                eff_type, _ = _promote(p.type)
            except VariadicPromotionError as exc:
                raise type(exc)(exc.message, path) from None
            v = _promote_value(p.type, v, desc, path)
        eff_values[i] = v
        encoded[i] = encode_v2(eff_type, v, desc, path)

    # ---- 阶段 2：共享单元逐位合并 ----
    unit_origins: dict[str, list] = {}     # id -> [逻辑位段, ...]
    for sp in plan.shared_units:
        su = next((u for u in shared_units if u.unit_id == sp.unit_id), None)
        if su is None:
            raise InvalidValueError(
                f"no SharedUnit value supplied for unit {sp.unit_id!r}",
                f"arg[{sp.owner}]")
        if not isinstance(su, SharedUnit):
            raise InvalidValueError("shared_units entries must be SharedUnit")
        if su.size != sp.size:
            raise InvalidValueError(
                f"SharedUnit {sp.unit_id!r} size {su.size} != planned {sp.size}",
                f"arg[{sp.owner}]")
        buf = bytearray(sp.size)
        occupied = bytearray(sp.size)
        origins: list = []
        for j in (sp.owner, *sp.aliases):
            vp = params[j]
            vname = vp.type.name
            if vname not in su.views:
                raise InvalidValueError(
                    f"SharedUnit {sp.unit_id!r} missing view {vname!r}",
                    f"arg[{j}]")
            vpath = f"arg[{j}](shared:{sp.unit_id})"
            vlay = layout_v2(vp.type, desc)
            view_val = su.views[vname]
            vbuf = encode_v2(vp.type, view_val, desc, vpath, vlay)
            # 视图按其各叶子存储单元解码为无符号整数，逻辑位与字节序无关
            for (ustart, usize, lsb, width, role, cls,
                 segpath) in _active_segs(vp.type, view_val, desc, vpath,
                                          vlay):
                origins.append((ustart, usize, lsb, width, role, cls, segpath))
                unit_uint = int.from_bytes(
                    vbuf[ustart:ustart + usize], desc.endian)
                for k in range(width):
                    logical = lsb + k
                    if desc.endian == LITTLE:
                        bi = ustart + logical // 8
                    else:
                        bi = ustart + (usize - 1 - logical // 8)
                    li = logical % 8
                    new_bit = (unit_uint >> logical) & 1
                    if occupied[bi] >> li & 1:
                        if (buf[bi] >> li) & 1 != new_bit:
                            raise OverlappingStorageError(
                                f"shared unit {sp.unit_id!r}: conflicting bit "
                                f"{bi * 8 + li} across overlapping views",
                                vpath)
                    else:
                        occupied[bi] |= 1 << li
                        buf[bi] |= new_bit << li
        unit_origins[sp.unit_id] = origins
        encoded[sp.owner] = bytes(buf)
        eff_values[sp.owner] = su.views[params[sp.owner].type.name]

    # ---- 阶段 3：落帧 / 寄存器 / 证据 ----
    rules = desc.int_regs
    int_regs: list = [None] * rules.int_count
    float_regs: list = [None] * rules.float_count
    frame = bytearray(plan.frame_size)
    evidence: list[EvidenceSeg] = []

    for i, p in enumerate(params):
        pl = plan.placements[i]
        if pl.kind in ("none", "alias"):
            continue
        data = encoded[i]
        if p.shared is not None:
            active = unit_origins[p.shared]   # 已是 7 元组逻辑位段
        else:
            etype = _promote(p.type)[0] if p.variadic else p.type
            elay = layout_v2(etype, desc)
            active = list(_active_segs(etype, eff_values[i], desc,
                                       f"arg[{i}]", elay))

        if pl.kind == "stack":
            frame[pl.stack_offset:pl.stack_offset + pl.size] = data
            for (ustart, usize, lsb, width, role, cls,
                 spath) in active:
                for (pb, pbit, take) in _logical_runs(
                        ustart, usize, lsb, width, desc.endian):
                    evidence.append(EvidenceSeg(
                        "frame", pl.stack_offset,
                        pl.stack_offset + pb, pbit, take, spath, role))
        else:
            ii = iter(pl.int_regs)
            fi = iter(pl.float_regs)
            int_slots = list(pl.int_regs)
            float_slots = list(pl.float_regs)
            i_used = f_used = 0
            for k, c in enumerate(pl.eightbytes):
                chunk = data[8 * k:8 * k + 8]
                c8 = chunk.ljust(8, b"\x00")  # 块左对齐；字节序在块内
                pat = int.from_bytes(c8, desc.endian)
                if c == INTEGER:
                    int_regs[int_slots[i_used]] = pat
                    i_used += 1
                else:
                    exp, mant = (pat >> 52) & 0x7FF, pat & ((1 << 52) - 1)
                    float_regs[float_slots[f_used]] = (
                        FloatBits(pat) if exp == 0x7FF and mant != 0
                        else struct.unpack(desc.struct_prefix + "d", c8)[0])
                    f_used += 1
            for (ustart, usize, lsb, width, role, cls,
                 spath) in active:
                for (pb, pbit, take) in _logical_runs(
                        ustart, usize, lsb, width, desc.endian):
                    chunk, wlsb = _chunk_of_phys(pb, pbit, desc.endian)
                    c = pl.eightbytes[chunk]
                    if c == INTEGER:
                        slot = pl.int_regs[pl.eightbytes[:chunk].count(INTEGER)]
                        evidence.append(EvidenceSeg(
                            "ireg", slot, 0, wlsb, take, spath, role))
                    else:
                        slot = pl.float_regs[pl.eightbytes[:chunk].count(FLOAT)]
                        evidence.append(EvidenceSeg(
                            "freg", slot, 0, wlsb, take, spath, role))

    # 变参寄存器保存区
    for e in plan.save_entries:
        if e.kind == "int":
            pat = int_regs[e.slot]
            if pat is not None:
                frame[e.offset:e.offset + 8] = pat.to_bytes(8, desc.endian)
        else:
            fv = float_regs[e.slot]
            if fv is not None:
                pat = fv.pattern if isinstance(fv, FloatBits) \
                    else struct.unpack(">Q", struct.pack(">d", fv))[0]
                frame[e.offset:e.offset + 8] = pat.to_bytes(8, desc.endian)
        evidence.append(EvidenceSeg(
            "frame", e.offset, e.offset, 0, 64, f"arg[{e.arg_index}]",
            "save-copy"))

    return_regs = None
    return_area = None
    rinfo = plan.return_info
    if rinfo is not None and ret_value is not None:
        return_regs, return_area = _encode_return(rinfo, ret_value, desc, frame)

    return CallImageV2(plan, tuple(int_regs), tuple(float_regs),
                       bytes(frame), tuple(evidence),
                       return_regs, return_area)


def _encode_return(rinfo: ReturnInfo, value, desc: ABIDescriptor,
                   frame: bytearray):
    data = encode_v2(rinfo.ret_type, value, desc, "return")
    if rinfo.indirect:
        frame[0:len(data)] = data
        return None, bytes(frame[0:rinfo.area_size])
    ipats: list[int] = []
    fpats: list[int] = []
    for k, c in enumerate(rinfo.eightbytes):
        chunk = data[8 * k:8 * k + 8]
        c8 = chunk.ljust(8, b"\x00")
        (ipats if c == INTEGER else fpats).append(
            int.from_bytes(c8, desc.endian))
    return (tuple(ipats), tuple(fpats)), None


def _slot_pattern(slot_val, desc: ABIDescriptor) -> int:
    if isinstance(slot_val, FloatBits):
        return slot_val.pattern
    if isinstance(slot_val, float):
        return struct.unpack(">Q", struct.pack(">d", slot_val))[0]
    return slot_val


def unpack_call_v2(params, image: CallImageV2, desc: ABIDescriptor):
    """从映像重建规范实参值（pack_call_v2 的逆；NaN 还原为 FloatBits）。"""
    if not isinstance(image, CallImageV2):
        raise InvalidValueError("expected a CallImageV2")
    if len(params) != len(image.plan.placements):
        raise InvalidValueError(
            f"{len(params)} params but image has "
            f"{len(image.plan.placements)} placements")

    # 先还原每个非别名实参的原始字节（属主共享字节缓存）
    raw: dict[int, bytes] = {}
    for i, p in enumerate(params):
        pl = image.plan.placements[i]
        if pl.kind == "none":
            raw[i] = b""
        elif pl.kind == "alias":
            continue
        elif pl.kind == "stack":
            raw[i] = image.frame[pl.stack_offset:pl.stack_offset + pl.size]
        elif pl.kind == "registers":
            chunks: list[bytes] = []
            ii = iter(pl.int_regs)
            fi = iter(pl.float_regs)
            for c in pl.eightbytes:
                if c == INTEGER:
                    pat = image.int_regs[next(ii)]
                else:
                    pat = _slot_pattern(image.float_regs[next(fi)], desc)
                chunks.append(pat.to_bytes(8, desc.endian))
            raw[i] = b"".join(chunks)[:pl.size]
        else:  # pragma: no cover
            raise InvalidValueError(f"unknown placement kind {pl.kind!r}")

    values: list = [None] * len(params)
    for i, p in enumerate(params):
        pl = image.plan.placements[i]
        path = f"arg[{i}]"
        if pl.kind == "alias":
            owner = pl.alias_of
            etype = p.type
            elay = layout_v2(etype, desc)
            values[i] = decode_v2(etype, raw[owner][:elay.size], desc, path,
                                  elay, p.tag_hint)
            continue
        if p.variadic:
            etype, _ = _promote(p.type)
            elay = layout_v2(etype, desc)
            v = decode_v2(etype, raw[i], desc, path, elay)
            values[i] = _demote_value(p.type, v, path)
        else:
            etype = p.type
            elay = layout_v2(etype, desc)
            values[i] = decode_v2(etype, raw[i], desc, path, elay, p.tag_hint)
    return values


def unpack_return_v2(ret_type: Type, image: CallImageV2, desc: ABIDescriptor):
    """从映像取回返回值（间接区或返回寄存器）。"""
    rinfo = image.plan.return_info
    if rinfo is None:
        raise InvalidValueError("call was planned without a return type")
    if rinfo.indirect:
        lay = layout_v2(ret_type, desc)
        return decode_v2(ret_type, image.frame[0:lay.size], desc, "return", lay)
    data = bytearray()
    ip = iter(rinfo.int_regs)
    fp = iter(rinfo.float_regs)
    patterns_i = image.return_regs[0]
    patterns_f = image.return_regs[1]
    ii = fi = 0
    for c in rinfo.eightbytes:
        pat = patterns_i[ii] if c == INTEGER else patterns_f[fi]
        if c == INTEGER:
            ii += 1
        else:
            fi += 1
        data += pat.to_bytes(8, desc.endian)
    lay = layout_v2(ret_type, desc)
    return decode_v2(ret_type, bytes(data[:lay.size]), desc, "return", lay)


# ---------------------------------------------------------------------------
# 证书：独立校验，不信任求解器自报的布尔标志
#
# LayoutCertificate：重算类型的几何（尺寸/对齐/逐位覆盖），核对
#   1) 每个声明位段落在类型字节范围内；2) 位段两两不重叠；
#   3) 覆盖的位恰为所有叶子位（填充位不要求覆盖）。
# FrameCertificate：重放规划，核对
#   1) 帧区域互不重叠、不越 frame_size；2) 返回区在偏移 0 且满足返回对齐；
#   3) 每个实参的栈槽对齐；4) 保存区槽位与寄存器分配一致；
#   5) 反字节序重放后，返回值/实参对齐约束仍满足。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutCertificate:
    type: Type
    desc: ABIDescriptor
    size: int
    align: int
    leaf_bits: int             # 所有叶子（非填充）位数
    covered_bits: int
    overlaps: tuple
    out_of_range: tuple
    valid: bool

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise InvalidTypeError(
                f"layout certificate invalid for {self.type.name}: "
                f"overlaps={self.overlaps}, out_of_range={self.out_of_range}, "
                f"covered {self.covered_bits}/{self.leaf_bits} bits")


@dataclass(frozen=True)
class FrameCertificate:
    plan: PlannedCall
    region_overlaps: tuple
    frame_size_ok: bool
    return_ok: bool
    stack_align_ok: bool
    save_ok: bool
    flipped_align_ok: bool
    valid: bool

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise InvalidTypeError(
                "frame certificate invalid: "
                f"region_overlaps={self.region_overlaps}, "
                f"frame_size_ok={self.frame_size_ok}, return_ok={self.return_ok}, "
                f"stack_align_ok={self.stack_align_ok}, save_ok={self.save_ok}, "
                f"flipped_align_ok={self.flipped_align_ok}")


def _leaf_bit_count(t: Type, desc: ABIDescriptor) -> int:
    """独立统计叶子位数（不读 LayoutV2.segs，直接走类型结构）。"""
    if isinstance(t, (IntType, Int128Type, PtrType, FloatType, Float32Type)):
        if isinstance(t, Int128Type):
            return 128
        if isinstance(t, PtrType):
            return 64
        if isinstance(t, FloatType):
            return 64
        if isinstance(t, Float32Type):
            return 32
        return t.size * 8
    if isinstance(t, ArrayType):
        return t.count * _leaf_bit_count(t.elem, desc)
    if isinstance(t, RecordType):
        total = 0
        for _n, f in t.fields:
            if isinstance(f, BitField):
                total += f.width
            else:
                total += _leaf_bit_count(f, desc)
        return total
    if isinstance(t, UnionType):
        tag = t.tag_type.size * 8 if t.tag_type is not None else 0
        return tag + max((_leaf_bit_count(m, desc) for _n, m in t.members),
                         default=0)
    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def _cert_collect(t: Type, desc: ABIDescriptor, base: int, domain: tuple,
                  uid: list, out: list):
    """递归收集带“冲突域”的位区间。

    同一域内区间不得重叠；联合的每个成员进入不同的子域（成员间天然重叠，
    合法），标签留在父域（始终存在）。out 追加 (lo, hi, name, domain)。
    """
    lay = layout_v2(t, desc)
    if isinstance(t, (IntType, Int128Type, PtrType, FloatType, Float32Type)):
        n = (16 if isinstance(t, Int128Type) else
             8 if isinstance(t, (PtrType, FloatType)) else
             4 if isinstance(t, Float32Type) else t.size)
        out.append((base * 8, base * 8 + n * 8, t.name, domain))
    elif isinstance(t, ArrayType):
        el = layout_v2(t.elem, desc)
        for i in range(t.count):
            _cert_collect(t.elem, desc, base + i * el.size, domain, uid, out)
    elif isinstance(t, RecordType):
        for f in lay.fields:
            if f.kind == "bits":
                for s in f.segs:
                    out.append((base * 8 + s.byte * 8 + s.lsb,
                                base * 8 + s.byte * 8 + s.lsb + s.width,
                                s.name, domain))
            else:
                _cert_collect(f.type, desc, base + f.offset, domain, uid, out)
    elif isinstance(t, UnionType):
        if t.tag_type is not None:
            n = t.tag_type.size
            out.append((base * 8, base * 8 + n * 8, "$tag", domain))
        my = uid[0]
        uid[0] += 1
        for idx, (_n, mt) in enumerate(t.members):
            _cert_collect(mt, desc, base + lay.member_offset,
                          domain + ((my, idx),), uid, out)


def certify_layout(t: Type, desc: ABIDescriptor,
                   lay: LayoutV2 | None = None) -> LayoutCertificate:
    """独立核对类型几何位段。

    冲突域规则：同一域内位段必须落在类型范围内且两两不重叠；联合的不同
    成员处于不同域（彼此可重叠）。覆盖口径：标签位 + 最大成员叶子位。
    """
    if lay is None:
        lay = layout_v2(t, desc)
    intervals: list = []
    _cert_collect(t, desc, 0, (), [0], intervals)

    total_bits = lay.size * 8
    out: list[str] = []
    overlaps: list[str] = []
    for lo, hi, name, dom in intervals:
        if lo < 0 or hi > total_bits:
            out.append(f"{name}@{lo}:{hi}>{total_bits}")
    by_domain: dict = {}
    for lo, hi, name, dom in intervals:
        by_domain.setdefault(dom, []).append((lo, hi, name))
    for dom, ivs in by_domain.items():
        ivs.sort()
        for k in range(1, len(ivs)):
            if ivs[k][0] < ivs[k - 1][1]:
                overlaps.append(f"{ivs[k-1][2]}<->{ivs[k][2]}@{dom}")

    leaf = _leaf_bit_count(t, desc)
    covered = len({(lo + k, dom) for lo, hi, _n, dom in intervals
                   for k in range(hi - lo)})

    def _has_union(x: Type) -> bool:
        if isinstance(x, UnionType):
            return True
        if isinstance(x, ArrayType):
            return _has_union(x.elem)
        if isinstance(x, RecordType):
            return any(not isinstance(f, BitField) and _has_union(f)
                       for _n, f in x.fields)
        return False

    # 非联合类型：覆盖必须恰为全部叶子位；含联合时成员跨域，严格的“每位
    # 恰有一值”口径改由 verify_image 对具体活动值逐位核对。
    cover_ok = (covered == leaf) if not _has_union(t) else True
    valid = not out and not overlaps and cover_ok
    return LayoutCertificate(t, desc, lay.size, lay.align, leaf, covered,
                             tuple(overlaps), tuple(out), valid)


def _regions_disjoint(regions) -> list:
    rs = sorted((r.offset, r.offset + r.size, r.name) for r in regions)
    bad = []
    for k in range(1, len(rs)):
        if rs[k][0] < rs[k - 1][1]:
            bad.append(f"{rs[k - 1][2]}<->{rs[k][2]}")
    return bad


@dataclass(frozen=True)
class ImageCertificate:
    """映像逐位内容证书（独立重编码核对，不信任求解器标志）。"""

    plan: PlannedCall
    args_total: int
    args_verified: int
    data_bits: int           # 实参活动位总数（不含保存副本）
    evidence_bits: int       # 证据声明的实参数据位数
    evidence_conflicts: tuple
    content_mismatches: tuple
    save_ok: bool
    flipped_ok: bool
    return_ok: bool
    valid: bool

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise InvalidTypeError(
                "image certificate invalid: "
                f"args {self.args_verified}/{self.args_total}, "
                f"bits {self.evidence_bits}/{self.data_bits}, "
                f"conflicts={self.evidence_conflicts}, "
                f"content_mismatches={self.content_mismatches}, "
                f"save_ok={self.save_ok}, flipped_ok={self.flipped_ok}, "
                f"return_ok={self.return_ok}")


def _image_param_bytes(image: CallImageV2, i: int, desc: ABIDescriptor) -> bytes:
    pl = image.plan.placements[i]
    if pl.kind == "stack":
        return image.frame[pl.stack_offset:pl.stack_offset + pl.size]
    chunks: list[bytes] = []
    ii = iter(pl.int_regs)
    fi = iter(pl.float_regs)
    for c in pl.eightbytes:
        if c == INTEGER:
            pat = image.int_regs[next(ii)]
        else:
            v = image.float_regs[next(fi)]
            pat = v.pattern if isinstance(v, FloatBits) \
                else struct.unpack(">Q", struct.pack(">d", v))[0]
        chunks.append(pat.to_bytes(8, desc.endian))
    return b"".join(chunks)[:pl.size]


def verify_image(params, values, image: CallImageV2, desc: ABIDescriptor,
                 shared_units=(), ret_value=None) -> ImageCertificate:
    """独立核对映像：逐实参重编码比对 + 证据位一一覆盖 + 保存区 + 双端重放。"""
    plan = image.plan

    # 1) 期望字节：独立重编码每个实参（变参按提升类型）
    expected: dict[int, bytes] = {}
    data_bits = 0
    n_verifiable = 0
    for i, p in enumerate(params):
        pl = plan.placements[i]
        if pl.kind in ("none", "alias"):
            continue
        n_verifiable += 1
        if p.shared is not None:
            sp = next(u for u in plan.shared_units if u.unit_id == p.shared)
            su = next(u for u in shared_units if u.unit_id == p.shared)
            buf = bytearray(sp.size)
            occ = bytearray(sp.size)
            for j in (sp.owner, *sp.aliases):
                vp = params[j]
                vlay = layout_v2(vp.type, desc)
                vb = encode_v2(vp.type, su.views[vp.type.name], desc,
                               f"arg[{j}]", vlay)
                for (ustart, usize, lsb, width, _r, _c,
                     _p) in _active_segs(vp.type, su.views[vp.type.name],
                                         desc, f"arg[{j}]", vlay):
                    unit_uint = int.from_bytes(
                        vb[ustart:ustart + usize], desc.endian)
                    for k in range(width):
                        logical = lsb + k
                        bi = (ustart + logical // 8) if desc.endian == LITTLE \
                            else ustart + (usize - 1 - logical // 8)
                        bl = logical % 8
                        nb = (unit_uint >> logical) & 1
                        if occ[bi] >> bl & 1:
                            assert ((buf[bi] >> bl) & 1) == nb
                        else:
                            occ[bi] |= 1 << bl
                            buf[bi] |= nb << bl
            expected[i] = bytes(buf)
            data_bits += sum(bin(x).count("1") for x in occ)
            continue
        etype = _promote(p.type)[0] if p.variadic else p.type
        v = _promote_value(p.type, values[i], desc, f"arg[{i}]") \
            if p.variadic else values[i]
        b = encode_v2(etype, v, desc, f"arg[{i}]")
        expected[i] = b
        elay = layout_v2(etype, desc)
        data_bits += sum(w for _us, _uz, _l, w, _r, _c, _p
                         in _active_segs(etype, v, desc, f"arg[{i}]", elay))

    # 2) 逐实参字节精确比对（值位 + 零填充；不以“填充相等”为唯一标准）
    mismatches: list[str] = []
    for i, b in expected.items():
        actual = _image_param_bytes(image, i, desc)
        if actual != b:
            mismatches.append(f"arg[{i}]")

    # 3) 证据位：每个数据物理位必须被覆盖；共享视图可对同一位做多路径声明，
    #    此时只核对各位读出的值一致（合法别名），不因路径不同而判冲突。
    claimed: dict = {}
    conflicts: list[str] = []
    for e in image.evidence:
        if e.role == "save-copy":
            continue
        for k in range(e.width):
            if e.loc == "frame":
                key = ("frame", e.byte * 8 + e.lsb + k)
            else:
                key = (e.loc, e.slot, e.lsb + k)
            if e.loc == "frame":
                ab = e.byte + (e.lsb + k) // 8
                bl = (e.lsb + k) % 8
                bitval = (image.frame[ab] >> bl) & 1
            elif e.loc == "ireg":
                bitval = (image.int_regs[e.slot] >> (e.lsb + k)) & 1
            else:
                v = image.float_regs[e.slot]
                pat = v.pattern if isinstance(v, FloatBits) \
                    else struct.unpack(">Q", struct.pack(">d", v))[0]
                bitval = (pat >> (e.lsb + k)) & 1
            if key in claimed and claimed[key][0] != bitval:
                conflicts.append(f"{key} {e.path}")
                continue
            if key not in claimed:
                claimed[key] = (bitval, [e.path])
            else:
                claimed[key][1].append(e.path)
    evidence_bits = len(claimed)

    # 4) 保存区：保存槽内容等于对应变参寄存器整词
    save_ok = True
    for ent in plan.save_entries:
        word = image.frame[ent.offset:ent.offset + 8]
        if ent.kind == "int":
            want = image.int_regs[ent.slot]
            if want is None or word != want.to_bytes(8, desc.endian):
                save_ok = False
        else:
            v = image.float_regs[ent.slot]
            if v is None:
                save_ok = False
            else:
                pat = v.pattern if isinstance(v, FloatBits) \
                    else struct.unpack(">Q", struct.pack(">d", v))[0]
                if word != pat.to_bytes(8, desc.endian):
                    save_ok = False

    # 5) 反字节序重放：几何对齐 + 实参/返回可解码一致
    flipped = desc.flipped()
    flipped_ok = True
    rtype = plan.return_info.ret_type if plan.return_info else None
    try:
        pf = plan_call(params, flipped, rtype)
        if any(r.kind == "arg-slot"
               and r.offset % flipped.stack_align != 0
               for r in pf.regions):
            flipped_ok = False
        if pf.return_info and pf.return_info.indirect \
                and pf.return_info.area_offset != 0:
            flipped_ok = False
    except ABIError:
        flipped_ok = False

    # 6) 返回值
    return_ok = True
    rinfo = plan.return_info
    if rinfo is not None and ret_value is not None:
        got = unpack_return_v2(rinfo.ret_type, image, desc)
        if got != ret_value:
            return_ok = False

    valid = (not mismatches and not conflicts and save_ok and flipped_ok
             and return_ok and evidence_bits == data_bits
             and len(claimed) == data_bits and n_verifiable >= 0)
    return ImageCertificate(plan, len(params), n_verifiable, data_bits,
                            evidence_bits, tuple(conflicts),
                            tuple(mismatches), save_ok, flipped_ok,
                            return_ok, valid)


def certify_frame(params, plan: PlannedCall, desc: ABIDescriptor,
                  image: CallImageV2 | None = None) -> FrameCertificate:
    """独立核对帧规划：区域不重叠、返回区、栈对齐、保存区、反字节序重放。"""
    region_bad = _regions_disjoint(plan.regions)
    frame_ok = all(r.offset + r.size <= plan.frame_size for r in plan.regions)

    rinfo = plan.return_info
    if rinfo is None or not rinfo.indirect:
        return_ok = True
    else:
        ret_lay = layout_v2(rinfo.ret_type, desc)
        return_ok = (rinfo.area_offset == 0
                     and rinfo.area_size >= ret_lay.size
                     and rinfo.area_size % desc.stack_align == 0)

    stack_ok = all(
        r.kind != "arg-slot" or r.offset % desc.stack_align == 0
        for r in plan.regions)

    var_reg_slots = {"int": set(), "float": set()}
    for pl in plan.placements:
        if pl.kind == "registers" and pl.variadic:
            for eb, c in enumerate(pl.eightbytes):
                if c == INTEGER:
                    var_reg_slots["int"].add(pl.int_regs[
                        pl.eightbytes[:eb].count(INTEGER)])
                else:
                    var_reg_slots["float"].add(pl.float_regs[
                        pl.eightbytes[:eb].count(FLOAT)])
    seen_slots = {"int": set(), "float": set()}
    save_ok = True
    for e in plan.save_entries:
        seen_slots[e.kind].add(e.slot)
        expect_base = next((r.offset for r in plan.regions
                            if r.kind == f"save-{e.kind}"), None)
        if expect_base is None or e.offset != expect_base + e.slot * 8:
            save_ok = False
    if seen_slots != var_reg_slots:
        save_ok = False
    if any(p.variadic for p in params) and not any(
            r.kind == "save-int" for r in plan.regions):
        save_ok = False

    flipped = desc.flipped()
    plan_f = plan_call(params, flipped,
                       plan.return_info.ret_type if plan.return_info else None)
    flipped_ok = all(
        r.kind != "arg-slot" or r.offset % flipped.stack_align == 0
        for r in plan_f.regions)
    if plan_f.return_info and plan_f.return_info.indirect:
        flipped_ok = flipped_ok and plan_f.return_info.area_offset == 0
    flipped_ok = flipped_ok and all(
        pl.kind in ("registers", "stack", "none", "alias")
        for pl in plan_f.placements)

    valid = (not region_bad and frame_ok and return_ok and stack_ok
             and save_ok and flipped_ok)
    return FrameCertificate(plan, tuple(region_bad), frame_ok, return_ok,
                           stack_ok, save_ok, flipped_ok, valid)



# ---------------------------------------------------------------------------
# A -> B 调用映像桥接：先解码为规范值再按目标布局编组
# ---------------------------------------------------------------------------

def bridge_call(src_image: CallImageV2, params, src_desc: ABIDescriptor,
                dst_desc: ABIDescriptor, ret_type: Type | None = None,
                shared_units=(), dst_buffer: bytearray | None = None,
                ret_value=None):
    """把 src_desc 下的调用映像转换为 dst_desc 下的映像。

    流程：解码源映像到规范值（NaN 还原为 FloatBits 逐位携带）-> 按目标
    描述符重新规划并编码 -> 一次性产出。任何阶段失败抛 BridgeError 且不写
    dst_buffer。dst_buffer 给定时直接写入（长度不足抛错），返回同一对象。
    """
    try:
        canonical = unpack_call_v2(params, src_image, src_desc)
    except ABIError as exc:
        raise BridgeError(
            f"source decode failed: {exc.message}", exc.path,
            "tag-out-of-range" if isinstance(exc, InvalidUnionTagError)
            else "bridge-failed") from None

    # 共享视图：解码源单元后重新逻辑化（用属主原始字节再按每个视图解码），
    # 然后交给目标侧合并；这里直接用解码值重组 SharedUnit。
    rebuilt_shared = list(shared_units)
    if not shared_units and src_image.plan.shared_units:
        # 无外部 SharedUnit 输入时，从源映像自动重建视图值
        owner_bytes: dict[int, bytes] = {}
        for i, p in enumerate(params):
            pl = src_image.plan.placements[i]
            if pl.kind == "stack":
                owner_bytes[i] = src_image.frame[
                    pl.stack_offset:pl.stack_offset + pl.size]
            elif pl.kind == "registers":
                chunks = []
                ii = iter(pl.int_regs); fi = iter(pl.float_regs)
                for c in pl.eightbytes:
                    if c == INTEGER:
                        pat = src_image.int_regs[next(ii)]
                    else:
                        fv = src_image.float_regs[next(fi)]
                        pat = fv.pattern if isinstance(fv, FloatBits) \
                            else struct.unpack(">Q", struct.pack(">d", fv))[0]
                    chunks.append(pat.to_bytes(8, src_desc.endian))
                owner_bytes[i] = b"".join(chunks)[:pl.size]
        for sp in src_image.plan.shared_units:
            raw = owner_bytes[sp.owner]
            views = {}
            for j in (sp.owner, *sp.aliases):
                vp = params[j]
                vlay = layout_v2(vp.type, src_desc)
                views[vp.type.name] = decode_v2(
                    vp.type, raw[:vlay.size], src_desc,
                    f"arg[{j}]", vlay, vp.tag_hint)
            rebuilt_shared.append(SharedUnit(sp.unit_id, sp.size, views))

    vals: list = [None] * len(params)
    for i, p in enumerate(params):
        if p.shared is not None:
            vals[i] = None
        else:
            vals[i] = canonical[i]

    # 从源映像取回返回值的规范值（间接区或返回寄存器），供目标侧重编码。
    canonical_ret = None
    src_rinfo = src_image.plan.return_info
    if ret_type is not None and src_rinfo is not None:
        if src_rinfo.indirect:
            rlay = layout_v2(ret_type, src_desc)
            canonical_ret = decode_v2(
                ret_type, src_image.frame[0:rlay.size], src_desc,
                "return", rlay)
        elif src_image.return_regs is not None:
            # return_regs 的两个元组均为无符号 64 位位型
            data = bytearray()
            ii = fi = 0
            for c in src_rinfo.eightbytes:
                if c == INTEGER:
                    pat = src_image.return_regs[0][ii]; ii += 1
                else:
                    pat = src_image.return_regs[1][fi]; fi += 1
                data += pat.to_bytes(8, src_desc.endian)
            rlay = layout_v2(ret_type, src_desc)
            canonical_ret = decode_v2(ret_type, bytes(data[:rlay.size]),
                                     src_desc, "return", rlay)
    if ret_value is None:
        ret_value = canonical_ret

    try:
        dst = pack_call_v2(params, vals, dst_desc, ret_type, ret_value,
                           rebuilt_shared)
    except OverlappingStorageError as exc:
        raise BridgeError(
            f"shared storage conflict under {dst_desc.name}: {exc.message}",
            exc.path, "overlap-conflict") from None
    except InvalidUnionTagError as exc:
        raise BridgeError(f"union tag invalid: {exc.message}", exc.path,
                          "tag-out-of-range") from None
    except VariadicPromotionError as exc:
        raise BridgeError(
            f"variadic value not representable under {dst_desc.name}: "
            f"{exc.message}", exc.path, "unrepresentable-promotion") from None
    except IntegerOverflowError as exc:
        raise BridgeError(f"value out of range under {dst_desc.name}: "
                          f"{exc.message}", exc.path, "value-range") from None

    if dst_buffer is not None:
        if len(dst_buffer) < len(dst.frame):
            raise BridgeError(
                f"destination buffer {len(dst_buffer)} bytes too small for "
                f"{len(dst.frame)}-byte frame")
        dst_buffer[:len(dst.frame)] = dst.frame
    return dst


def bridge_call_into(dst_buffer: bytearray, src_image: CallImageV2, params,
                     src_desc: ABIDescriptor, dst_desc: ABIDescriptor,
                     ret_type: Type | None = None, shared_units=(),
                     ret_value=None) -> CallImageV2:
    """显式“写入给定缓冲区”的桥接；失败保证不修改 dst_buffer 任何字节。"""
    # 先在临时空间完成全部工作，成功才拷贝 -> 杜绝部分改写。
    return bridge_call(src_image, params, src_desc, dst_desc, ret_type,
                       shared_units, dst_buffer, ret_value)
