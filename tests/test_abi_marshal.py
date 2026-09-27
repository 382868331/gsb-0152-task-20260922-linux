"""abi_marshal 的单元测试。

独立性说明：本文件中的期望值来自两类独立来源——
1. 手工计算的硬编码字面量（偏移、size、分类、寄存器值）；
2. 本文件内独立编写的标量布局参考函数 ``ref_record_layout``，它按题面
   对齐规则独立实现，不调用被测核心。
pack/unpack 往返一致仅作为补充测试。
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abi_marshal as am


# ---------------------------------------------------------------------------
# 独立参考：仅按题面规则（自然对齐、上限 8、尾部填充到 align）独立实现，
# 不调用 abi_marshal 的任何函数。
# ---------------------------------------------------------------------------

def ref_record_layout(spec):
    """spec: [(field_name, size, align), ...] -> (offsets, size, align)。"""
    offsets = []
    off = 0
    align = 1
    for _name, size, al in spec:
        off = -(-off // al) * al  # 向上对齐（独立写法）
        offsets.append(off)
        off += size
        if al > align:
            align = al
    size = -(-off // align) * align
    return offsets, size, align


def rec(name, fields):
    r = am.RecordType(name)
    r.define(fields)
    return r


class TestScalarLayout(unittest.TestCase):
    def test_scalar_sizes_aligns(self):
        # 硬编码字面量期望
        for t, sz in [(am.INT8, 1), (am.UINT8, 1), (am.INT16, 2), (am.UINT16, 2),
                      (am.INT32, 4), (am.UINT32, 4), (am.INT64, 8), (am.UINT64, 8),
                      (am.DOUBLE, 8), (am.PTR, 8)]:
            lay = am.layout(t)
            self.assertEqual(lay.size, sz, t.name)
            self.assertEqual(lay.align, sz, t.name)
            self.assertEqual(lay.fields, ())

    def test_invalid_int_size_rejected(self):
        with self.assertRaises(am.InvalidTypeError):
            am.IntType(3, True)
        with self.assertRaises(am.InvalidTypeError):
            am.IntType(16, False)


class TestRecordLayoutOffsets(unittest.TestCase):
    def test_offsets_match_independent_reference(self):
        # 混合标量记录：逐字段偏移与独立参考逐一比对
        fields = [("a", am.UINT8), ("b", am.UINT32), ("c", am.DOUBLE),
                      ("d", am.UINT16), ("e", am.UINT8)]
        r = rec("Mix", fields)
        lay = am.layout(r)
        spec = [(n, am.layout(t).size, am.layout(t).align) for n, t in fields]
        # 独立参考只消费标量的 (size, align) 字面量，不消费核心布局结果：
        spec = [("a", 1, 1), ("b", 4, 4), ("c", 8, 8), ("d", 2, 2), ("e", 1, 1)]
        ref_offsets, ref_size, ref_align = ref_record_layout(spec)
        self.assertEqual([f.offset for f in lay.fields], ref_offsets)
        self.assertEqual(lay.size, ref_size)
        self.assertEqual(lay.align, ref_align)
        # 同时核对硬编码字面量
        self.assertEqual([f.offset for f in lay.fields], [0, 4, 8, 16, 18])
        self.assertEqual(lay.size, 24)
        self.assertEqual(lay.align, 8)

    def test_tail_padding(self):
        # {u64, u8}: 末尾填充到 align=8 -> size 16
        r = rec("Pad1", [("a", am.UINT64), ("b", am.UINT8)])
        lay = am.layout(r)
        self.assertEqual([f.offset for f in lay.fields], [0, 8])
        self.assertEqual(lay.size, 16)
        self.assertEqual(lay.align, 8)
        # {u8, u8}: 无填充，size 2 align 1
        r2 = rec("Pad2", [("a", am.UINT8), ("b", am.UINT8)])
        lay2 = am.layout(r2)
        self.assertEqual(lay2.size, 2)
        self.assertEqual(lay2.align, 1)
        # {u8, u32}: 间隙填充 + 尾填充 -> size 8
        r3 = rec("Pad3", [("a", am.UINT8), ("b", am.UINT32)])
        lay3 = am.layout(r3)
        self.assertEqual([f.offset for f in lay3.fields], [0, 4])
        self.assertEqual(lay3.size, 8)

    def test_nested_record_offsets(self):
        inner = rec("Inner", [("x", am.UINT8), ("y", am.DOUBLE)])
        outer = rec("Outer", [("tag", am.UINT16), ("in", inner), ("z", am.UINT8)])
        lay = am.layout(outer)
        # inner: x@0, y@8, size 16 align 8
        self.assertEqual([f.offset for f in am.layout(inner).fields], [0, 8])
        self.assertEqual(am.layout(inner).size, 16)
        # outer: tag@0, pad到8, in@8, z@24, size 32
        self.assertEqual([f.offset for f in lay.fields], [0, 8, 24])
        self.assertEqual(lay.size, 32)


class TestArrayLayout(unittest.TestCase):
    def test_array_contiguous_full_element_size(self):
        # 元素按完整 size 连续排列（含元素自身尾填充）
        elem = rec("E", [("a", am.UINT8), ("b", am.UINT32)])  # size 8 align 4
        arr = am.ArrayType(elem, 3)
        lay = am.layout(arr)
        self.assertEqual(lay.size, 24)
        self.assertEqual(lay.align, 4)

    def test_nested_array_in_record(self):
        r = rec("WithArr", [("vals", am.ArrayType(am.INT16, 3)),
                            ("flag", am.UINT8)])
        lay = am.layout(r)
        # vals@0 占 6 字节, flag@6, align 2 -> size 8
        self.assertEqual([f.offset for f in lay.fields], [0, 6])
        self.assertEqual(lay.size, 8)
        self.assertEqual(lay.align, 2)

    def test_array_of_array(self):
        arr = am.ArrayType(am.ArrayType(am.UINT8, 3), 2)
        lay = am.layout(arr)
        self.assertEqual(lay.size, 6)
        self.assertEqual(lay.align, 1)

    def test_bad_array_count_rejected(self):
        with self.assertRaises(am.InvalidTypeError):
            am.ArrayType(am.UINT8, -1)
        with self.assertRaises(am.InvalidTypeError):
            am.ArrayType(am.UINT8, am.MAX_ARRAY_COUNT + 1)
        with self.assertRaises(am.InvalidTypeError):
            am.ArrayType(am.UINT8, True)


class TestEmptyRecord(unittest.TestCase):
    def test_empty_record_convention(self):
        empty = rec("Empty", [])
        lay = am.layout(empty)
        self.assertEqual(lay.size, 0)
        self.assertEqual(lay.align, 1)
        self.assertEqual(am.classify(empty), ())

    def test_empty_record_field_does_not_shift_next(self):
        empty = rec("Empty2", [])
        r = rec("HasEmpty", [("e", empty), ("v", am.UINT32)])
        lay = am.layout(r)
        # 空记录 align=1 size=0，后续字段偏移不受影响
        self.assertEqual([f.offset for f in lay.fields], [0, 0])
        self.assertEqual(lay.size, 4)

    def test_empty_record_consumes_nothing(self):
        empty = rec("Empty3", [])
        img = am.pack_call([(empty, {})])
        self.assertEqual(img.placements[0].kind, "none")
        self.assertEqual(img.int_regs, (None,) * am.INT_REG_COUNT)
        self.assertEqual(img.float_regs, (None,) * am.FLOAT_REG_COUNT)
        self.assertEqual(img.stack, b"")
        self.assertEqual(am.unpack_call([empty], img), [{}])


class TestClassify(unittest.TestCase):
    def test_mixed_class_aggregate(self):
        # {f64, i32}: 块0 FLOAT, 块1 INTEGER
        r = rec("MF", [("d", am.DOUBLE), ("i", am.INT32)])
        self.assertEqual(am.classify(r), (am.FLOAT, am.INTEGER))
        # {i32, f64}: 块0 同时含整数与浮点字节 -> INTEGER；块1 仅 f64 -> FLOAT
        r2 = rec("MI", [("i", am.INT32), ("d", am.DOUBLE)])
        self.assertEqual(am.classify(r2), (am.INTEGER, am.FLOAT))
        # {f64, f64}
        r3 = rec("FF", [("a", am.DOUBLE), ("b", am.DOUBLE)])
        self.assertEqual(am.classify(r3), (am.FLOAT, am.FLOAT))
        # 指针按整数类
        r4 = rec("PD", [("p", am.PTR), ("d", am.DOUBLE)])
        self.assertEqual(am.classify(r4), (am.INTEGER, am.FLOAT))

    def test_scalar_classify(self):
        self.assertEqual(am.classify(am.UINT8), (am.INTEGER,))
        self.assertEqual(am.classify(am.DOUBLE), (am.FLOAT,))
        self.assertEqual(am.classify(am.PTR), (am.INTEGER,))

    def test_over_16_bytes_is_memory(self):
        r = rec("Big", [("a", am.DOUBLE), ("b", am.DOUBLE), ("c", am.DOUBLE)])
        self.assertEqual(am.layout(r).size, 24)
        self.assertEqual(am.classify(r), (am.MEMORY,))
        # 恰好 16 字节仍走寄存器分类
        r16 = rec("Exact16", [("a", am.DOUBLE), ("b", am.DOUBLE)])
        self.assertEqual(am.classify(r16), (am.FLOAT, am.FLOAT))
        # {u64, u8} size=16（尾填充后）-> 两块，块1 含 u8 -> INTEGER
        rpad = rec("PadCls", [("a", am.UINT64), ("b", am.UINT8)])
        self.assertEqual(am.classify(rpad), (am.INTEGER, am.INTEGER))


class TestPackCallRegisters(unittest.TestCase):
    def test_scalar_register_assignment(self):
        img = am.pack_call([(am.UINT64, 0x1122334455667788),
                            (am.DOUBLE, 1.5),
                            (am.INT32, -7)])
        self.assertEqual(img.int_regs[0], 0x1122334455667788)
        self.assertEqual(img.float_regs[0], 1.5)
        # i32 占块内低 4 字节，高 4 字节为零填充
        self.assertEqual(img.int_regs[1], (-7) & 0xFFFFFFFF)
        self.assertIsNone(img.int_regs[2])
        self.assertIsNone(img.float_regs[1])
        self.assertEqual(img.stack, b"")
        kinds = [p.kind for p in img.placements]
        self.assertEqual(kinds, ["registers"] * 3)

    def test_mixed_aggregate_register_packing(self):
        # {u8=0x11, u8=0x22} -> 单块 INTEGER，小端值 0x2211
        r = rec("TwoBytes", [("a", am.UINT8), ("b", am.UINT8)])
        img = am.pack_call([(r, {"a": 0x11, "b": 0x22})])
        self.assertEqual(img.placements[0].kind, "registers")
        self.assertEqual(img.placements[0].int_regs, (0,))
        self.assertEqual(img.int_regs[0], 0x2211)

    def test_mixed_class_aggregate_split_registers(self):
        # {f64, i32}: 块0 FLOAT -> 浮点寄存器，块1 INTEGER -> 整数寄存器
        r = rec("Split", [("d", am.DOUBLE), ("i", am.INT32)])
        img = am.pack_call([(r, {"d": 2.5, "i": 0x2A})])
        p = img.placements[0]
        self.assertEqual(p.kind, "registers")
        self.assertEqual(p.float_regs, (0,))
        self.assertEqual(p.int_regs, (0,))
        self.assertEqual(img.float_regs[0], 2.5)
        self.assertEqual(img.int_regs[0], 0x2A)

    def test_whole_fallback_when_int_registers_short(self):
        # 4 个 i64 占满整数寄存器；{i64,i64} 需 2 个整数寄存器 -> 整体走栈，
        # 且不消耗寄存器：其后的 f64 仍能拿到浮点寄存器 0。
        rr = rec("TwoI64", [("a", am.INT64), ("b", am.INT64)])
        args = ([(am.INT64, 1), (am.INT64, 2), (am.INT64, 3), (am.INT64, 4)]
                + [(rr, {"a": 5, "b": 6}), (am.DOUBLE, 9.5)])
        img = am.pack_call(args)
        self.assertEqual([img.int_regs[i] for i in range(4)], [1, 2, 3, 4])
        p = img.placements[4]
        self.assertEqual(p.kind, "stack")
        self.assertEqual(p.int_regs, ())
        self.assertEqual(p.stack_offset, 0)
        # 栈上小端映像 = 5,6
        self.assertEqual(img.stack[:16],
                         (5).to_bytes(8, "little") + (6).to_bytes(8, "little"))
        # 后续浮点仍进寄存器（回退未消耗任何寄存器）
        self.assertEqual(img.placements[5].kind, "registers")
        self.assertEqual(img.float_regs[0], 9.5)

    def test_whole_fallback_when_float_registers_short(self):
        ff = rec("TwoF64", [("a", am.DOUBLE), ("b", am.DOUBLE)])
        args = [(am.DOUBLE, 1.0), (am.DOUBLE, 2.0),
                (ff, {"a": 3.0, "b": 4.0}), (am.INT64, 7)]
        img = am.pack_call(args)
        self.assertEqual(img.float_regs, (1.0, 2.0))
        p = img.placements[2]
        self.assertEqual(p.kind, "stack")
        self.assertEqual(p.float_regs, ())
        # 回退不消耗整数寄存器，后续 i64 拿到整数寄存器 0
        self.assertEqual(img.placements[3].kind, "registers")
        self.assertEqual(img.int_regs[0], 7)

    def test_partial_fit_not_allowed(self):
        # 只剩 1 个整数寄存器时，需 2 块的聚合必须整体走栈（不可拆半）。
        r = rec("Need2", [("a", am.INT64), ("b", am.INT64)])
        args = [(am.INT64, 1), (am.INT64, 2), (am.INT64, 3),
                (r, {"a": 8, "b": 9})]
        img = am.pack_call(args)
        self.assertEqual(img.placements[3].kind, "stack")
        self.assertEqual(img.int_regs[3], None)  # 第 4 个整数寄存器未被消耗

    def test_memory_aggregate_goes_to_stack(self):
        big = rec("Big3", [("a", am.DOUBLE), ("b", am.DOUBLE), ("c", am.DOUBLE)])
        img = am.pack_call([(big, {"a": 1.0, "b": 2.0, "c": 3.0})])
        p = img.placements[0]
        self.assertEqual(p.kind, "stack")
        self.assertEqual(p.eightbytes, (am.MEMORY,))
        self.assertEqual(p.stack_offset, 0)
        self.assertEqual(p.size, 24)
        self.assertEqual(img.stack[:24],
                         struct.pack("<ddd", 1.0, 2.0, 3.0))
        self.assertEqual(len(img.stack), 32)  # 尾部补齐到 16 的倍数

    def test_stack_slots_16_byte_aligned(self):
        a1 = am.ArrayType(am.UINT8, 20)   # MEMORY, size 20
        a2 = am.ArrayType(am.UINT8, 17)   # MEMORY, size 17
        img = am.pack_call([(a1, list(range(20))), (a2, list(range(17)))])
        self.assertEqual(img.placements[0].stack_offset, 0)
        self.assertEqual(img.placements[1].stack_offset, 32)
        self.assertEqual(img.stack[0:20], bytes(range(20)))
        self.assertEqual(img.stack[32:49], bytes(range(17)))


class TestValueValidation(unittest.TestCase):
    def test_integer_overflow_rejected(self):
        with self.assertRaises(am.IntegerOverflowError):
            am.encode(am.UINT8, 256)
        with self.assertRaises(am.IntegerOverflowError):
            am.encode(am.UINT8, -1)
        with self.assertRaises(am.IntegerOverflowError):
            am.encode(am.INT8, 128)
        with self.assertRaises(am.IntegerOverflowError):
            am.encode(am.INT8, -129)
        # 边界值合法
        self.assertEqual(am.encode(am.INT8, 127), b"\x7f")
        self.assertEqual(am.encode(am.INT8, -128), b"\x80")
        self.assertEqual(am.encode(am.UINT64, 2**64 - 1), b"\xff" * 8)

    def test_bool_rejected_as_integer(self):
        with self.assertRaises(am.InvalidValueError):
            am.encode(am.UINT8, True)
        with self.assertRaises(am.InvalidValueError):
            am.encode(am.PTR, False)

    def test_pointer_range(self):
        self.assertEqual(am.encode(am.PTR, 0xDEADBEEF),
                         bytes.fromhex("efbeadde00000000"))
        with self.assertRaises(am.IntegerOverflowError):
            am.encode(am.PTR, 2**64)

    def test_non_finite_float_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(am.NonFiniteFloatError):
                am.encode(am.DOUBLE, bad)

    def test_int_not_accepted_as_float(self):
        # 拒绝 int 以避免 >= 2**53 时的静默精度损失
        with self.assertRaises(am.InvalidValueError):
            am.encode(am.DOUBLE, 3)

    def test_array_length_mismatch(self):
        arr = am.ArrayType(am.INT16, 3)
        with self.assertRaises(am.InvalidValueError):
            am.encode(arr, [1, 2])
        with self.assertRaises(am.InvalidValueError):
            am.encode(arr, [1, 2, 3, 4])

    def test_record_field_mismatch(self):
        r = rec("Rec", [("a", am.UINT8), ("b", am.UINT8)])
        with self.assertRaises(am.InvalidValueError):
            am.encode(r, {"a": 1})
        with self.assertRaises(am.InvalidValueError):
            am.encode(r, {"a": 1, "b": 2, "c": 3})

    def test_error_path_is_locatable(self):
        inner = rec("In", [("v", am.UINT8)])
        r = rec("Out", [("items", am.ArrayType(inner, 2))])
        with self.assertRaises(am.ABIError) as ctx:
            am.encode(r, {"items": [{"v": 1}, {"v": 300}]})
        self.assertEqual(ctx.exception.path, "value.items[1].v")
        # pack_call 的参数级路径
        with self.assertRaises(am.ABIError) as ctx2:
            am.pack_call([(am.UINT8, 1), (am.UINT8, 999)])
        self.assertEqual(ctx2.exception.path, "arg[1]")


class TestRecursiveTypes(unittest.TestCase):
    def test_direct_recursion_rejected(self):
        a = am.RecordType("A")
        a.define([("self", a)])
        with self.assertRaises(am.RecursiveTypeError):
            am.layout(a)

    def test_indirect_recursion_rejected(self):
        a = am.RecordType("A2")
        b = am.RecordType("B2")
        a.define([("b", b)])
        b.define([("a", a)])
        with self.assertRaises(am.RecursiveTypeError):
            am.layout(a)
        with self.assertRaises(am.RecursiveTypeError):
            am.classify(a)

    def test_recursion_through_array_rejected(self):
        a = am.RecordType("A3")
        arr = am.ArrayType(a, 2)
        a.define([("kids", arr)])
        with self.assertRaises(am.RecursiveTypeError):
            am.layout(a)

    def test_undefined_record_rejected(self):
        a = am.RecordType("Undef")
        with self.assertRaises(am.InvalidTypeError):
            am.layout(a)

    def test_redefine_rejected(self):
        a = rec("Once", [("x", am.UINT8)])
        with self.assertRaises(am.InvalidTypeError):
            a.define([("y", am.UINT8)])

    def test_duplicate_field_rejected(self):
        a = am.RecordType("Dup")
        with self.assertRaises(am.InvalidTypeError):
            a.define([("x", am.UINT8), ("x", am.UINT16)])


class TestRoundTrip(unittest.TestCase):
    """往返一致性（补充性质，不作为正确性的唯一依据）。"""

    def test_roundtrip_mixed_call(self):
        empty = rec("E0", [])
        inner = rec("In2", [("x", am.UINT8), ("y", am.DOUBLE)])
        pkt = rec("Pkt", [("hd", inner),
                          ("samples", am.ArrayType(am.INT16, 4)),
                          ("e", empty)])
        args = [
            (am.UINT64, 0xCAFEBABE),
            (am.DOUBLE, -0.25),
            (pkt, {"hd": {"x": 7, "y": 3.5},
                   "samples": [-1, 2, -3, 4], "e": {}}),
            (am.ArrayType(am.UINT8, 20), list(range(20))),
            (empty, {}),
            (am.PTR, 0x1000),
        ]
        types = [t for t, _ in args]
        values = [v for _, v in args]
        img = am.pack_call(args)
        self.assertEqual(am.unpack_call(types, img), values)

    def test_roundtrip_after_fallback(self):
        rr = rec("RR", [("a", am.INT64), ("b", am.INT64)])
        args = ([(am.INT64, i) for i in range(4)]
                + [(rr, {"a": -5, "b": 6}), (am.DOUBLE, 1.25)])
        types = [t for t, _ in args]
        values = [v for _, v in args]
        img = am.pack_call(args)
        self.assertEqual(am.unpack_call(types, img), values)

    def test_unpack_rejects_truncated_stack(self):
        big = am.ArrayType(am.UINT8, 24)
        img = am.pack_call([(big, list(range(24)))])
        bad = am.CallImage(img.int_regs, img.float_regs, img.stack[:8],
                           img.placements)
        with self.assertRaises(am.InvalidValueError):
            am.unpack_call([big], bad)


class TestEncodeBytes(unittest.TestCase):
    def test_scalar_encoding_little_endian(self):
        self.assertEqual(am.encode(am.UINT16, 0x1234), b"\x34\x12")
        self.assertEqual(am.encode(am.INT32, -2),
                         (-2).to_bytes(4, "little", signed=True))
        self.assertEqual(am.encode(am.DOUBLE, 1.0), struct.pack("<d", 1.0))

    def test_record_encoding_with_padding(self):
        r = rec("Enc", [("a", am.UINT8), ("b", am.UINT32)])
        data = am.encode(r, {"a": 0xAA, "b": 0x01020304})
        self.assertEqual(len(data), 8)
        self.assertEqual(data, bytes([0xAA, 0, 0, 0, 4, 3, 2, 1]))


class TestPackReturnRegisters(unittest.TestCase):
    def test_scalar_and_small_array_returns(self):
        img = am.pack_return(am.INT32, -7)
        self.assertEqual(img.kind, "registers")
        self.assertEqual(img.eightbytes, (am.INTEGER,))
        self.assertEqual(img.int_regs[0], (-7) & 0xFFFFFFFF)
        self.assertEqual(am.unpack_return(am.INT32, img), -7)

        arr = am.ArrayType(am.UINT8, 7)
        img2 = am.pack_return(arr, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual((img2.kind, img2.size), ("registers", 7))
        self.assertEqual(img2.int_regs[0], 0x07060504030201)
        self.assertEqual(am.unpack_return(arr, img2), [1, 2, 3, 4, 5, 6, 7])

    def test_mixed_nested_return_uses_both_register_classes(self):
        inner = rec("RetInner", [("x", am.UINT8), ("y", am.UINT8)])
        mixed = rec("RetMixed", [("d", am.DOUBLE), ("tag", inner)])
        lay = am.layout(mixed)
        self.assertEqual(lay.size, 16)
        value = {"d": -2.5, "tag": {"x": 0xAA, "y": 0xBB}}
        img = am.pack_return(mixed, value)
        self.assertEqual(img.kind, "registers")
        self.assertEqual(img.eightbytes, (am.FLOAT, am.INTEGER))
        self.assertEqual(img.float_regs[0], -2.5)
        self.assertEqual(img.int_regs[0], 0xBBAA)
        self.assertEqual(am.unpack_return(mixed, img), value)

    def test_empty_record_return_consumes_nothing(self):
        empty = rec("RetEmpty", [])
        img = am.pack_return(empty, {})
        self.assertEqual(img.kind, "none")
        self.assertEqual(img.eightbytes, ())
        self.assertEqual(img.int_regs, (None,) * am.INT_REG_COUNT)
        self.assertEqual(img.float_regs, (None,) * am.FLOAT_REG_COUNT)
        self.assertEqual(am.unpack_return(empty, img), {})


class TestMemoryReturn(unittest.TestCase):
    def test_nested_record_and_array_rebuilt_from_buffer(self):
        inner = rec("BigInner", [("a", am.UINT8), ("b", am.UINT32)])
        big = rec("BigReturn", [("items", am.ArrayType(inner, 3)),
                                 ("count", am.UINT32)])
        lay = am.layout(big)
        self.assertEqual((lay.size, lay.align), (28, 4))
        self.assertEqual(am.classify(big), (am.MEMORY,))

        value = {
            "items": [{"a": 1, "b": 0x04030201},
                      {"a": 2, "b": 0x08070605},
                      {"a": 3, "b": 0x0C0B0A09}],
            "count": 3,
        }
        buf = am.ResultBuffer(big, 0x1000)
        img = am.pack_return(big, value, buf)
        self.assertEqual(img.kind, "memory")
        self.assertEqual(img.eightbytes, (am.MEMORY,))
        self.assertEqual(img.result_address, 0x1000)
        self.assertEqual(img.int_regs[0], 0x1000)
        self.assertEqual(img.float_regs, (None, None))
        self.assertEqual(img.size, 28)
        expected = bytes([
            1, 0, 0, 0, 1, 2, 3, 4,
            2, 0, 0, 0, 5, 6, 7, 8,
            3, 0, 0, 0, 9, 10, 11, 12,
            3, 0, 0, 0,
        ])
        self.assertEqual(buf.contents, expected)
        self.assertEqual(am.unpack_return(big, img, buf), value)

    def test_sret_address_is_first_argument_and_shifts_later_args(self):
        big = am.ArrayType(am.UINT8, 20)
        buf = am.ResultBuffer(big, 0x2000)
        args = [(am.INT64, 10), (am.INT64, 20), (am.INT64, 30),
                (am.INT64, 40), (am.DOUBLE, 1.25)]
        img = am.pack_call(args, return_type=big, result_buffer=buf)

        self.assertEqual(img.sret_pointer.kind, "registers")
        self.assertEqual(img.sret_pointer.int_regs, (0,))
        self.assertEqual(img.result_address, 0x2000)
        self.assertEqual([p.kind for p in img.placements],
                         ["registers", "registers", "registers", "stack",
                          "registers"])
        self.assertEqual(img.int_regs, (0x2000, 10, 20, 30))
        self.assertEqual(img.stack[:8], (40).to_bytes(8, "little"))
        self.assertEqual(img.float_regs[0], 1.25)
        self.assertEqual(am.unpack_call([t for t, _ in args], img),
                         [10, 20, 30, 40, 1.25])

    def test_result_buffer_validation(self):
        big = am.ArrayType(am.UINT8, 20)
        small = am.ArrayType(am.UINT8, 8)
        with self.assertRaises(am.InvalidTypeError):
            am.ResultBuffer(small)
        with self.assertRaises(am.InvalidValueError):
            am.ResultBuffer(big, 0x1004)
        with self.assertRaises(am.IntegerOverflowError):
            am.ResultBuffer(big, 2**64)
        with self.assertRaises(am.InvalidValueError):
            am.pack_call([], return_type=big)
        buf = am.ResultBuffer(big)
        with self.assertRaises(am.InvalidValueError):
            am.pack_call([], return_type=small, result_buffer=buf)
        with self.assertRaises(am.InvalidValueError):
            am.pack_return(small, [0] * 8, buf)

    def test_unpack_rejects_address_mismatch(self):
        big = am.ArrayType(am.UINT8, 20)
        buf = am.ResultBuffer(big, 0x3000)
        img = am.pack_return(big, list(range(20)), buf)
        other = am.ResultBuffer(big, 0x4000)
        with self.assertRaises(am.InvalidValueError):
            am.unpack_return(big, img, other)


if __name__ == "__main__":
    unittest.main()
