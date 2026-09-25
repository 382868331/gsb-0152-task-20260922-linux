"""abi_marshal 返回值编组的单元测试。

独立性说明：期望值来自手工计算的硬编码字面量（字节串、寄存器值、偏移），
或由测试手工构造 ReturnImage/CallImage 后单向读取——不依赖
pack_return/unpack_return 往返自证。往返一致仅作补充。
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abi_marshal as am


def rec(name, fields):
    r = am.RecordType(name)
    r.define(fields)
    return r


# 复用的类型
MIXED = rec("RetMixed", [("weight", am.DOUBLE), ("id", am.INT32)])
# weight@0(8) id@8(4) size=16 align=8 -> (FLOAT, INTEGER)
BIG24 = rec("RetBig24", [("a", am.DOUBLE), ("b", am.DOUBLE), ("c", am.DOUBLE)])
# size=24 > 16 -> indirect
NESTED = rec("RetNested", [("m", am.ArrayType(am.ArrayType(am.INT16, 2), 2)),
                           ("x", am.DOUBLE), ("tag", am.UINT8)])
# m@0(8) x@8(8) tag@16(1) 尾填充7 -> size=24 align=8 -> indirect
EMPTY = rec("RetEmpty", [])


class TestReturnPlan(unittest.TestCase):
    def test_scalar_plans(self):
        self.assertEqual(am.plan_return(am.UINT8),
                         am.ReturnPlan("registers", (am.INTEGER,), 1, 1))
        self.assertEqual(am.plan_return(am.INT64),
                         am.ReturnPlan("registers", (am.INTEGER,), 8, 8))
        self.assertEqual(am.plan_return(am.DOUBLE),
                         am.ReturnPlan("registers", (am.FLOAT,), 8, 8))
        self.assertEqual(am.plan_return(am.PTR),
                         am.ReturnPlan("registers", (am.INTEGER,), 8, 8))

    def test_empty_record_no_return_registers(self):
        self.assertEqual(am.plan_return(EMPTY), am.ReturnPlan("none", (), 0, 1))

    def test_small_aggregate_registers(self):
        self.assertEqual(am.plan_return(MIXED),
                         am.ReturnPlan("registers", (am.FLOAT, am.INTEGER),
                                       16, 8))
        # 恰好 16 字节仍走返回寄存器
        r16 = rec("RetExact16", [("a", am.UINT64), ("b", am.UINT8)])
        self.assertEqual(am.plan_return(r16).kind, "registers")
        self.assertEqual(am.plan_return(r16).eightbytes,
                         (am.INTEGER, am.INTEGER))

    def test_large_aggregate_indirect(self):
        self.assertEqual(am.plan_return(BIG24),
                         am.ReturnPlan("indirect", (am.MEMORY,), 24, 8))
        self.assertEqual(am.plan_return(NESTED).kind, "indirect")
        # 17 字节数组同样间接
        self.assertEqual(
            am.plan_return(am.ArrayType(am.UINT8, 17)).kind, "indirect")


class TestPackReturnRegisters(unittest.TestCase):
    def test_mixed_int_float_record(self):
        img = am.pack_return(MIXED, {"weight": 0.5, "id": 42})
        self.assertEqual(img.plan.kind, "registers")
        # 块0 FLOAT -> 浮点返回寄存器0；块1 INTEGER -> 整数返回寄存器0
        self.assertEqual(img.float_regs, (0.5, None))
        self.assertEqual(img.int_regs, (42, None))
        self.assertEqual(img.memory, b"")

    def test_scalar_return_zero_padded_chunk(self):
        # i32=-7 占块内低4字节，高4字节零填充（不符号扩展到8字节）
        img = am.pack_return(am.INT32, -7)
        self.assertEqual(img.int_regs, ((-7) & 0xFFFFFFFF, None))
        img2 = am.pack_return(am.DOUBLE, -0.25)
        self.assertEqual(img2.float_regs, (-0.25, None))
        self.assertEqual(img2.int_regs, (None, None))

    def test_two_registers_per_class(self):
        ff = rec("RetFF", [("a", am.DOUBLE), ("b", am.DOUBLE)])
        img = am.pack_return(ff, {"a": 1.0, "b": 2.0})
        self.assertEqual(img.float_regs, (1.0, 2.0))
        ii = rec("RetII", [("a", am.INT64), ("b", am.INT64)])
        img2 = am.pack_return(ii, {"a": -1, "b": 2})
        self.assertEqual(img2.int_regs, (0xFFFFFFFFFFFFFFFF, 2))

    def test_padding_written_zero(self):
        # {u8, u32}: 间隙3字节+尾填充3字节必须写零
        r = rec("RetPad", [("a", am.UINT8), ("b", am.UINT32)])
        img = am.pack_return(r, {"a": 0xAA, "b": 0x01020304})
        # 手工字节布局: AA 00 00 00 | 04 03 02 01 -> 小端整数
        self.assertEqual(img.int_regs[0], 0x01020304_000000AA)

    def test_empty_record_return_image(self):
        img = am.pack_return(EMPTY, {})
        self.assertEqual(img.int_regs, (None, None))
        self.assertEqual(img.float_regs, (None, None))
        self.assertEqual(img.memory, b"")

    def test_indirect_memory_bytes_nested_arrays(self):
        # 嵌套数组记录：手工拼出期望的 24 字节映像
        value = {"m": [[1, 2], [3, 4]], "x": 1.5, "tag": 0xAB}
        img = am.pack_return(NESTED, value)
        self.assertEqual(img.plan.kind, "indirect")
        self.assertEqual(img.int_regs, (None, None))
        self.assertEqual(img.float_regs, (None, None))
        expected = (
            b"\x01\x00\x02\x00\x03\x00\x04\x00"   # m = [[1,2],[3,4]] 小端
            + struct.pack("<d", 1.5)              # x @ 8
            + b"\xab"                             # tag @ 16
            + b"\x00" * 7                         # 尾填充写零
        )
        self.assertEqual(len(expected), 24)
        self.assertEqual(img.memory, expected)
        # 显式断言 padding 字节全零
        self.assertEqual(img.memory[17:], b"\x00" * 7)

    def test_return_integer_overflow_rejected(self):
        with self.assertRaises(am.IntegerOverflowError) as ctx:
            am.pack_return(am.UINT8, 300)
        self.assertEqual(ctx.exception.path, "return")
        with self.assertRaises(am.IntegerOverflowError):
            am.pack_return(am.INT16, 40000)

    def test_return_value_validated_like_args(self):
        with self.assertRaises(am.InvalidValueError):
            am.pack_return(am.UINT8, True)          # bool 冒充整数
        with self.assertRaises(am.NonFiniteFloatError):
            am.pack_return(am.DOUBLE, float("nan"))
        with self.assertRaises(am.InvalidValueError):
            am.pack_return(MIXED, {"weight": 0.5})  # 字段缺失

    def test_plan_mismatch_rejected(self):
        with self.assertRaises(am.InvalidValueError):
            am.pack_return(MIXED, {"weight": 0.5, "id": 1},
                           am.plan_return(BIG24))


class TestUnpackReturnIndependent(unittest.TestCase):
    """手工构造 ReturnImage 单向读取，不经过 pack_return。"""

    def test_read_mixed_record_from_handmade_image(self):
        plan = am.ReturnPlan("registers", (am.FLOAT, am.INTEGER), 16, 8)
        img = am.ReturnImage(plan, (0x2A, None), (2.5, None), b"")
        self.assertEqual(am.unpack_return(MIXED, img),
                         {"weight": 2.5, "id": 0x2A})

    def test_padding_not_read_as_value(self):
        # {u8, u8}: 寄存器高 6 字节是 padding，填入垃圾不得影响读取
        r = rec("RetTwoB", [("a", am.UINT8), ("b", am.UINT8)])
        plan = am.ReturnPlan("registers", (am.INTEGER,), 2, 1)
        img = am.ReturnImage(plan, (0xDEADBEEFCAFE2211, None), (None, None),
                             b"")
        self.assertEqual(am.unpack_return(r, img), {"a": 0x11, "b": 0x22})
        # 间接返回：缓冲区 padding 填入 0xFF 垃圾，字段值不变
        plan2 = am.plan_return(NESTED)
        memory = (b"\x01\x00\x02\x00\x03\x00\x04\x00"
                  + struct.pack("<d", 1.5) + b"\xab" + b"\xff" * 7)
        img2 = am.ReturnImage(plan2, (None, None), (None, None), memory)
        self.assertEqual(am.unpack_return(NESTED, img2),
                         {"m": [[1, 2], [3, 4]], "x": 1.5, "tag": 0xAB})

    def test_read_indirect_from_handmade_memory(self):
        plan = am.plan_return(BIG24)
        memory = struct.pack("<ddd", 1.0, 2.0, 3.0)
        img = am.ReturnImage(plan, (None, None), (None, None), memory)
        self.assertEqual(am.unpack_return(BIG24, img),
                         {"a": 1.0, "b": 2.0, "c": 3.0})

    def test_truncated_memory_rejected(self):
        plan = am.plan_return(BIG24)
        img = am.ReturnImage(plan, (None, None), (None, None), b"\x00" * 8)
        with self.assertRaises(am.InvalidValueError):
            am.unpack_return(BIG24, img)

    def test_empty_register_slot_rejected(self):
        plan = am.ReturnPlan("registers", (am.INTEGER,), 1, 1)
        img = am.ReturnImage(plan, (None, None), (None, None), b"")
        with self.assertRaises(am.InvalidValueError):
            am.unpack_return(am.UINT8, img)

    def test_image_plan_mismatch_rejected(self):
        img = am.pack_return(MIXED, {"weight": 1.0, "id": 1})
        with self.assertRaises(am.InvalidValueError):
            am.unpack_return(BIG24, img)


class TestHiddenPointer(unittest.TestCase):
    def test_hidden_ptr_takes_first_int_register(self):
        buf = am.ReturnBuffer(0x1000, 64)
        img = am.pack_call([(am.INT64, 7)], ret_type=BIG24, ret_buffer=buf)
        self.assertEqual(img.ret_plan, am.plan_return(BIG24))
        self.assertIsNotNone(img.hidden_ptr)
        self.assertEqual(img.hidden_ptr.kind, "registers")
        self.assertEqual(img.hidden_ptr.int_regs, (0,))
        self.assertEqual(img.int_regs[0], 0x1000)   # 隐藏指针 = 缓冲区地址
        self.assertEqual(img.int_regs[1], 7)        # 原参数顺延

    def test_hidden_ptr_pushes_last_register_to_stack(self):
        # 4 个 i64 参数：隐藏指针占寄存器0，前3个进寄存器1-3，
        # 第4个被刚好挤到栈上。
        buf = am.ReturnBuffer(0x2000, 32)
        args = [(am.INT64, 1), (am.INT64, 2), (am.INT64, 3), (am.INT64, 4)]
        img = am.pack_call(args, ret_type=BIG24, ret_buffer=buf)
        self.assertEqual([img.int_regs[i] for i in range(4)],
                         [0x2000, 1, 2, 3])
        p = img.placements[3]
        self.assertEqual(p.kind, "stack")
        self.assertEqual(p.int_regs, ())
        self.assertEqual(p.stack_offset, 0)
        self.assertEqual(img.stack[:8], (4).to_bytes(8, "little"))
        # 被调方视角：用户参数仍能完整恢复（隐藏指针不计入 placements）
        self.assertEqual(len(img.placements), 4)
        self.assertEqual(am.unpack_call([am.INT64] * 4, img), [1, 2, 3, 4])

    def test_hidden_ptr_can_push_whole_aggregate_to_stack(self):
        # 隐藏指针 + 3 个 i64 占满整数寄存器后，{i64,i64} 整体走栈
        rr = rec("RetTwoI64", [("a", am.INT64), ("b", am.INT64)])
        buf = am.ReturnBuffer(0x3000, 32)
        args = ([(am.INT64, 1), (am.INT64, 2), (am.INT64, 3),
                 (rr, {"a": 8, "b": 9})])
        img = am.pack_call(args, ret_type=BIG24, ret_buffer=buf)
        self.assertEqual(img.placements[3].kind, "stack")
        self.assertEqual(img.placements[3].int_regs, ())
        self.assertEqual(img.stack[:16],
                         (8).to_bytes(8, "little") + (9).to_bytes(8, "little"))

    def test_register_return_needs_no_hidden_ptr(self):
        img = am.pack_call([(am.INT64, 5)], ret_type=MIXED)
        self.assertIsNone(img.hidden_ptr)
        self.assertEqual(img.int_regs[0], 5)  # 第0个整数寄存器仍给参数
        # 空记录返回同样不占参数寄存器
        img2 = am.pack_call([(am.INT64, 6)], ret_type=EMPTY)
        self.assertIsNone(img2.hidden_ptr)
        self.assertEqual(img2.int_regs[0], 6)


class TestReturnBufferValidation(unittest.TestCase):
    def test_valid_buffer_accepted(self):
        buf = am.ReturnBuffer(0x1000, 24)
        img = am.pack_call([], ret_type=BIG24, ret_buffer=buf)
        self.assertEqual(img.int_regs[0], 0x1000)
        # 容量大于类型 size 也合法
        img2 = am.pack_call([], ret_type=BIG24,
                            ret_buffer=am.ReturnBuffer(0x1000, 4096))
        self.assertEqual(img2.int_regs[0], 0x1000)

    def test_invalid_buffers_rejected(self):
        cases = [
            None,                                  # 缺失
            am.ReturnBuffer(0x1004, 64),           # 未按 16 对齐
            am.ReturnBuffer(0x1000, 16),           # 容量 < 24
            am.ReturnBuffer(0x1000, 0),            # 容量为 0
            am.ReturnBuffer(-16, 64),              # 负地址
            am.ReturnBuffer((1 << 64) - 16, 64),   # 地址+size 溢出 2^64
            am.ReturnBuffer(0x1000, am.MAX_RET_BUFFER_SIZE + 16),  # 容量超限
        ]
        for buf in cases:
            with self.assertRaises(am.InvalidValueError,
                                   msg=f"buf={buf!r}") as ctx:
                am.pack_call([], ret_type=BIG24, ret_buffer=buf)
            self.assertEqual(ctx.exception.path, "ret_buffer")

    def test_buffer_validated_before_any_arg_encoding(self):
        # arg[0] 的值非法（u8=999），但缓冲区也非法：必须先报缓冲区，
        # 证明验证发生在任何参数编码之前（不会留下部分编组结果）。
        with self.assertRaises(am.InvalidValueError) as ctx:
            am.pack_call([(am.UINT8, 999)], ret_type=BIG24,
                         ret_buffer=am.ReturnBuffer(0x1004, 64))
        self.assertEqual(ctx.exception.path, "ret_buffer")

    def test_buffer_supplied_but_not_needed_rejected(self):
        with self.assertRaises(am.InvalidValueError):
            am.pack_call([], ret_type=MIXED,
                         ret_buffer=am.ReturnBuffer(0x1000, 64))
        with self.assertRaises(am.InvalidValueError):
            am.pack_call([], ret_buffer=am.ReturnBuffer(0x1000, 64))


class TestSimulateCall(unittest.TestCase):
    def test_full_flow_register_return(self):
        # 参数编码 -> 模拟计算 -> 返回编码 -> 调用方读取（混合整浮记录）
        def weighted_avg(a, b):
            return {"weight": (a + b) / 2.0, "id": 7}

        result = am.simulate_call(weighted_avg,
                                  [(am.DOUBLE, 1.0), (am.DOUBLE, 3.0)],
                                  MIXED)
        self.assertEqual(result, {"weight": 2.0, "id": 7})

    def test_full_flow_indirect_return_nested_arrays(self):
        def make_matrix(scale, tag):
            return {"m": [[scale, 2 * scale], [3 * scale, 4 * scale]],
                    "x": scale / 2.0, "tag": tag}

        buf = am.ReturnBuffer(0x4000, 64)
        result = am.simulate_call(make_matrix,
                                  [(am.INT32, 5), (am.UINT8, 0xAB)],
                                  NESTED, ret_buffer=buf)
        self.assertEqual(result, {"m": [[5, 10], [15, 20]],
                                  "x": 2.5, "tag": 0xAB})

    def test_full_flow_empty_record_return(self):
        result = am.simulate_call(lambda: {}, [], EMPTY)
        self.assertEqual(result, {})

    def test_return_overflow_aborts_call(self):
        # 模拟计算产生越界返回值：整体抛错，无部分结果
        def bad(_x):
            return 300  # u8 越界

        with self.assertRaises(am.IntegerOverflowError):
            am.simulate_call(bad, [(am.UINT8, 1)], am.UINT8)

    def test_arg_error_aborts_before_computation(self):
        called = []

        def spy(_x):
            called.append(True)
            return 0

        with self.assertRaises(am.IntegerOverflowError):
            am.simulate_call(spy, [(am.UINT8, 999)], am.UINT8)
        self.assertEqual(called, [])  # 参数非法时 func 未被执行


class TestReturnRoundTrip(unittest.TestCase):
    """往返一致性（补充性质，不作为正确性的唯一依据）。"""

    def test_roundtrip_all_kinds(self):
        cases = [
            (am.INT8, -5), (am.UINT64, 2**64 - 1), (am.DOUBLE, -0.25),
            (am.PTR, 0xDEADBEEF),
            (MIXED, {"weight": 3.25, "id": -9}),
            (EMPTY, {}),
            (NESTED, {"m": [[-1, 2], [-3, 4]], "x": 0.125, "tag": 0x7F}),
            (BIG24, {"a": 1.0, "b": -2.0, "c": 3.5}),
        ]
        for t, v in cases:
            with self.subTest(type=t.name):
                img = am.pack_return(t, v)
                self.assertEqual(am.unpack_return(t, img), v)


if __name__ == "__main__":
    unittest.main()
