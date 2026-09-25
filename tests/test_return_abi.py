"""返回值 ABI 编组的单元测试（返回寄存器 / 有界缓冲区间接返回 / 隐藏指针）。

独立性说明：本文件的期望值**不**由被测核心的 pack/encode 产生，来源有二：
1. 手工计算并硬编码的逐字节字面量（完整 16/32 字节布局、寄存器 8 字节块）；
2. 本文件内独立编写的标量偏移参考 ``ref_offsets`` / 独立字节拼装
   ``ref_build``，它们只消费题面规则与 struct 标量格式，不调用 abi_marshal
   的布局/编码函数。
pack/unpack 往返一致仅作为补充，绝不作为正确性的唯一依据。
"""

import dataclasses
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


# ---------------------------------------------------------------------------
# 独立参考：仅按题面规则（自然对齐、上限 8、尾填充）手工计算偏移，
# 不调用 abi_marshal.layout。
# ---------------------------------------------------------------------------

def ref_offsets(spec):
    """spec: [(name, size, align), ...] -> (逐字段偏移, 总 size, align)。"""
    offs = []
    off = 0
    align = 1
    for _n, sz, al in spec:
        off = (off + al - 1) // al * al
        offs.append(off)
        off += sz
        align = max(align, al)
    total = (off + align - 1) // align * align
    return offs, total, align


def ref_grid_bytes(seq, scale, tag):
    """独立字节拼装 grid 返回：直接按手工偏移写 i16/f64/u8，padding 显式零。

    不调用 abi_marshal 的任何布局/编码函数。
    """
    buf = bytearray(32)
    for i, v in enumerate(seq):
        buf[i * 2:i * 2 + 2] = v.to_bytes(2, "little", signed=True)
    # 12..15 对齐 padding 与 25..31 尾 padding：bytearray 初始即零
    buf[16:24] = struct.pack("<d", scale)
    buf[24] = tag
    return bytes(buf)


# ---------------------------------------------------------------------------
# 测试用类型
# ---------------------------------------------------------------------------

def make_types():
    # 混合整浮记录 1：{f64, i32} -> (FLOAT, INTEGER)
    mix_di = rec("MixDI", [("d", am.DOUBLE), ("i", am.INT32)])
    # 混合整浮记录 2：{i32, f64} -> (INTEGER, FLOAT)
    mix_id = rec("MixID", [("i", am.INT32), ("d", am.DOUBLE)])
    two_i64 = rec("TwoI64", [("a", am.INT64), ("b", am.INT64)])
    two_f64 = rec("TwoF64", [("x", am.DOUBLE), ("y", am.DOUBLE)])
    # 嵌套数组大聚合（32 字节）：
    #   i16 grid[2][3] @0 (12 字节), pad 12..15, f64 scale @16, u8 tag @24,
    #   尾填充 25..31
    grid_t = rec("Grid", [
        ("grid", am.ArrayType(am.ArrayType(am.INT16, 3), 2)),
        ("scale", am.DOUBLE),
        ("tag", am.UINT8),
    ])
    big3 = rec("Big3", [("a", am.DOUBLE), ("b", am.DOUBLE), ("c", am.DOUBLE)])
    empty = rec("EmptyRet", [])
    exactly16 = rec("Exact16Ret", [("a", am.UINT64), ("b", am.UINT64)])
    # 真正的 17 字节聚合（记录总会尾填充到 8 的倍数，故用数组）：
    seventeen = am.ArrayType(am.UINT8, 17)
    return dict(mix_di=mix_di, mix_id=mix_id, two_i64=two_i64,
                two_f64=two_f64, grid=grid_t, big3=big3, empty=empty,
                exactly16=exactly16, seventeen=seventeen)


T = make_types()


# ---------------------------------------------------------------------------
# 布局 / 分类：与独立参考逐字节核对
# ---------------------------------------------------------------------------

class TestReturnLayoutIndependent(unittest.TestCase):
    def test_grid_offsets_match_reference(self):
        # 独立偏移表（手写 size/align 字面量，不读核心结果）
        spec = [("grid", 12, 2), ("scale", 8, 8), ("tag", 1, 1)]
        ref_off, ref_total, ref_align = ref_offsets(spec)
        self.assertEqual(ref_off, [0, 16, 24])
        self.assertEqual((ref_total, ref_align), (32, 8))
        lay = am.layout(T["grid"])
        self.assertEqual([f.offset for f in lay.fields], ref_off)
        self.assertEqual((lay.size, lay.align), (ref_total, ref_align))

    def test_grid_classify_is_memory(self):
        self.assertEqual(am.classify(T["grid"]), (am.MEMORY,))
        # 恰好 16 仍寄存器，17 即 MEMORY
        self.assertEqual(am.classify(T["exactly16"]), (am.INTEGER, am.INTEGER))
        self.assertEqual(am.classify(T["seventeen"]), (am.MEMORY,))
        self.assertEqual(am.layout(T["seventeen"]).size, 17)
        self.assertEqual(am.layout(T["seventeen"]).align, 1)

    def test_return_convention_kinds(self):
        self.assertEqual(am.return_convention(T["empty"]).kind, "none")
        self.assertEqual(am.return_convention(am.DOUBLE).kind, "registers")
        self.assertEqual(
            am.return_convention(T["mix_di"]).kind, "registers")
        self.assertEqual(
            am.return_convention(T["exactly16"]).kind, "registers")
        c = am.return_convention(T["grid"], 0x1000)
        self.assertEqual(c.kind, "memory")
        self.assertEqual(c.address, 0x1000)
        self.assertEqual((c.size, c.align), (32, 8))
        # memory 约定缺地址即拒绝
        with self.assertRaises(am.BufferError):
            am.return_convention(T["grid"])
        with self.assertRaises(am.BufferError):
            am.return_convention(T["grid"], -1)


# ---------------------------------------------------------------------------
# 寄存器返回：独立按字节核对两个返回寄存器
# ---------------------------------------------------------------------------

class TestRegisterReturnBytes(unittest.TestCase):
    def test_float_then_int_chunks(self):
        # {f64 d @0, i32 i @8}，值 d=-2.5, i=0x1234
        inv = am.pack_invoke(T["mix_di"], [])
        ret = am.encode_return(inv, {"d": -2.5, "i": 0x1234})
        self.assertEqual(ret.kind, "registers")
        # 块0 -> 浮点返回寄存器 0；块1 -> 整数返回寄存器 0；第二槽未用
        self.assertEqual(ret.float_regs, (-2.5, None))
        self.assertIsNone(ret.int_regs[1])
        # 独立硬编码：-2.5 的 binary64 小端
        self.assertEqual(struct.pack("<d", ret.float_regs[0]),
                         bytes.fromhex("00000000000004c0"))
        # 块1 字节 = 34 12 00 00 | 00 00 00 00（低 4 字节值，高 4 字节零填充）
        block1 = ret.int_regs[0].to_bytes(8, "little")
        self.assertEqual(block1, bytes.fromhex("3412000000000000"))
        self.assertEqual(ret.int_regs[0], 0x0000000000001234)

    def test_int_then_float_chunks_with_padding(self):
        # {i32 i @0, f64 d @8}，值 i=0x01020304, d=1.5
        inv = am.pack_invoke(T["mix_id"], [])
        ret = am.encode_return(inv, {"i": 0x01020304, "d": 1.5})
        self.assertEqual(ret.eightbytes, (am.INTEGER, am.FLOAT))
        # 独立硬编码块字节
        self.assertEqual(ret.int_regs[0].to_bytes(8, "little"),
                         bytes.fromhex("0403020100000000"))
        self.assertEqual(struct.pack("<d", ret.float_regs[0]),
                         bytes.fromhex("000000000000f83f"))
        self.assertIsNone(ret.int_regs[1])
        self.assertIsNone(ret.float_regs[1])

    def test_two_int_and_two_float_return_registers(self):
        inv = am.pack_invoke(T["two_i64"], [])
        ret = am.encode_return(inv, {"a": -1, "b": 0xAA55})
        self.assertEqual(ret.int_regs,
                         (0xFFFFFFFFFFFFFFFF, 0xAA55))
        self.assertEqual(ret.float_regs, (None, None))
        inv = am.pack_invoke(T["two_f64"], [])
        ret = am.encode_return(inv, {"x": 1.0, "y": -0.5})
        self.assertEqual(ret.float_regs, (1.0, -0.5))

    def test_scalar_returns(self):
        inv = am.pack_invoke(am.UINT64, []); self.assertEqual(
            am.decode_return(inv, am.encode_return(inv, 7)), 7)
        inv = am.pack_invoke(am.INT8, [])
        ret = am.encode_return(inv, -1)
        # i8=-1 在块内做无符号 8 字节表示，读回按 i8 仍为 -1
        self.assertEqual(ret.int_regs[0], 0xFF)
        self.assertEqual(am.decode_return(inv, ret), -1)
        inv = am.pack_invoke(am.DOUBLE, [])
        ret = am.encode_return(inv, 3.25)
        self.assertEqual(struct.pack("<d", ret.float_regs[0]),
                         bytes.fromhex("0000000000000a40"))

    def test_empty_return_consumes_nothing(self):
        inv = am.pack_invoke(T["empty"], [(am.INT64, 9)])
        ret = am.encode_return(inv, {})
        self.assertEqual(ret.kind, "none")
        self.assertEqual(ret.int_regs, (None, None))
        self.assertEqual(ret.float_regs, (None, None))
        # 空返回不注入隐藏指针，参数照常占用整数寄存器 0
        self.assertEqual(inv.call_image.int_regs[0], 9)
        self.assertEqual(am.decode_return(inv, ret), {})

    def test_register_return_rejects_buffer(self):
        with self.assertRaises(am.InvalidValueError):
            am.pack_invoke(am.UINT64, [], am.Memory(64), 0, 64)


# ---------------------------------------------------------------------------
# 间接返回：逐字节硬编码核对 32 字节缓冲区
# ---------------------------------------------------------------------------

GRID_VALUE = {"grid": [[1, -2, 3], [-4, 5, -6]], "scale": 1.5, "tag": 0x7F}
# 手工硬编码：见文件头说明。32 字节完整布局。
GRID_HEX = (
    "0100feff0300"          # grid[0] = 1, -2, 3（i16 小端）
    "fcff0500faff"          # grid[1] = -4, 5, -6
    "00000000"              # 12..15 对齐 padding（必须为零）
    "000000000000f83f"      # scale @16 = 1.5
    "7f"                    # tag @24 = 0x7f
    "00000000000000"        # 25..31 尾 padding（必须为零）
)
GRID_VALUE_OVERFLOW = {"grid": [[1, 32768, 3], [-4, 5, -6]],
                       "scale": 1.5, "tag": 1}


class TestIndirectReturnBytes(unittest.TestCase):
    def setUp(self):
        self.mem = am.Memory(1 << 16)
        self.addr = 0x1000  # 显式非零、8 对齐地址，便于识别隐藏指针

    def invoke(self):
        return am.pack_invoke(T["grid"], [], self.mem, self.addr, 64)

    def test_buffer_bytes_match_hardcoded_layout(self):
        inv = self.invoke()
        self.assertEqual(inv.convention.kind, "memory")
        self.assertEqual(inv.hidden_pointer, self.addr)
        ret = am.encode_return(inv, GRID_VALUE)
        self.assertEqual(ret.kind, "memory")
        # 寄存器返回槽全部未用
        self.assertEqual(ret.int_regs, (None, None))
        self.assertEqual(ret.float_regs, (None, None))
        raw = self.mem.read(self.addr, 32)
        # 与硬编码逐字节比较
        self.assertEqual(raw, bytes.fromhex(GRID_HEX))
        # 与独立拼装器比较（独立参考，不经过 am.encode）
        self.assertEqual(raw, ref_grid_bytes([1, -2, 3, -4, 5, -6], 1.5, 0x7F))
        # 隐藏指针位于第一个整数参数寄存器
        self.assertEqual(inv.call_image.int_regs[0], self.addr)

    def test_padding_is_zero_and_never_read_as_value(self):
        inv = self.invoke()
        am.encode_return(inv, GRID_VALUE)
        raw = self.mem.read(self.addr, 32)
        self.assertEqual(raw[12:16], b"\x00" * 4)
        self.assertEqual(raw[25:32], b"\x00" * 7)
        # 外部污染 padding：读取结果不得受影响（读取只按类型化叶子）
        self.mem.fill(self.addr + 12, 4, 0xFF)
        self.mem.fill(self.addr + 25, 7, 0xFF)
        decoded = am.decode_return(inv,
                                   am.ReturnImage("memory", (am.MEMORY,)))
        self.assertEqual(decoded, GRID_VALUE)

    def test_caller_reads_indirect_return(self):
        inv = self.invoke()
        ret = am.encode_return(inv, GRID_VALUE)
        self.assertEqual(am.decode_return(inv, ret), GRID_VALUE)

    def test_integer_overflow_writes_nothing(self):
        # 预置哨兵；越界发生在写入前，32 字节必须原样不变
        self.mem.fill(self.addr, 64, 0xEE)
        inv = self.invoke()
        with self.assertRaises(am.IntegerOverflowError) as ctx:
            am.encode_return(inv, GRID_VALUE_OVERFLOW)
        self.assertEqual(ctx.exception.path, "ret.grid[0][1]")
        self.assertEqual(self.mem.read(self.addr, 32), b"\xee" * 32)

    def test_callee_revalidates_untrusted_buffer(self):
        # 模拟被调方拿到一个实际容量不足的缓冲区（dataclasses 替换）：
        # 必须拒绝且不写。
        good = self.invoke()
        am.encode_return(good, GRID_VALUE)
        tiny = am.Memory(8)
        bad = dataclasses.replace(good, memory=tiny, address=0)
        tiny.fill(0, 8, 0xDD)
        with self.assertRaises(am.BufferError):
            am.encode_return(bad, GRID_VALUE)
        self.assertEqual(tiny.read(0, 8), b"\xdd" * 8)
        # 未对齐同样拒绝
        mis = am.Memory(64)
        bad2 = dataclasses.replace(good, memory=mis, address=4)
        with self.assertRaises(am.BufferError):
            am.encode_return(bad2, GRID_VALUE)
        self.assertEqual(mis.read(0, 64), b"\x00" * 64)

    def test_kind_mismatch_rejected(self):
        inv = self.invoke()
        with self.assertRaises(am.InvalidValueError):
            am.decode_return(inv, am.ReturnImage("registers"))


# ---------------------------------------------------------------------------
# 缓冲区：大小、对齐、范围先验证，失败不写
# ---------------------------------------------------------------------------

class TestBufferValidation(unittest.TestCase):
    def setUp(self):
        self.t = T["grid"]  # size 32, align 8

    def _sentinel(self, mem, addr, n, b=0xCD):
        mem.fill(addr, n, b)

    def test_capacity_exact_passes(self):
        mem = am.Memory(32)
        am.prepare_return_buffer(self.t, mem, 0, 32)  # 不抛即可

    def test_capacity_short_rejected_without_write(self):
        mem = am.Memory(32)
        self._sentinel(mem, 0, 32)
        with self.assertRaises(am.BufferError):
            am.pack_invoke(self.t, [], mem, 0, 31)
        self.assertEqual(mem.read(0, 32), b"\xcd" * 32)

    def test_misaligned_rejected_without_write(self):
        mem = am.Memory(64)
        self._sentinel(mem, 4, 32)
        with self.assertRaises(am.BufferError):
            am.pack_invoke(self.t, [], mem, 4, 60)
        self.assertEqual(mem.read(4, 32), b"\xcd" * 32)

    def test_range_exceeds_address_space(self):
        mem = am.Memory(64)
        with self.assertRaises(am.BufferError):
            am.pack_invoke(self.t, [], mem, 2**64 - 16, 32)
        with self.assertRaises(am.BufferError):
            am.prepare_return_buffer(self.t, mem, 2**64 - 1, 32)

    def test_range_exceeds_backing_memory(self):
        mem = am.Memory(24)  # 地址空间够但后备内存不够
        with self.assertRaises(am.BufferError):
            am.pack_invoke(self.t, [], mem, 0, 32)

    def test_align_1_type_accepts_any_address(self):
        # u8[17] 是 size=17（MEMORY）、align=1 的大聚合：任意地址都满足对齐
        t = am.ArrayType(am.UINT8, 17)
        mem = am.Memory(32)
        inv = am.pack_invoke(t, [], mem, 5, 27)  # 地址 5 对齐 1 合法
        vals = list(range(17))
        ret = am.encode_return(inv, vals)
        self.assertEqual(mem.read(5, 17), bytes(vals))
        self.assertEqual(am.decode_return(inv, ret), vals)

    def test_bad_arguments_rejected(self):
        mem = am.Memory(64)
        with self.assertRaises(am.BufferError):
            am.prepare_return_buffer(self.t, mem, True, 32)
        with self.assertRaises(am.BufferError):
            am.prepare_return_buffer(self.t, mem, 0, True)
        with self.assertRaises(am.BufferError):
            am.prepare_return_buffer(self.t, "not a memory", 0, 32)
        with self.assertRaises(am.InvalidTypeError):
            am.Memory(-1)


# ---------------------------------------------------------------------------
# 隐藏指针先占整数参数寄存器：刚好挤掉最后一个聚合
# ---------------------------------------------------------------------------

class TestHiddenPointerDisplacement(unittest.TestCase):
    def test_last_aggregate_just_fits_then_is_pushed_to_stack(self):
        args = [(am.INT64, 0x1111), (am.INT64, 0x2222),
                (T["two_i64"], {"a": 0x3333, "b": 0x4444})]

        # 对照：返回走寄存器（无隐藏指针）时两个标量占 int 0/1，
        # TwoI64 恰好占 int 2/3。
        inv0 = am.pack_invoke(am.UINT64, args)
        img0 = inv0.call_image
        self.assertEqual(img0.int_regs[:4],
                         (0x1111, 0x2222, 0x3333, 0x4444))
        self.assertEqual(img0.placements[2].kind, "registers")
        self.assertEqual(img0.placements[2].int_regs, (2, 3))
        self.assertEqual(img0.stack, b"")

        # 间接返回：隐藏指针先占 int 0，标量改占 int 1/2，
        # TwoI64 需要 2 个寄存器但只剩 int 3 -> 整体走栈。
        mem = am.Memory(1 << 16)
        inv1 = am.pack_invoke(T["big3"], args, mem, 0x1000, 64)
        img1 = inv1.call_image
        self.assertEqual(inv1.hidden_pointer, 0x1000)
        self.assertEqual(img1.int_regs[0], 0x1000)
        self.assertEqual(img1.int_regs[1], 0x1111)
        self.assertEqual(img1.int_regs[2], 0x2222)
        self.assertIsNone(img1.int_regs[3])
        p = img1.placements[2]
        self.assertEqual(p.kind, "stack")
        self.assertEqual(p.int_regs, ())          # 回退不消耗寄存器
        self.assertEqual(p.stack_offset, 0)
        # 栈映像逐字节独立核对：两个 i64 小端
        self.assertEqual(
            img1.stack[:16],
            (0x3333).to_bytes(8, "little") + (0x4444).to_bytes(8, "little"))
        # 浮点参数寄存器完全不受隐藏指针影响
        self.assertEqual(img1.float_regs, (None, None))
        # 被调方仍能从栈映像完整重建该聚合
        self.assertEqual(
            am.unpack_call([am.INT64, am.INT64, T["two_i64"]], img1),
            [0x1111, 0x2222, {"a": 0x3333, "b": 0x4444}])

    def test_hidden_pointer_consumes_no_float_register(self):
        # i64 + TwoF64（需要全部 2 个浮点寄存器）。
        args = [(am.INT64, 5), (T["two_f64"], {"x": 1.0, "y": 2.0})]
        # 对照：无隐藏指针时 i64@int0，TwoF64@float0/1。
        inv0 = am.pack_invoke(am.UINT64, args)
        self.assertEqual(inv0.call_image.placements[1].kind, "registers")
        self.assertEqual(inv0.call_image.placements[1].float_regs, (0, 1))
        # 间接返回：隐藏指针占 int0，i64 改占 int1，但 TwoF64 仍拿到
        # 全部两个浮点寄存器——证明隐藏指针不消耗任何浮点寄存器。
        mem = am.Memory(1 << 16)
        inv = am.pack_invoke(T["big3"], args, mem, 0x2000, 64)
        img = inv.call_image
        self.assertEqual(img.int_regs[0], 0x2000)
        self.assertEqual(img.int_regs[1], 5)
        self.assertEqual(img.float_regs, (1.0, 2.0))
        p = img.placements[1]
        self.assertEqual(p.kind, "registers")
        self.assertEqual(p.float_regs, (0, 1))

    def test_hidden_with_four_int_scalars(self):
        # 隐藏指针 + 4 个 i64：第 4 个标量被挤到栈
        args = [(am.INT64, i + 1) for i in range(4)]
        mem = am.Memory(1 << 16)
        inv = am.pack_invoke(T["big3"], args, mem, 0x800, 64)
        img = inv.call_image
        self.assertEqual([img.int_regs[i] for i in range(4)],
                         [0x800, 1, 2, 3])
        self.assertEqual(img.placements[3].kind, "stack")
        self.assertEqual(img.stack[:8], (4).to_bytes(8, "little"))


# ---------------------------------------------------------------------------
# 完整流程：参数编码 -> 模拟计算 -> 返回编码 -> 调用方读取
# （独立字节核对中间映像，而非仅靠往返）
# ---------------------------------------------------------------------------

class TestFullInvokeFlow(unittest.TestCase):
    def test_indirect_flow_byte_exact(self):
        # --- 调用方 ---
        mem = am.Memory(1 << 16)
        arg_types = [am.UINT16, am.DOUBLE, T["two_i64"]]
        args = [(am.UINT16, 1000), (am.DOUBLE, 0.25),
                (T["two_i64"], {"a": 40, "b": 50})]
        inv = am.pack_invoke(T["grid"], args, mem, 0x4000, 4096)
        img = inv.call_image

        # 独立核对参数映像字节：
        # 隐藏指针 @int0；u16=1000 @int1（块内低 2 字节，高 6 字节零）；
        # f64=0.25 @float0；TwoI64 只剩 int2/3 -> 恰好仍寄存器。
        self.assertEqual(img.int_regs[0], 0x4000)
        self.assertEqual(img.int_regs[1].to_bytes(8, "little"),
                         (1000).to_bytes(2, "little") + b"\x00" * 6)
        self.assertEqual(img.int_regs[2:4], (40, 50))
        self.assertEqual(img.placements[2].kind, "registers")
        self.assertEqual(struct.pack("<d", img.float_regs[0]),
                         bytes.fromhex("000000000000d03f"))  # 0.25

        # --- 被调方：解参 + 模拟计算 ---
        received = am.unpack_call(arg_types, img)
        self.assertEqual(received, [1000, 0.25, {"a": 40, "b": 50}])
        result = {
            "grid": [[1000, -1000, 41], [52, -3, 4]],
            "scale": 0.5,
            "tag": 7,
        }

        # --- 被调方：编码返回（写入调用方缓冲区）---
        ret = am.encode_return(inv, result)

        # --- 独立逐字节核对缓冲区 ---
        raw = mem.read(0x4000, 32)
        expect = ref_grid_bytes([1000, -1000, 41, 52, -3, 4], 0.5, 7)
        self.assertEqual(raw, expect)
        self.assertEqual(raw[12:16], b"\x00\x00\x00\x00")
        self.assertEqual(raw[25:32], b"\x00" * 7)

        # --- 调用方：读回 ---
        self.assertEqual(am.decode_return(inv, ret), result)

    def test_register_flow_byte_exact(self):
        # 标量 + 混合聚合参数；混合整浮聚合寄存器返回
        args = [(am.INT64, 3), (T["mix_id"], {"i": 12, "d": 4.0})]
        inv = am.pack_invoke(T["mix_di"], args)  # 返回 {f64,i32}
        img = inv.call_image
        # 无隐藏指针：i64 @int0；{i32,f64} -> int1, float0
        self.assertEqual(img.int_regs[0], 3)
        self.assertEqual(img.int_regs[1].to_bytes(8, "little"),
                         (12).to_bytes(4, "little") + b"\x00" * 4)
        self.assertEqual(img.float_regs[0], 4.0)
        # 被调方计算
        n, m = am.unpack_call([am.INT64, T["mix_id"]], img)
        result = {"d": m["d"] + 0.5, "i": n + m["i"]}  # 4.5, 15
        ret = am.encode_return(inv, result)
        # 独立块核对
        self.assertEqual(struct.pack("<d", ret.float_regs[0]),
                         bytes.fromhex("0000000000001240"))  # 4.5
        self.assertEqual(ret.int_regs[0].to_bytes(8, "little"),
                         (15).to_bytes(4, "little") + b"\x00" * 4)
        self.assertEqual(am.decode_return(inv, ret), result)

    def test_internal_buffer_convenience(self):
        # 不传 memory：pack_invoke 自建缓冲区，调用方经 InvokeImage 读回
        inv = am.pack_invoke(T["grid"], [])
        self.assertIsInstance(inv.memory, am.Memory)
        self.assertEqual(inv.address, 0)
        ret = am.encode_return(inv, GRID_VALUE)
        self.assertEqual(am.decode_return(inv, ret), GRID_VALUE)
        self.assertEqual(inv.memory.read(0, 32), bytes.fromhex(GRID_HEX))


if __name__ == "__main__":
    unittest.main()
