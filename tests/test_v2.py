"""abi_marshal v2 的快速回归测试。

期望值来源：
1. tests/reference.py 中**独立书写**的平行参考（布局/编码/规划/映像重放/
   证书核对），不调用 abi_marshal 的 layout/encode/plan_call 等取期望值；
2. 手工硬编码字面量（位物理位置、标签字节、槽号、偏移）。
桥接测试比较规范值与约定保留的位（NaN 载荷、有符号边界），不把填充字节
相等当作标准。
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abi_marshal as am
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref

A, B = am.DESCRIPTOR_A, am.DESCRIPTOR_B
# 第三个描述符：禁止位域跨存储单元（用于桥接不可表示拒绝）
C = am.AbiDescriptor(
    name="C-big-nospan", endian="big", int_reg_count=4, float_reg_count=2,
    max_reg_bytes=16, align_int={1: 1, 2: 2, 4: 4, 8: 8, 16: 16},
    bitfield_span=False, i128_even_pair=True, sret_threshold=8)


def rec(name, fields):
    r = am.RecordType(name)
    r.define(fields)
    return r


def uni(name, members, tag=am.UINT8):
    u = am.UnionType(name)
    u.define(members, tag_type=tag)
    return u


def nan_with_payload(p):
    return struct.unpack("<d", struct.pack("<Q", 0x7FF0_0000_0000_0000 | p))[0]


class TestDescriptorGuards(unittest.TestCase):
    def test_endian_required(self):
        with self.assertRaises(am.InvalidTypeError):
            am.AbiDescriptor(name="x", endian="middle")
        with self.assertRaises(am.InvalidTypeError):
            am.AbiDescriptor(name="x", endian="little", max_reg_bytes=12)

    def test_v1_rejects_v2_types(self):
        with self.assertRaises(am.InvalidTypeError):
            am.layout(am.INT128)
        with self.assertRaises(am.InvalidTypeError):
            am.layout(am.FLOAT32)
        with self.assertRaises(am.InvalidTypeError):
            am.encode(am.INT128, 0)
        # v1 仍然拒绝 NaN
        with self.assertRaises(am.NonFiniteFloatError):
            am.encode(am.DOUBLE, float("nan"))


class TestBitfieldSmallDomain(unittest.TestCase):
    """确定性小参考域：宽度 1..12、两种字节序，逐位独立核对。"""

    def _expected_bits(self, endian, g, width, run_bytes):
        """参考公式：值的第 j 位 -> (字节, 字节内位)。"""
        out = []
        for j in range(width):
            lin = g + j
            if endian == "little":
                out.append((lin // 8, lin % 8))
            else:
                out.append((run_bytes - 1 - lin // 8, 7 - lin % 8))
        return out

    def test_single_field_all_widths_both_endians(self):
        for desc in (A, B):
            for w in range(1, 13):
                r = rec(f"BF{desc.name}{w}",
                        [("x", am.BitField(w, True)),
                         ("pad", am.UINT8)])
                lay = am.layout(r, abi=desc)
                fl = lay.fields[0]
                run_bytes = (w + 7) // 8
                exp = self._expected_bits(desc.endian, 0, w, run_bytes)
                got = sorted((s.byte, s.lsb + k)
                             for s in fl.bit_segments
                             for k in range(s.count))
                self.assertEqual(got, sorted(exp),
                                 f"{desc.name} w={w} physical bits")
                rr = ref.ref_layout(desc, r)
                # 全部边界值逐位核对
                for v in (0, -(1 << (w - 1)), (1 << (w - 1)) - 1):
                    data = am.encode(r, {"x": v, "pad": 0}, abi=desc)
                    exp_data = ref.ref_encode(desc, r, rr,
                                              {"x": v, "pad": 0})
                    self.assertEqual(data, exp_data,
                                     f"{desc.name} w={w} v={v}")
                    self.assertEqual(am.decode(r, data, abi=desc)["x"], v)

    def test_crossing_unit_positions(self):
        # 宽度为 5 的字段后紧跟宽度 1..11 的字段，必产生跨字节
        for desc in (A, B):
            for w2 in range(1, 12):
                r = rec(f"Cr{desc.name}{w2}",
                        [("a", am.BitField(5)), ("b", am.BitField(w2))])
                lay = am.layout(r, abi=desc)
                fb = lay.fields[1]
                phys = sorted((s.byte, s.lsb + k)
                              for s in fb.bit_segments
                              for k in range(s.count))
                run_bytes = (5 + w2 + 7) // 8
                exp = sorted(self._expected_bits(desc.endian, 5, w2,
                                                 run_bytes))
                self.assertEqual(phys, exp)
                v = (1 << w2) - 1
                data = am.encode(r, {"a": 0b10101, "b": v}, abi=desc)
                dec = am.decode(r, data, abi=desc)
                self.assertEqual(dec, {"a": 0b10101, "b": v})

    def test_negative_bitfield_boundary(self):
        r = rec("Nbf", [("a", am.BitField(3, True)),
                        ("b", am.BitField(9, True)),
                        ("tail", am.UINT8)])
        for desc in (A, B):
            lay = am.layout(r, abi=desc)
            self.assertEqual(lay.size, 3)
            vals = {"a": -4, "b": -256, "tail": 0}
            data = am.encode(r, vals, abi=desc)
            rr = ref.ref_layout(desc, r)
            self.assertEqual(data, ref.ref_encode(desc, r, rr, vals))
            self.assertEqual(am.decode(r, data, abi=desc), vals)
            with self.assertRaises(am.IntegerOverflowError):
                am.encode(r, {"a": -5, "b": 0, "tail": 0}, abi=desc)
            with self.assertRaises(am.IntegerOverflowError):
                am.encode(r, {"a": 0, "b": 256, "tail": 0}, abi=desc)

    def test_overlap_and_pin_rejected_with_path(self):
        bad = am.RecordType("Overlap")
        bad.define([("a", am.BitField(4)), ("b", am.BitField(4), 0)])
        with self.assertRaises(am.BitFieldError) as ctx:
            am.layout(bad, abi=A)
        self.assertIn("field 'b'", ctx.exception.path)
        # 部分重叠（2 位）
        bad2 = am.RecordType("Overlap2")
        bad2.define([("a", am.BitField(6)), ("b", am.BitField(4), 4)])
        with self.assertRaises(am.BitFieldError):
            am.layout(bad2, abi=A)

    def test_crossing_forbidden(self):
        r = rec("NoSpan", [("a", am.BitField(5)), ("b", am.BitField(6))])
        with self.assertRaises(am.BitFieldError) as ctx:
            am.layout(r, abi=C)
        self.assertIn("field 'b'", ctx.exception.path)
        # 不跨越时合法
        r2 = rec("NoSpanOK", [("a", am.BitField(5)), ("b", am.BitField(3))])
        self.assertEqual(am.layout(r2, abi=C).size, 1)


class TestUnionSmallDomain(unittest.TestCase):
    def setUp(self):
        self.u = uni("U", [("b", am.UINT8, 7),
                           ("w", am.UINT16, 0x23),
                           ("d", am.DOUBLE, 255)], tag=am.UINT8)

    def test_layout_literal(self):
        for desc in (A, B):
            lay = am.layout(self.u, abi=desc)
            self.assertEqual(lay.tag_offset, 0)
            self.assertEqual(lay.storage_offset, 8)  # tag1 -> pad 到 8
            self.assertEqual(lay.size, 16)
            self.assertEqual(lay.align, 8)

    def test_every_member_roundtrip_and_bytes(self):
        for desc in (A, B):
            rr = ref.ref_layout(desc, self.u)
            cases = [("b", 0xAB), ("w", 0xCAFE),
                     ("d", nan_with_payload(0xDEAD))]
            for mn, val in cases:
                data = am.encode(self.u, am.Tagged(mn, val), abi=desc)
                self.assertEqual(
                    data, ref.ref_encode(
                        desc, self.u, rr, am.Tagged(mn, val)))
                back = am.decode(self.u, data, abi=desc)
                self.assertEqual(back.member, mn)
                if mn == "d":
                    self.assertEqual(struct.pack(desc.endian_char + "d",
                                                 back.value),
                                     struct.pack(desc.endian_char + "d", val))
                else:
                    self.assertEqual(back.value, val)

    def test_unknown_member_and_tag_paths(self):
        with self.assertRaises(am.TagError) as ctx:
            am.encode(self.u, am.Tagged("nope", 1), abi=A)
        self.assertEqual(ctx.exception.path, "value")
        # 映像字节里标签 99（无成员）→ decode 拒绝，定位到参数路径
        data = bytearray(16)
        data[0] = 99
        with self.assertRaises(am.TagError) as ctx:
            am.decode(self.u, bytes(data), abi=A, path="arg[3].u")
        self.assertEqual(ctx.exception.path, "arg[3].u")

    def test_duplicate_tags_rejected(self):
        u = am.UnionType("DupTag")
        with self.assertRaises(am.InvalidTypeError):
            u.define([("a", am.UINT8, 1), ("b", am.UINT8, 1)])


class TestPromotions(unittest.TestCase):
    def test_integer_and_float_promotion(self):
        sig = am.Signature(args=(am.INT64,),
                           varargs=(am.UINT8, am.INT16, am.FLOAT32, am.INT64))
        for desc in (A, B):
            p = am.plan_call(desc, sig)
            self.assertEqual(p.promoted[0].name, "u32")
            self.assertTrue(p.promoted[0].signed is False)
            self.assertEqual(p.promoted[1].name, "i32")
            self.assertEqual(p.promoted[2].name, "f64")
            self.assertIsNone(p.promoted[3])
            self.assertIsNotNone(p.save)

    def test_i128_variadic_unrepresentable(self):
        with self.assertRaises(am.PromotionError) as ctx:
            am.plan_call(A, am.Signature(varargs=(am.UINT128,)))
        self.assertEqual(ctx.exception.path, "vararg[0]")

    def test_f32_exact_domain_required_for_named(self):
        # 具名 f32 参数：不能被 binary32 精确表示的值拒绝（不静默舍入）
        p = am.plan_call(A, am.Signature(args=(am.FLOAT32,)))
        with self.assertRaises(am.InvalidValueError):
            am.encode_call(p, [0.1])  # f64 的 0.1 不在 f32 域
        # 精确可表示的值合法
        img = am.encode_call(p, [-2.5])
        self.assertEqual(am.decode_call(p, img).args[0], -2.5)

    def test_f32_nan_payload_promotion_roundtrip(self):
        sig = am.Signature(varargs=(am.FLOAT32,))
        for desc in (A, B):
            p = am.plan_call(desc, sig)
            x = nan_with_payload(0x2A)  # f64 NaN
            img = am.encode_call(p, [], [x])
            got = am.decode_call(p, img).varargs[0]
            # 经 f32 规范域：载荷按 binary32 保留
            exp = struct.unpack(desc.endian_char + "f",
                                struct.pack(desc.endian_char + "f", x))[0]
            self.assertEqual(struct.pack(desc.endian_char + "f", got),
                             struct.pack(desc.endian_char + "f", exp))
            self.assertNotEqual(got, got)  # 仍是 NaN


class TestWideIntegers(unittest.TestCase):
    def test_i128_boundaries_and_slots(self):
        for desc, slots0 in ((A, (0, 1)),):
            p = am.plan_call(desc, am.Signature(args=(am.UINT128,)))
            self.assertEqual(p.placements[0].int_regs, slots0)
        for v in (0, 2 ** 128 - 1):
            p = am.plan_call(A, am.Signature(args=(am.UINT128,)))
            img = am.encode_call(p, [v])
            self.assertEqual(am.decode_call(p, img).args[0], v)
        for v in (-(2 ** 127), 2 ** 127 - 1):
            p = am.plan_call(A, am.Signature(args=(am.INT128,)))
            img = am.encode_call(p, [v])
            self.assertEqual(am.decode_call(p, img).args[0], v)

    def test_i128_even_pair_big_endian(self):
        # 一个 u64 占槽 0；i128 必须偶数号起 -> 槽 (2,3)，槽 1 留空
        p = am.plan_call(B, am.Signature(args=(am.UINT64, am.UINT128)))
        self.assertEqual(p.placements[1].int_regs, (2, 3))
        img = am.encode_call(p, [0xAAAA, 2 ** 128 - 1])
        self.assertIsNone(img.int_regs[1])
        self.assertEqual(am.decode_call(p, img).args[1], 2 ** 128 - 1)
        # 槽不够配对（u64,u64,u128：需 (2,3) 但只剩 2,3? 实际够；
        # u64*3 后 i128 无偶数对空间 -> 整体走栈且不消耗寄存器）
        p2 = am.plan_call(B, am.Signature(
            args=(am.UINT64, am.UINT64, am.UINT64, am.UINT128)))
        self.assertEqual(p2.placements[3].kind, "stack")
        self.assertEqual(p2.placements[3].int_regs, ())

    def test_i128_bytes_endianness(self):
        p = am.plan_call(A, am.Signature(args=(am.UINT128,)))
        img = am.encode_call(p, [0x0102030405060708090A0B0C0D0E0F10])
        self.assertEqual(img.int_regs[0], 0x090A0B0C0D0E0F10)
        self.assertEqual(img.int_regs[1], 0x0102030405060708)
        pB = am.plan_call(B, am.Signature(args=(am.UINT128,)))
        imgB = am.encode_call(pB, [0x0102030405060708090A0B0C0D0E0F10])
        # 槽是块内存别名：槽 0 = 块低 8 字节（大端 [01..08]），槽 1 = 高 8
        self.assertEqual(imgB.int_regs[0], 0x0807060504030201)
        self.assertEqual(imgB.int_regs[1], 0x100F0E0D0C0B0A09)


class TestRegisterExhaustion(unittest.TestCase):
    def test_scalar_spill_both_regfiles(self):
        # A 有 6 个整数槽：前 6 进寄存器，第 7 个走栈
        p = am.plan_call(A, am.Signature(
            args=tuple([am.INT64] * 7)))
        self.assertEqual([pl.kind for pl in p.placements[:6]],
                         ["registers"] * 6)
        self.assertEqual(p.placements[6].kind, "stack")
        self.assertEqual(p.placements[6].frame_offset, 0)
        # B 只有 2 个浮点槽
        pB = am.plan_call(B, am.Signature(args=tuple([am.DOUBLE] * 3)))
        self.assertEqual(pB.placements[0].kind, "registers")
        self.assertEqual(pB.placements[2].kind, "stack")

    def test_aggregate_whole_fallback_consumes_nothing(self):
        rr = rec("Two", [("a", am.INT64), ("b", am.INT64)])
        # B: 3 个 i64 后只剩槽 3；两整块聚合放不下 -> 整体走栈
        p = am.plan_call(B, am.Signature(
            args=(am.INT64, am.INT64, am.INT64, rr, am.DOUBLE)))
        self.assertEqual(p.placements[3].kind, "stack")
        self.assertEqual(p.placements[3].int_regs, ())
        # 未消耗槽 3，且后续 double 仍拿浮点槽
        self.assertEqual(p.placements[4].kind, "registers")
        img = am.encode_call(p, [1, 2, 3, {"a": 4, "b": 5}, 6.5])
        self.assertEqual(am.decode_call(p, img).args[3],
                         {"a": 4, "b": 5})

    def test_memory_aggregate_stack(self):
        big = rec("Big3", [("a", am.DOUBLE), ("b", am.DOUBLE),
                           ("c", am.DOUBLE)])
        # A 阈值 32 字节：24 字节仍可寄存器
        pA = am.plan_call(A, am.Signature(args=(big,)))
        self.assertEqual(pA.placements[0].kind, "registers")
        # B 阈值 16：24 -> MEMORY 走栈
        pB = am.plan_call(B, am.Signature(args=(big,)))
        self.assertEqual(pB.placements[0].kind, "stack")


class TestEmptyAggregates(unittest.TestCase):
    def test_empty_record_and_zero_array(self):
        e = rec("E", [])
        z = am.ArrayType(am.UINT16, 0)
        for desc in (A, B):
            self.assertEqual(am.layout(e, abi=desc).size, 0)
            self.assertEqual(am.layout(z, abi=desc).size, 0)
            p = am.plan_call(desc, am.Signature(args=(e, z, am.INT32)))
            self.assertEqual(p.placements[0].kind, "none")
            self.assertEqual(p.placements[1].kind, "none")
            self.assertEqual(p.placements[2].int_regs, (0,))
            img = am.encode_call(p, [{}, [], 7])
            d = am.decode_call(p, img)
            self.assertEqual(d.args, [{}, [], 7])


class TestNestingAndScale(unittest.TestCase):
    def _build_deep(self, depth):
        # 每层交替 record/array，最内层 union + 位域
        bf = rec("Bits", [("lo", am.BitField(5)), ("hi", am.BitField(7)),
                          ("flag", am.BitField(1))])
        u = uni("U", [("n", bf, 0), ("v", am.INT64, 1)])
        t = u
        for d in range(depth):
            t = am.ArrayType(t, 2)
            inner = t
            t = rec(f"L{d}", [("x", am.UINT8),
                              ("child", inner),
                              ("y", am.INT16)])
        return t, u

    def test_six_levels_independent_reference(self):
        t, u = self._build_deep(6)

        def make(level):
            leaf = am.Tagged("n", {"lo": 17, "hi": 100, "flag": 1})
            if level == 0:
                return {"x": level & 0xFF, "child": [leaf, leaf],
                        "y": -1000 - level}
            return {"x": level & 0xFF,
                    "child": [make(level - 1), make(level - 1)],
                    "y": -1000 - level}

        val = make(5)
        for desc in (A, B):
            sig = am.Signature(args=(t, am.UINT128, am.PTR), ret=t,
                              cells=(("buf", am.ArrayType(am.UINT8, 8)),))
            p = am.plan_call(desc, sig)
            img = am.encode_call(p, [val, 2 ** 128 - 1, am.CellPtr("buf")],
                                 cell_values={"buf": list(range(8))},
                                 ret_value=val)
            issues, _ = ref.verify_image(
                desc, sig, p, img,
                [val, 2 ** 128 - 1, am.CellPtr("buf")], [],
                {"buf": list(range(8))}, val)
            self.assertEqual(issues, [], "\n".join(issues))
            d = am.decode_call(p, img)
            self.assertEqual(d.args[0], val)
            self.assertEqual(d.ret, val)
            self.assertEqual(d.args[1], 2 ** 128 - 1)
            self.assertEqual(d.args[2], am.CellPtr("buf"))
            # 对齐约束：所有栈偏移/单元偏移满足各自对齐
            for pl in p.placements:
                if pl.kind == "stack":
                    L = am.layout(sig.args[p.placements.index(pl)], abi=desc)
                    self.assertEqual(pl.frame_offset % L.align, 0)

    def test_32_args(self):
        types = []
        values = []
        for i in range(32):
            if i % 3 == 0:
                types.append(am.INT64)
                values.append(-(2 ** 63) + i)
            elif i % 3 == 1:
                types.append(am.DOUBLE)
                values.append(float(i) * 0.5)
            else:
                types.append(am.UINT32)
                values.append(0xFFFF_FFFF - i)
        sig = am.Signature(args=tuple(types))
        for desc in (A, B):
            p = am.plan_call(desc, sig)
            img = am.encode_call(p, values)
            issues, _ = ref.verify_image(desc, sig, p, img, values, [],
                                         {}, None)
            self.assertEqual(issues, [], "\n".join(issues))
            self.assertEqual(am.decode_call(p, img).args, values)


class TestSretCellsSaveArea(unittest.TestCase):
    def setUp(self):
        self.big = rec("BigRet",
                       [("a", am.INT64), ("b", am.INT64), ("c", am.INT32)])
        self.buf_t = am.ArrayType(am.UINT8, 24)

    def test_implicit_sret_threshold_differs(self):
        sig = am.Signature(args=(am.INT32,), ret=self.big)
        pA = am.plan_call(A, sig)   # 20 > 16 -> sret
        pB = am.plan_call(B, sig)   # 20 > 8  -> sret
        self.assertEqual(pA.ret.kind, "sret")
        self.assertEqual(pB.ret.kind, "sret")
        val = {"a": -1, "b": 0xDEAD_BEEF_CAFE, "c": -7}
        for p, desc in ((pA, A), (pB, B)):
            img = am.encode_call(p, [11], ret_value=val)
            issues, _ = ref.verify_image(desc, sig, p, img, [11], [], {}, val)
            self.assertEqual(issues, [], "\n".join(issues))
            self.assertEqual(am.decode_call(p, img).ret, val)
            # 隐藏指针是首参数，指向 $sret 单元
            h = p.ret.hidden_arg
            self.assertEqual(h.path, "$sret.hidden")

    def test_small_return_uses_registers(self):
        small = rec("SmallRet", [("a", am.UINT32), ("b", am.UINT32)])
        p = am.plan_call(B, am.Signature(ret=small))  # 8 <= 8
        self.assertEqual(p.ret.kind, "reg")
        img = am.encode_call(p, [], ret_value={"a": 1, "b": 2})
        self.assertEqual(am.decode_call(p, img).ret, {"a": 1, "b": 2})

    def test_narrow_scalar_return_both_endians(self):
        # 窄于 8 字节的寄存器返回（槽低 n 字节 + 零），两端均往返
        for desc in (A, B):
            for t, v in ((am.UINT8, 200), (am.INT16, -2000),
                         (am.UINT32, 0xDEADBEEF), (am.FLOAT32, -2.5)):
                p = am.plan_call(desc, am.Signature(ret=t))
                img = am.encode_call(p, [], ret_value=v)
                self.assertEqual(am.decode_call(p, img).ret, v,
                                 f"{desc.name} {t.name}")
                # 槽高 8-n 字节必须为零（零扩展约定）
                n = 4 if isinstance(t, am.Float32Type) else t.size
                raws = img.ret_float if isinstance(t, am.Float32Type) \
                    else img.ret_int
                tail = raws[0].to_bytes(8, "little")[n:]
                self.assertEqual(tail, b"\x00" * len(tail))

    def test_explicit_ret_cell(self):
        t = rec("CellRet", [("v", am.INT64)])
        sig = am.Signature(args=(am.PTR,), ret=t,
                           cells=(("out", t),), ret_cell="out")
        p = am.plan_call(A, sig)
        self.assertEqual(p.ret.kind, "cell")
        img = am.encode_call(p, [am.CellPtr("out")], ret_value={"v": -9})
        d = am.decode_call(p, img)
        self.assertEqual(d.ret, {"v": -9})
        self.assertEqual(d.args[0], am.CellPtr("out"))

    def test_save_area_independent_replay(self):
        sig = am.Signature(args=(am.INT64, am.INT64),
                           varargs=(am.DOUBLE, am.UINT8, am.FLOAT32))
        p = am.plan_call(A, sig)
        vals = [10, 20]
        vv = [3.5, 200, -1.25]
        img = am.encode_call(p, vals, vv)
        issues, _ = ref.verify_image(A, sig, p, img, vals, vv, {}, None)
        self.assertEqual(issues, [], "\n".join(issues))
        # 保存区是寄存器原始位的逐字节拷贝（小端位映像）
        sv = p.save
        self.assertEqual(
            img.frame[sv.int_offset:sv.int_offset + 8],
            img.int_regs[sv.int_slots[0]].to_bytes(8, "little"))
        self.assertEqual(
            img.frame[sv.float_offset:sv.float_offset + 8],
            img.float_regs[sv.float_slots[0]].to_bytes(8, "little"))
        d = am.decode_call(p, img)
        self.assertEqual(d.args, vals)
        self.assertEqual(d.varargs, vv)


class TestNaNSignedBits(unittest.TestCase):
    def test_nan_payload_slots_and_stack(self):
        payloads = [0x00000000000001, 0x000F_FFFF_FFFF_FFFF,
                    0x000A_AAAA_AAAA_AAAA]
        rr = rec("NanAgg", [("a", am.DOUBLE), ("b", am.DOUBLE),
                            ("c", am.DOUBLE)])
        for desc in (A, B):
            vals = [nan_with_payload(p) for p in payloads]
            agg = {"a": vals[0], "b": vals[1], "c": vals[2]}
            sig = am.Signature(args=(am.DOUBLE, rr))
            p = am.plan_call(desc, sig)
            img = am.encode_call(p, vals[:1] + [agg])
            issues, _ = ref.verify_image(
                desc, sig, p, img, [vals[0], agg], [], {}, None)
            self.assertEqual(issues, [], "\n".join(issues))
            d = am.decode_call(p, img)
            self.assertEqual(
                struct.pack(desc.endian_char + "d", d.args[0]),
                struct.pack(desc.endian_char + "d", vals[0]))
            for got, exp in zip(d.args[1].values(), vals):
                self.assertEqual(
                    struct.pack(desc.endian_char + "d", got),
                    struct.pack(desc.endian_char + "d", exp))

    def test_signed_boundaries(self):
        cases = [(am.INT8, -128), (am.INT8, 127),
                 (am.INT16, -32768), (am.INT32, -(2 ** 31)),
                 (am.INT64, -(2 ** 63)), (am.INT64, 2 ** 63 - 1),
                 (am.INT128, -(2 ** 127)), (am.INT128, 2 ** 127 - 1)]
        for desc in (A, B):
            for t, v in cases:
                data = am.encode(t, v, abi=desc)
                self.assertEqual(am.decode(t, data, abi=desc), v,
                                 f"{desc.name} {t.name} {v}")


class TestBridge(unittest.TestCase):
    def setUp(self):
        self.bf = rec("BF", [("a", am.BitField(5)),
                             ("b", am.BitField(9, True)),
                             ("tail", am.UINT16)])
        self.u = uni("U", [("i", am.INT32, 0),
                           ("d", am.DOUBLE, 1),
                           ("p", am.ArrayType(am.UINT8, 8), 2)])
        self.big = rec("Big", [("x", am.INT64), ("y", am.INT64),
                               ("z", am.INT32)])

    def test_A_to_B_canonical_and_preserved_bits(self):
        sig = am.Signature(args=(self.bf, self.u, am.INT64, am.UINT128),
                           varargs=(am.FLOAT32, am.UINT8), ret=self.big,
                           cells=(("buf", am.ArrayType(am.UINT8, 8)),))
        pA = am.plan_call(A, sig)
        bf_v = {"a": 20, "b": -200, "tail": 0xCAFE}
        nan = nan_with_payload(0xBEEF)
        u_v = am.Tagged("d", nan)
        args = [bf_v, u_v, -(2 ** 63), 2 ** 128 - 1]
        vv = [nan, 250]
        ret_v = {"x": -1, "y": 0x1122_3344_5566_7788, "z": -3}
        cells = {"buf": list(range(7, -1, -1))}
        imgA = am.encode_call(pA, args, vv, cells, ret_v)
        imgB = am.bridge_call(pA, imgA, B)
        issues, _ = ref.verify_image(B, sig, imgB.plan, imgB,
                                     args, vv, cells, ret_v)
        self.assertEqual(issues, [], "\n".join(issues))
        d = am.decode_call(imgB.plan, imgB)
        self.assertEqual(d.args[0], bf_v)
        self.assertEqual(d.args[1].member, "d")
        self.assertEqual(
            struct.pack(">d", d.args[1].value), struct.pack(">d", nan))
        self.assertEqual(d.args[2], -(2 ** 63))
        self.assertEqual(d.args[3], 2 ** 128 - 1)
        f32_nan = struct.unpack("<f", struct.pack("<f", nan))[0]
        self.assertEqual(struct.pack("<f", d.varargs[0]),
                         struct.pack("<f", f32_nan))
        self.assertEqual(d.varargs[1], 250)
        self.assertEqual(d.ret, ret_v)
        self.assertEqual(d.cells["buf"], cells["buf"])
        self.assertEqual(d.args[0] is not None, True)

    def test_B_to_A_other_union_member(self):
        sig = am.Signature(args=(self.u,))
        pB = am.plan_call(B, sig)
        v = am.Tagged("p", list(b"ABCDEFGH"))
        imgB = am.encode_call(pB, [v])
        imgA = am.bridge_call(pB, imgB, A)
        d = am.decode_call(imgA.plan, imgA)
        self.assertEqual(d.args[0], v)

    def test_bridge_bad_tag_rejected_with_path(self):
        sig = am.Signature(args=(self.u,))
        pA = am.plan_call(A, sig)
        imgA = am.encode_call(pA, [am.Tagged("i", 5)])
        # 篡改标签字节（联合体走寄存器：找到存放标签字节的槽）
        pl = pA.placements[0]
        regs = list(imgA.int_regs)
        slot = pl.int_regs[0]
        raw = regs[slot].to_bytes(8, "little")
        bad = bytes([99]) + raw[1:]
        regs[slot] = int.from_bytes(bad, "little")
        corrupt = am.V2CallImage(tuple(regs), imgA.float_regs, imgA.frame,
                                 imgA.placements, imgA.vararg_placements,
                                 imgA.ret_int, imgA.ret_float, pA)
        with self.assertRaises(am.BridgeError) as ctx:
            am.bridge_call(pA, corrupt, B)
        self.assertEqual(ctx.exception.reason, "tag")
        self.assertIn("arg[0]", ctx.exception.path)

    def test_bridge_unrepresentable_crossing_bitfield(self):
        sig = am.Signature(args=(self.bf,))
        pA = am.plan_call(A, sig)
        imgA = am.encode_call(pA, [{"a": 20, "b": -200, "tail": 1}])
        with self.assertRaises(am.BridgeError) as ctx:
            am.bridge_call(pA, imgA, C)
        self.assertEqual(ctx.exception.reason, "unidentifiable")

    def test_bridge_into_atomicity(self):
        sig = am.Signature(args=(self.bf,))
        pA = am.plan_call(A, sig)
        imgA = am.encode_call(pA, [{"a": 20, "b": -200, "tail": 1}])
        buf = bytearray(b"\x5A" * 32)
        # 失败路径：缓冲原样
        status, plan = am.bridge_into(pA, imgA, C, buf)
        self.assertEqual(status, "rejected")
        self.assertIsNone(plan)
        self.assertEqual(bytes(buf), b"\x5A" * 32)
        # 成功路径：整体写入
        status, plan = am.bridge_into(pA, imgA, B, buf)
        self.assertEqual(status, "ok")
        self.assertEqual(len(buf), plan.frame_size)
        # 内容与重新桥接产出的目标帧逐字节相同
        imgB = am.bridge_call(pA, imgA, B)
        self.assertEqual(bytes(buf), imgB.frame)

    def test_bridge_does_not_require_padding_equality(self):
        # 含内部尾填充的记录：桥接后比较规范值，不比较填充字节
        pad = rec("Pad", [("a", am.UINT8), ("b", am.UINT64)])  # size 16
        sig = am.Signature(args=(pad,))
        pA = am.plan_call(A, sig)
        imgA = am.encode_call(pA, [{"a": 0x77, "b": 0x0102030405060708}])
        imgB = am.bridge_call(pA, imgA, B)
        d = am.decode_call(imgB.plan, imgB)
        self.assertEqual(d.args[0], {"a": 0x77, "b": 0x0102030405060708})
        # 填充字节在两种布局下均被定义为零（约定保留位），但即使有人为
        # 脏填充也不影响规范值：手工把 A 映像槽内填充位改脏后再桥
        regs = list(imgA.int_regs)
        slot = pA.placements[0].int_regs[0]
        raw = regs[slot].to_bytes(8, "little")
        dirty = raw[:1] + b"\xEE" * 3 + raw[4:]
        regs[slot] = int.from_bytes(dirty, "little")
        dirty_img = am.V2CallImage(tuple(regs), imgA.float_regs, imgA.frame,
                                   imgA.placements, imgA.vararg_placements,
                                   imgA.ret_int, imgA.ret_float, pA)
        imgB2 = am.bridge_call(pA, dirty_img, B)
        d2 = am.decode_call(imgB2.plan, imgB2)
        self.assertEqual(d2.args[0]["a"], 0x77)
        self.assertEqual(d2.args[0]["b"], 0x0102030405060708)


class TestBudgetAndDeterminism(unittest.TestCase):
    def test_budget_exhausted_distinct_from_rejection(self):
        sig = am.Signature(args=(am.ArrayType(am.UINT8, 4096),))
        with self.assertRaises(am.BudgetExhausted):
            am.plan_call(A, sig, budget=1)
        # 充足预算下正常
        p = am.plan_call(A, sig)
        self.assertEqual(p.status, "ok")
        # 桥接预算
        pp = am.plan_call(A, am.Signature(args=(am.INT64,)))
        img = am.encode_call(pp, [1])
        with self.assertRaises(am.BridgeError) as ctx:
            am.bridge_call(pp, img, B, budget=0)
        self.assertEqual(ctx.exception.reason, "budget")

    def test_deterministic_repetition(self):
        sig = am.Signature(args=(rec("D", [("a", am.UINT8)]),),
                           varargs=(am.FLOAT32,))
        i1 = am.encode_call(am.plan_call(A, sig), [{"a": 3}], [2.5])
        i2 = am.encode_call(am.plan_call(A, sig), [{"a": 3}], [2.5])
        self.assertEqual(i1.frame, i2.frame)
        self.assertEqual(i1.int_regs, i2.int_regs)
        self.assertEqual(i1.float_regs, i2.float_regs)

    def test_arg_count_limit(self):
        with self.assertRaises(am.InvalidTypeError):
            am.plan_call(A, am.Signature(args=tuple([am.INT64] * 33)))


if __name__ == "__main__":
    unittest.main()
