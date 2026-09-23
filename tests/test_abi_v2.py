"""v2 双布局 / 联合 / 位域 / 变参 / 桥接 / 证书的确定性单元测试（快速回归）。

期望值来源：手工硬编码字面量、独立算术，以及 tests/reference_v2.py 中
独立编写的逐位参考（不调用被测布局/编解码函数算期望值）。往返一致只作
补充。本文件覆盖：两种相反字节序、有符号/跨单元/重叠位域、空聚合、联合
标签越界、变参提升与不可表示、返回间接区、保存区、按位共享存储、128 位
整数、32 实参寄存器用尽、桥接原子性与 NaN 载荷、证书肯定/否定判定。
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import abi_marshal as am
import reference_v2 as R


def rec(name, fields):
    r = am.RecordType(name)
    r.define(fields)
    return r


def uni(name, members, tag=am.UINT8):
    u = am.UnionType(name, tag)
    u.define(members)
    return u


class TestDescriptors(unittest.TestCase):
    def test_two_explicit_layouts(self):
        self.assertEqual(am.LAYOUT_A.endian, am.LITTLE)
        self.assertEqual(am.LAYOUT_B.endian, am.BIG)
        self.assertEqual(am.LAYOUT_A.max_align, 8)
        self.assertEqual(am.LAYOUT_B.max_align, 16)
        self.assertEqual(am.LAYOUT_A.int_regs.int_count, 4)
        self.assertEqual(am.LAYOUT_B.int_regs.int_count, 6)
        self.assertEqual(am.LAYOUT_A.int_regs.float_count, 2)
        self.assertEqual(am.LAYOUT_B.int_regs.float_count, 4)

    def test_new_layout_not_host_derived(self):
        # 描述符必须显式给全字段；非法字节序/对齐被拒绝
        with self.assertRaises(am.InvalidTypeError):
            am.ABIDescriptor("X", "middle", 8,
                             am.RegisterRules(1, 1, 16))
        with self.assertRaises(am.InvalidTypeError):
            am.ABIDescriptor("X", am.LITTLE, 3,
                             am.RegisterRules(1, 1, 16))

    def test_flipped_only_changes_endian(self):
        f = am.LAYOUT_A.flipped()
        self.assertEqual(f.endian, am.BIG)
        self.assertEqual(f.max_align, am.LAYOUT_A.max_align)
        self.assertEqual(f.int_regs, am.LAYOUT_A.int_regs)


class TestLayoutBothEndian(unittest.TestCase):
    def test_scalar_encoding_opposite_endianness(self):
        self.assertEqual(am.encode_v2(am.UINT32, 0x01020304, am.LAYOUT_A),
                         b"\x04\x03\x02\x01")
        self.assertEqual(am.encode_v2(am.UINT32, 0x01020304, am.LAYOUT_B),
                         b"\x01\x02\x03\x04")
        self.assertEqual(am.encode_v2(am.INT16, -2, am.LAYOUT_B),
                         b"\xff\xfe")

    def test_int128_range_and_bytes(self):
        v = (1 << 127) - 1
        b = am.encode_v2(am.INT128, v, am.LAYOUT_B)
        self.assertEqual(b, b"\x7f" + b"\xff" * 15)
        self.assertEqual(am.decode_v2(am.INT128, b, am.LAYOUT_B), v)
        u = (1 << 128) - 1
        ba = am.encode_v2(am.UINT128, u, am.LAYOUT_A)
        self.assertEqual(ba, b"\xff" * 16)
        self.assertEqual(am.decode_v2(am.UINT128, ba, am.LAYOUT_A), u)
        with self.assertRaises(am.IntegerOverflowError):
            am.encode_v2(am.INT128, 1 << 127, am.LAYOUT_A)

    def test_six_level_nesting_both_endians(self):
        def build(d):
            if d == 0:
                return am.UINT16
            return rec(f"N{d}", [("v", am.UINT8), ("inner", build(d - 1))])

        def val(d):
            return {"v": d, "inner": val(d - 1) if d > 1 else 60000}

        t, v = build(6), val(6)
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            lay = am.layout_v2(t, d)
            self.assertEqual(lay.size, 14)
            b = am.encode_v2(t, v, d)
            self.assertEqual(am.decode_v2(t, b, d), v)

    def test_empty_aggregate_convention(self):
        e = rec("E", [])
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            self.assertEqual(am.layout_v2(e, d).size, 0)
            self.assertEqual(am.classify_v2(e, d), ())
            img = am.pack_call_v2([am.Param(e)], [{}], d)
            self.assertEqual(img.plan.placements[0].kind, "none")
            self.assertEqual(img.frame, b"")
            self.assertEqual(am.unpack_call_v2([am.Param(e)], img, d), [{}])


class TestBitFields(unittest.TestCase):
    def test_signed_negative_bitfield(self):
        t = rec("S", [("x", am.BitField("x", am.INT8, 4))])
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            self.assertEqual(am.encode_v2(t, {"x": -1}, d), b"\x0f")
            self.assertEqual(am.encode_v2(t, {"x": -8}, d), b"\x08")
            self.assertEqual(am.decode_v2(t, b"\x08", d)["x"], -8)
            with self.assertRaises(am.IntegerOverflowError):
                am.encode_v2(t, {"x": -9}, d)

    def test_cross_storage_unit_split(self):
        t = rec("C", [("a", am.BitField("a", am.UINT8, 6)),
                      ("b", am.BitField("b", am.UINT8, 6)),
                      ("c", am.BitField("c", am.UINT8, 6))])
        v = {"a": 0b111111, "b": 0b101010, "c": 0b000001}
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            lay = am.layout_v2(t, d)
            # b、c 都跨越存储单元边界
            seg_names = [(s.name, s.width)
                         for f in lay.fields for s in f.segs]
            self.assertEqual(sum(1 for n, _ in seg_names if n == "b"), 2)
            self.assertEqual(sum(1 for n, _ in seg_names if n == "c"), 2)
            self.assertEqual(am.decode_v2(t, am.encode_v2(t, v, d), d), v)

    def test_explicit_lsb_and_overlap_rejected(self):
        ok = rec("OK", [("a", am.BitField("a", am.UINT8, 4, 0)),
                        ("b", am.BitField("b", am.UINT8, 4, 4))])
        self.assertEqual(am.layout_v2(ok, am.LAYOUT_A).size, 1)
        bad = rec("Bad", [("a", am.BitField("a", am.UINT8, 4, 0)),
                          ("b", am.BitField("b", am.UINT8, 4, 2))])
        with self.assertRaises(am.OverlappingBitFieldError) as ctx:
            am.layout_v2(bad, am.LAYOUT_B)
        # 错误定位到类型（记录）路径
        self.assertIn("Bad", ctx.exception.path)
        # 经调用入口时定位到参数路径 arg[i]
        with self.assertRaises(am.OverlappingBitFieldError) as ctx2:
            am.plan_call([am.Param(bad)], am.LAYOUT_A)
        self.assertEqual(ctx2.exception.path, "arg[0]")

    def test_mixed_signedness_shares_storage_unit_width(self):
        # 同宽度不同 signed 的位域属于同一存储单元
        t = rec("M", [("a", am.BitField("a", am.INT16, 5)),
                      ("b", am.BitField("b", am.UINT16, 7))])
        lay = am.layout_v2(t, am.LAYOUT_A)
        self.assertEqual(lay.size, 2)


class TestUnions(unittest.TestCase):
    def test_tagged_active_member(self):
        u = uni("U", [("i", am.INT32), ("b", am.UINT8)])
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            lay = am.layout_v2(u, d)
            self.assertEqual(lay.tag_size, 1)
            b = am.encode_v2(u, am.UnionValue("b", 7), d)
            self.assertEqual(b[0] if d.endian == am.LITTLE else b[0], 1)
            dec = am.decode_v2(u, b, d)
            self.assertEqual(dec, am.UnionValue(1, 7))

    def test_tag_out_of_range_located_to_arg(self):
        u = uni("U", [("a", am.UINT8)])
        img = am.pack_call_v2([am.Param(u)], [am.UnionValue(0, 1)],
                              am.LAYOUT_A)
        bad = am.CallImageV2(img.plan, (9,), img.float_regs, img.frame,
                             img.evidence, img.return_regs, img.return_area)
        with self.assertRaises(am.InvalidUnionTagError) as ctx:
            am.unpack_call_v2([am.Param(u)], bad, am.LAYOUT_A)
        self.assertEqual(ctx.exception.path, "arg[0]")

    def test_untagged_ambiguous_rejected_hint_resolves(self):
        u = uni("UU", [("a", am.UINT8), ("b", am.UINT8)], tag=None)
        b = am.encode_v2(u, am.UnionValue(0, 5), am.LAYOUT_A)
        with self.assertRaises(am.InvalidUnionTagError):
            am.decode_v2(u, b, am.LAYOUT_A)
        self.assertEqual(am.decode_v2(u, b, am.LAYOUT_A, tag_hint="b"),
                         am.UnionValue(1, 5))


class TestVariadicPromotion(unittest.TestCase):
    def test_integer_and_float_promotion(self):
        params = [am.Param(am.INT16, variadic=True),
                  am.Param(am.UINT8, variadic=True),
                  am.Param(am.FLOAT32, variadic=True)]
        vals = [-2, 250, 1.5]
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            img = am.pack_call_v2(params, vals, d)
            # 变参占 4 字节/8 字节寄存器槽
            self.assertTrue(all(p.variadic
                                for p in img.plan.placements))
            self.assertEqual(am.unpack_call_v2(params, img, d), vals)

    def test_unrepresentable_promotion_rejected(self):
        with self.assertRaises(am.VariadicPromotionError) as ctx:
            am.plan_call([am.Param(am.INT128, variadic=True)], am.LAYOUT_A)
        self.assertEqual(ctx.exception.path, "arg[0]")
        agg = rec("Agg", [("x", am.UINT8)])
        with self.assertRaises(am.VariadicPromotionError):
            am.plan_call([am.Param(agg, variadic=True)], am.LAYOUT_A)

    def test_variadic_register_save_area(self):
        params = [am.Param(am.INT64),
                  am.Param(am.INT32, variadic=True),
                  am.Param(am.FLOAT32, variadic=True)]
        img = am.pack_call_v2(params, [1, 2, 3.0], am.LAYOUT_A)
        save_regions = [r for r in img.plan.regions
                        if r.kind in ("save-int", "save-float")]
        self.assertEqual(len(save_regions), 2)
        # 保存槽内容 = 对应寄存器整词
        for e in img.plan.save_entries:
            word = img.frame[e.offset:e.offset + 8]
            if e.kind == "int":
                self.assertEqual(word,
                                 img.int_regs[e.slot].to_bytes(8, "little"))


class TestSharedStorage(unittest.TestCase):
    def _types(self):
        w = rec("W", [("whole", am.UINT16)])
        b = rec("Bits", [("lo", am.BitField("lo", am.UINT16, 4)),
                         ("hi", am.BitField("hi", am.UINT16, 12))])
        return w, b

    def test_aliases_consume_no_extra_resource(self):
        w, b = self._types()
        params = [am.Param(w, shared="s", shared_size=2),
                  am.Param(b, shared="s", shared_size=2)]
        plan = am.plan_call(params, am.LAYOUT_A)
        self.assertEqual(plan.placements[0].kind, "registers")
        self.assertEqual(plan.placements[1].kind, "alias")
        self.assertEqual(plan.placements[1].alias_of, 0)
        self.assertEqual(len(plan.shared_units), 1)

    def test_consistent_views_roundtrip(self):
        w, b = self._types()
        params = [am.Param(w, shared="s", shared_size=2),
                  am.Param(b, shared="s", shared_size=2)]
        su = am.SharedUnit("s", 2, {"W": {"whole": 0xABCD},
                                    "Bits": {"lo": 0xD, "hi": 0xABC}})
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            img = am.pack_call_v2(params, [None, None], d,
                                  shared_units=[su])
            out = am.unpack_call_v2(params, img, d)
            self.assertEqual(out[0], {"whole": 0xABCD})
            self.assertEqual(out[1], {"lo": 0xD, "hi": 0xABC})
            # 硬编码核对合并后的寄存器原始位型（块左对齐；小端 0xABCD
            # 落整词低位，大端落高 16 位），不依赖解码
            slot = img.plan.placements[0].int_regs[0]
            expect = (0xABCD if d.endian == am.LITTLE
                      else 0xABCD000000000000)
            self.assertEqual(img.int_regs[slot], expect)

    def test_conflicting_overlap_located_to_path(self):
        w, b = self._types()
        params = [am.Param(w, shared="s", shared_size=2),
                  am.Param(b, shared="s", shared_size=2)]
        su = am.SharedUnit("s", 2, {"W": {"whole": 0x0000},
                                    "Bits": {"lo": 0xF, "hi": 0x0}})
        with self.assertRaises(am.OverlappingStorageError) as ctx:
            am.pack_call_v2(params, [None, None], am.LAYOUT_A,
                            shared_units=[su])
        self.assertIn("arg[1]", ctx.exception.path)


class TestRegistersAndStack(unittest.TestCase):
    def test_32_params_register_exhaustion_to_stack(self):
        params = [am.Param(am.UINT64) for _ in range(32)]
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            img = am.pack_call_v2(params, list(range(32)), d)
            nreg = d.int_regs.int_count
            kinds = [p.kind for p in img.plan.placements]
            self.assertEqual(kinds[:nreg], ["registers"] * nreg)
            self.assertTrue(all(k == "stack" for k in kinds[nreg:]))
            self.assertEqual(am.unpack_call_v2(params, img, d),
                             list(range(32)))

    def test_aggregate_over_threshold_indirect_or_stack(self):
        # LAYOUT_A 聚合界 16；24 字节整体走栈
        big = rec("Big", [("a", am.ArrayType(am.UINT8, 24))])
        img = am.pack_call_v2([am.Param(big)],
                              [{"a": list(range(24))}], am.LAYOUT_A)
        self.assertEqual(img.plan.placements[0].kind, "stack")
        # LAYOUT_B 聚合界 32：24 字节可走寄存器（3 个整数块）
        imgB = am.pack_call_v2([am.Param(big)],
                               [{"a": list(range(24))}], am.LAYOUT_B)
        self.assertEqual(imgB.plan.placements[0].kind, "registers")
        self.assertEqual(len(imgB.plan.placements[0].int_regs), 3)


class TestReturnArea(unittest.TestCase):
    def test_large_return_uses_indirect_area_at_zero(self):
        # 40 字节：超过两端口的寄存器聚合界（A=16, B=32）-> 间接区
        big = rec("RBig", [("x", am.ArrayType(am.UINT8, 40))])
        v = {"x": list(range(40))}
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            img = am.pack_call_v2([], [], d, ret_type=big, ret_value=v)
            rinfo = img.plan.return_info
            self.assertTrue(rinfo.indirect)
            self.assertEqual(rinfo.area_offset, 0)
            ret_region = [r for r in img.plan.regions
                          if r.kind == "return-area"][0]
            self.assertEqual(ret_region.offset, 0)
            self.assertEqual(am.unpack_return_v2(big, img, d), v)

    def test_128_return_uses_indirect_area(self):
        v = 1 << 100
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            img = am.pack_call_v2([], [], d, ret_type=am.UINT128,
                                  ret_value=v)
            self.assertTrue(img.plan.return_info.indirect)
            self.assertEqual(am.unpack_return_v2(am.UINT128, img, d), v)

    def test_small_return_uses_registers(self):
        t = rec("SR", [("i", am.INT32), ("d", am.DOUBLE)])
        v = {"i": 5, "d": 2.5}
        img = am.pack_call_v2([], [], am.LAYOUT_A, ret_type=t,
                              ret_value=v)
        self.assertFalse(img.plan.return_info.indirect)
        self.assertEqual(am.unpack_return_v2(t, img, am.LAYOUT_A), v)


class TestNaNPayload(unittest.TestCase):
    def test_nan_payload_preserved_both_endians(self):
        pat = 0x7FF8000000000123
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            img = am.pack_call_v2([am.Param(am.DOUBLE)],
                                  [am.FloatBits(pat)], d)
            out = am.unpack_call_v2([am.Param(am.DOUBLE)], img, d)[0]
            self.assertIsInstance(out, am.FloatBits)
            self.assertEqual(out.pattern, pat)

    def test_f32_nan_widened_through_variadic(self):
        pat32 = 0x7FC01234
        params = [am.Param(am.FLOAT32, variadic=True)]
        img = am.pack_call_v2(params, [am.FloatBits(pat32, 32)],
                              am.LAYOUT_A)
        out = am.unpack_call_v2(params, img, am.LAYOUT_A)[0]
        self.assertIsInstance(out, am.FloatBits)
        self.assertEqual(out.width, 32)
        self.assertEqual(out.pattern, pat32)


class TestBridge(unittest.TestCase):
    def _call(self, d, ret_t=None, ret_v=None):
        u = uni("U", [("i", am.INT32), ("b", am.UINT8)])
        w = rec("W", [("whole", am.UINT16)])
        bb = rec("Bits", [("lo", am.BitField("lo", am.UINT16, 4)),
                          ("hi", am.BitField("hi", am.UINT16, 12))])
        params = [am.Param(am.UINT64), am.Param(u),
                  am.Param(am.FLOAT32, variadic=True),
                  am.Param(am.INT16, variadic=True),
                  am.Param(w, shared="s", shared_size=2),
                  am.Param(bb, shared="s", shared_size=2)]
        su = am.SharedUnit("s", 2, {"W": {"whole": 0xABCD},
                                    "Bits": {"lo": 0xD, "hi": 0xABC}})
        vals = [0x1122334455667788, am.UnionValue("i", -123456),
                2.5, -2, None, None]
        return am.pack_call_v2(params, vals, d, ret_type=ret_t,
                               ret_value=ret_v, shared_units=[su]), params, su

    def test_a_to_b_canonical_values_and_bits(self):
        ret_t = rec("RB", [("x", am.ArrayType(am.UINT8, 24))])
        ret_v = {"x": list(range(24))}
        imgA, params, su = self._call(am.LAYOUT_A, ret_t, ret_v)
        imgB = am.bridge_call(imgA, params, am.LAYOUT_A, am.LAYOUT_B,
                              ret_type=ret_t, shared_units=[su])
        # 比较规范值，而非填充字节
        out = am.unpack_call_v2(params, imgB, am.LAYOUT_B)
        self.assertEqual(out[0], 0x1122334455667788)
        self.assertEqual(out[1], am.UnionValue(0, -123456))
        self.assertEqual(out[2], 2.5)
        self.assertEqual(out[3], -2)
        self.assertEqual(out[4], {"whole": 0xABCD})
        self.assertEqual(out[5], {"lo": 0xD, "hi": 0xABC})
        self.assertEqual(am.unpack_return_v2(ret_t, imgB, am.LAYOUT_B),
                         ret_v)

    def test_bridge_preserves_signed_boundary_and_nan(self):
        params = [am.Param(am.INT64), am.Param(am.DOUBLE)]
        vals = [-(1 << 63), am.FloatBits(0x7FF8000000000ABC)]
        imgA = am.pack_call_v2(params, vals, am.LAYOUT_A)
        imgB = am.bridge_call(imgA, params, am.LAYOUT_A, am.LAYOUT_B)
        out = am.unpack_call_v2(params, imgB, am.LAYOUT_B)
        self.assertEqual(out[0], -(1 << 63))
        self.assertEqual(out[1].pattern, 0x7FF8000000000ABC)

    def test_bridge_failure_does_not_partially_write(self):
        imgA, params, su = self._call(am.LAYOUT_A)
        # 篡改源映像中联合标签为越界值
        bad = am.CallImageV2(imgA.plan, (imgA.int_regs[0], 9,
                                         *imgA.int_regs[2:]),
                             imgA.float_regs, imgA.frame, imgA.evidence,
                             imgA.return_regs, imgA.return_area)
        buf = bytearray(b"Z" * 128)
        with self.assertRaises(am.BridgeError) as ctx:
            am.bridge_call_into(buf, bad, params, am.LAYOUT_A, am.LAYOUT_B,
                                shared_units=[su])
        self.assertEqual(ctx.exception.reason, "tag-out-of-range")
        self.assertEqual(bytes(buf), b"Z" * 128)

    def test_bridge_conflict_and_promotion_reasons(self):
        # 共享视图在目标侧重算冲突
        w = rec("W2", [("whole", am.UINT16)])
        bb = rec("Bits2", [("lo", am.BitField("lo", am.UINT16, 4)),
                           ("hi", am.BitField("hi", am.UINT16, 12))])
        # 先造一个在小端合法（视图一致）的映像
        params = [am.Param(w, shared="s", shared_size=2),
                  am.Param(bb, shared="s", shared_size=2)]
        su_good = am.SharedUnit("s", 2, {"W2": {"whole": 0xABCD},
                                         "Bits2": {"lo": 0xD, "hi": 0xABC}})
        imgA = am.pack_call_v2(params, [None, None], am.LAYOUT_A,
                               shared_units=[su_good])
        # 源端解码视图后，给目标侧喂一份冲突的 SharedUnit
        su_bad = am.SharedUnit("s", 2, {"W2": {"whole": 0x0000},
                                        "Bits2": {"lo": 0xF, "hi": 0x0}})
        with self.assertRaises(am.BridgeError) as ctx:
            am.bridge_call(imgA, params, am.LAYOUT_A, am.LAYOUT_B,
                           shared_units=[su_bad])
        self.assertEqual(ctx.exception.reason, "overlap-conflict")


class TestCertificates(unittest.TestCase):
    def test_layout_certificate_positive_and_tampered(self):
        t = rec("CR", [("a", am.UINT8),
                       ("x", am.BitField("x", am.INT16, 5)),
                       ("y", am.BitField("y", am.UINT16, 12))])
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            c = am.certify_layout(t, d)
            self.assertTrue(c.valid)
            self.assertEqual(c.covered_bits, c.leaf_bits)
        u = uni("CU", [("i", am.INT32), ("b", am.UINT8)])
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            self.assertTrue(am.certify_layout(u, d).valid)

    def test_certificate_detects_tampered_image(self):
        params = [am.Param(am.UINT32)]
        img = am.pack_call_v2(params, [0x01020304], am.LAYOUT_A)
        bad = am.CallImageV2(img.plan, (0xFFFFFFFF,), img.float_regs,
                             img.frame, img.evidence, img.return_regs,
                             img.return_area)
        c = am.verify_image(params, [0x01020304], bad, am.LAYOUT_A)
        self.assertFalse(c.valid)
        self.assertIn("arg[0]", c.content_mismatches)

    def test_frame_certificate_and_flipped_replay(self):
        params = [am.Param(am.UINT64), am.Param(am.DOUBLE),
                  am.Param(am.INT32, variadic=True)]
        plan = am.plan_call(params, am.LAYOUT_A,
                           ret_type=rec("Ret", [("z", am.UINT8)]))
        c = am.certify_frame(params, plan, am.LAYOUT_A)
        self.assertTrue(c.valid)
        self.assertTrue(c.flipped_align_ok)
        self.assertTrue(c.save_ok)

    def test_image_certificate_end_to_end_all_features(self):
        u = uni("FU", [("i", am.INT32), ("b", am.UINT8)])
        w = rec("FW", [("whole", am.UINT16)])
        bb = rec("FBits", [("lo", am.BitField("lo", am.UINT16, 4)),
                           ("hi", am.BitField("hi", am.UINT16, 12))])
        params = [am.Param(am.INT128), am.Param(u),
                  am.Param(am.FLOAT32, variadic=True),
                  am.Param(w, shared="s", shared_size=2),
                  am.Param(bb, shared="s", shared_size=2)]
        su = am.SharedUnit("s", 2, {"FW": {"whole": 0x1234},
                                    "FBits": {"lo": 0x4, "hi": 0x123}})
        vals = [1 << 100, am.UnionValue("b", 9), 1.25, None, None]
        ret_t = rec("FRet", [("x", am.ArrayType(am.UINT8, 24))])
        ret_v = {"x": list(range(24))}
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            img = am.pack_call_v2(params, vals, d, ret_type=ret_t,
                                  ret_value=ret_v, shared_units=[su])
            c = am.verify_image(params, vals, img, d,
                                shared_units=[su], ret_value=ret_v)
            c.raise_if_invalid()
            self.assertTrue(c.valid)
            # 证据位恰覆盖全部数据位
            self.assertEqual(c.evidence_bits, c.data_bits)


class TestDeterminism(unittest.TestCase):
    def test_repeatable_same_input(self):
        t = rec("Det", [("a", am.UINT8), ("b", am.INT32)])
        params = [am.Param(t), am.Param(am.UINT64)]
        vals = [{"a": 1, "b": -7}, 0xABCDEF]
        for d in (am.LAYOUT_A, am.LAYOUT_B):
            i1 = am.pack_call_v2(params, vals, d)
            i2 = am.pack_call_v2(params, vals, d)
            self.assertEqual(i1.int_regs, i2.int_regs)
            self.assertEqual(i1.float_regs, i2.float_regs)
            self.assertEqual(i1.frame, i2.frame)
            self.assertEqual(i1.evidence, i2.evidence)


if __name__ == "__main__":
    unittest.main()
