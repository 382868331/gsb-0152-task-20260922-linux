"""扩展验证（独立逐位参考差分）。

与快速回归分开运行：

    python -m unittest tests.test_abi_v2_extended -v   # 仅扩展验证
    python -m unittest discover -s tests -v            # 全部（仍然很快）

内容：
1. 小参考域**完整穷举**：u8/s8 全部取值、有符号/无符号位域笛卡尔积、
   带标签联合的标签×边界值——每个值都用 tests/reference_v2.py 的独立
   逐位参考核对，并在两种字节序下核对；
2. 随机嵌套聚合（≥6 层、位域跨单元、空聚合、联合）编码与完整帧编组，
   期望值全部由独立参考计算（绝不调用被测求解器算期望）；
3. 寄存器用尽转栈、A<->B 桥接后比较**规范值与约定保留位**（NaN 载荷、
   有符号边界），不把填充字节相等当标准；
4. 证书在每个随机例子上独立复核；
5. 一个 32 实参 + 128 位整数 + 6 层嵌套 + 共享单元 + 变参 + 大返回的
   主规模非退化端到端例子。
"""

import os
import random
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import abi_marshal as am
import reference_v2 as R

DESCS = (am.LAYOUT_A, am.LAYOUT_B)


# ---------------------------------------------------------------------------
# 1) 小参考域完整穷举
# ---------------------------------------------------------------------------

class TestExhaustiveSmallDomain(unittest.TestCase):
    def test_all_u8_and_s8_values_both_endians(self):
        for bits, signed in ((8, False), (8, True)):
            spec = ("int", bits, signed)
            t = R.build_type(spec)
            lo = -(1 << 7) if signed else 0
            hi = (1 << 7) - 1 if signed else 255
            for v in range(lo, hi + 1):
                for d in DESCS:
                    got = am.encode_v2(t, v, d)
                    want = R.ref_encode(spec, v, d.endian, d.max_align)
                    self.assertEqual(got, want, (v, d.name))
                    self.assertEqual(am.decode_v2(t, got, d), v)

    def test_bitfield_cartesian_product_both_endians(self):
        # a: u3 (0..7) × b: s3 (-2..1) ，64 个组合全部穷举
        spec = ("rec", [("bf", "a", 8, 3, None, 0),
                        ("bf", "b", 8, 3, None, 1)])
        t = R.build_type(spec)
        for a in range(8):
            for b in range(-2, 2):
                v = {"a": a, "b": b}
                for d in DESCS:
                    got = am.encode_v2(t, v, d)
                    want = R.ref_encode(spec, v, d.endian, d.max_align)
                    self.assertEqual(got, want, (v, d.name))
                    self.assertEqual(am.decode_v2(t, got, d), v)

    def test_tagged_union_tags_and_boundary_values(self):
        # 成员 i:s16 与 b:u8；标签全枚举 × 每成员边界值
        spec = ("union", [("i", ("int", 16, 1)), ("b", ("int", 8, 0))], True)
        t = R.build_type(spec)
        cases = [(0, -32768), (0, -1), (0, 0), (0, 32767),
                 (1, 0), (1, 255)]
        for idx, memv in cases:
            uv = am.UnionValue(idx, memv)
            for d in DESCS:
                got = am.encode_v2(t, uv, d)
                want = R.ref_encode(spec, uv, d.endian, d.max_align)
                self.assertEqual(got, want, (idx, memv, d.name))
                self.assertEqual(am.decode_v2(t, got, d), uv)

    def test_explicit_lsb_bitfields_full_patterns(self):
        # 同字节内两个 4 位域：0..15 × 0..15 笛卡尔积
        spec = ("rec", [("bf", "lo", 8, 4, 0, 0),
                        ("bf", "hi", 8, 4, 4, 0)])
        t = R.build_type(spec)
        for lo in range(16):
            for hi in range(16):
                v = {"lo": lo, "hi": hi}
                for d in DESCS:
                    self.assertEqual(
                        am.encode_v2(t, v, d),
                        R.ref_encode(spec, v, d.endian, d.max_align))


# ---------------------------------------------------------------------------
# 2/3) 随机嵌套聚合差分
# ---------------------------------------------------------------------------

def _rnd_spec(rng, depth):
    g = rng.random
    if depth <= 0 or g() < 0.32:
        return rng.choice([
            ("int", 8, 1), ("int", 16, 0), ("int", 32, 1),
            ("int", 64, 0), ("int", 128, 1), ("int", 128, 0),
            ("f", 64), ("f", 32), ("p",)])
    k = g()
    if k < 0.22:
        return ("arr", _rnd_spec(rng, depth - 1), rng.randint(0, 4))
    if k < 0.86:
        fields = []
        for nm in range(rng.randint(0, 5)):
            if g() < 0.4:
                base = rng.choice([8, 16, 32])
                fields.append(("bf", f"b{nm}", base,
                               rng.randint(1, base),
                               rng.choice([None, None, None, 0]),
                               rng.choice([0, 1])))
            else:
                fields.append((f"f{nm}", _rnd_spec(rng, depth - 1)))
        return ("rec", fields)
    members = [(f"m{j}", _rnd_spec(rng, depth - 1))
               for j in range(rng.randint(1, 3))]
    return ("union", members, True)


def _rnd_val(rng, spec):
    t = spec[0]
    if t == "int":
        b = spec[1]
        if spec[2]:
            return rng.choice([-(1 << (b - 1)), -1, 0, 1,
                               (1 << (b - 1)) - 1,
                               rng.randint(-(1 << (b - 1)),
                                           (1 << (b - 1)) - 1)])
        return rng.choice([0, (1 << b) - 1, rng.randint(0, (1 << b) - 1)])
    if t == "f":
        if spec[1] == 32:
            # 取成 binary32 可精确表示的值（f32 域内取值），使编解码往返严格相等
            x = rng.choice([1.5, -2.25, 0.0, -0.0, rng.uniform(-1e9, 1e9)])
            return struct.unpack(">f", struct.pack(">f", x))[0]
        return rng.choice([1.5, -2.25, 0.0, -0.0, rng.uniform(-1e9, 1e9)])
    if t == "p":
        return rng.choice([0, 2**64 - 1, rng.randint(0, 2**64 - 1)])
    if t == "arr":
        return [_rnd_val(rng, spec[1]) for _ in range(spec[2])]
    if t == "rec":
        out = {}
        for f in spec[1]:
            if f[0] == "bf":
                _, name, base, w, _lsb, sg = f
                lo = -(1 << (w - 1)) if sg else 0
                hi = (1 << (w - 1)) - 1 if sg else (1 << w) - 1
                out[name] = rng.randint(lo, hi)
            else:
                out[f[0]] = _rnd_val(rng, f[1])
        return out
    idx = rng.randrange(len(spec[1]))
    return am.UnionValue(idx, _rnd_val(rng, spec[1][idx][1]))


class TestRandomDifferential(unittest.TestCase):
    N_SPEC = 220
    N_FRAME = 90

    def test_random_encoding_matches_reference(self):
        rng = random.Random(20260924)
        mism = 0
        checked = 0
        for _ in range(self.N_SPEC):
            spec = _rnd_spec(rng, 6)
            try:
                value = _rnd_val(rng, spec)
                t = R.build_type(spec)
            except (am.InvalidTypeError, am.OverlappingBitFieldError):
                continue
            for d in DESCS:
                try:
                    want = R.ref_encode(spec, value, d.endian, d.max_align)
                except am.OverlappingBitFieldError:
                    with self.assertRaises(am.OverlappingBitFieldError):
                        am.layout_v2(t, d)
                    continue
                got = am.encode_v2(t, value, d)
                checked += 1
                if got != want:
                    mism += 1
                    self.assertEqual(got, want, (spec, d.name, value))
                # 几何证书 + 往返
                cert = am.certify_layout(t, d)
                self.assertTrue(cert.valid, (spec, d.name))
                self.assertEqual(am.decode_v2(t, got, d), value)
        self.assertGreater(checked, 300)
        self.assertEqual(mism, 0)

    def test_random_frame_matches_reference(self):
        rng = random.Random(424242)
        checked = 0

        def frame_spec(depth):
            g = rng.random
            if depth <= 0 or g() < 0.5:
                return rng.choice([
                    ("int", 8, 1), ("int", 16, 0), ("int", 32, 1),
                    ("int", 64, 0), ("f", 64), ("f", 32), ("p",),
                    ("rec", [("a", ("int", 8, 0)), ("d", ("f", 64))]),
                    ("rec", [("x", ("int", 64, 1)), ("y", ("int", 64, 0))]),
                    ("arr", ("int", 8, 0), rng.randint(2, 5))])
            if g() < 0.6:
                return ("rec", [(f"f{k}", frame_spec(depth - 1))
                                for k in range(rng.randint(1, 3))])
            return ("arr", frame_spec(depth - 1), rng.randint(1, 4))

        for _ in range(self.N_FRAME):
            nargs = rng.randint(1, 14)
            specs = [frame_spec(2) for _ in range(nargs)]
            variadic = [rng.random() < 0.35 and sp[0] in ("int", "f", "p")
                        for sp in specs]
            ok = all((not v) or not (sp[0] == "int" and sp[1] == 128)
                     for sp, v in zip(specs, variadic))
            if not ok:
                continue
            try:
                types = [R.build_type(sp) for sp in specs]
                vals = [_rnd_val(rng, sp) for sp in specs]
            except am.InvalidTypeError:
                continue
            args = list(zip(specs, variadic))
            params = [am.Param(ty, variadic=v)
                      for ty, v in zip(types, variadic)]
            for d in DESCS:
                try:
                    img = am.pack_call_v2(params, vals, d)
                    ri, rf, fr, _rp, plan = R.ref_pack(args, vals, d)
                except am.ABIError:
                    continue
                checked += 1
                self.assertEqual(list(img.int_regs), ri)
                fpats = [None if v is None else
                         v.pattern if isinstance(v, am.FloatBits)
                         else struct.unpack(">Q", struct.pack(">d", v))[0]
                         for v in img.float_regs]
                self.assertEqual(fpats, rf)
                self.assertEqual(img.frame, fr)
                self.assertEqual(img.plan.frame_size, plan["frame_size"])
                # 解码回的规范值必须等于输入
                self.assertEqual(
                    am.unpack_call_v2(params, img, d), vals)
                # 帧几何证书 + 内容证书
                fc = am.certify_frame(params, img.plan, d)
                self.assertTrue(fc.valid)
                ic = am.verify_image(params, vals, img, d)
                self.assertTrue(ic.valid, ic.content_mismatches)
        self.assertGreater(checked, 40)

    def test_random_bridge_compares_canonical_values(self):
        rng = random.Random(909090)
        checked = 0
        for _ in range(70):
            nargs = rng.randint(1, 8)
            specs = [_rnd_spec(rng, 2) for _ in range(nargs)]
            if any(sp[0] not in ("int", "f", "p") for sp in specs):
                # 桥接随机聚合也可以，但这里聚焦标量+边界+NaN，聚合见主例
                pass
            try:
                types = [R.build_type(sp) for sp in specs]
            except am.InvalidTypeError:
                continue
            vals = []
            for sp in specs:
                if sp[0] == "f" and rng.random() < 0.3:
                    # 带不同载荷的 NaN：桥接必须逐位保留
                    vals.append(am.FloatBits(
                        0x7FF8000000000000 | rng.randint(1, 2**52 - 1)))
                else:
                    vals.append(_rnd_val(rng, sp))
            params = [am.Param(t) for t in types]
            try:
                imgA = am.pack_call_v2(params, vals, am.LAYOUT_A)
                imgB = am.bridge_call(imgA, params, am.LAYOUT_A, am.LAYOUT_B)
                imgA2 = am.bridge_call(imgB, params, am.LAYOUT_B,
                                       am.LAYOUT_A)
            except am.ABIError:
                continue
            checked += 1
            outB = am.unpack_call_v2(params, imgB, am.LAYOUT_B)
            outA = am.unpack_call_v2(params, imgA2, am.LAYOUT_A)
            for sp, a, b, back in zip(specs, vals, outB, outA):
                if isinstance(a, am.FloatBits):
                    self.assertIsInstance(b, am.FloatBits)
                    self.assertEqual(b.pattern, a.pattern)
                    self.assertEqual(back.pattern, a.pattern)
                else:
                    self.assertEqual(b, a)
                    self.assertEqual(back, a)
        self.assertGreater(checked, 30)

    def test_register_exhaustion_both_layouts(self):
        rng = random.Random(7)
        specs = [("int", 64, 0)] * 32
        types = [R.build_type(sp) for sp in specs]
        vals = [rng.randint(0, 2**64 - 1) for _ in range(32)]
        params = [am.Param(t) for t in types]
        for d in DESCS:
            img = am.pack_call_v2(params, vals, d)
            _ri, _rf, fr, _rp, plan = R.ref_pack(
                list(zip(specs, [False] * 32)), vals, d)
            self.assertEqual(img.frame, fr)
            nreg = d.int_regs.int_count
            self.assertEqual(
                [p.kind for p in img.plan.placements[:nreg]],
                ["registers"] * nreg)
            self.assertTrue(all(
                p.kind == "stack" for p in img.plan.placements[nreg:]))
            self.assertEqual(am.unpack_call_v2(params, img, d), vals)
            am.certify_frame(params, img.plan, d).raise_if_invalid()


# ---------------------------------------------------------------------------
# 5) 主规模非退化端到端例子：32 实参 + 128 位 + 6 层嵌套 + 共享 + 变参
#    + 大返回，独立参考核对并 A->B->A 桥接。
# ---------------------------------------------------------------------------

class TestMainScaleEndToEnd(unittest.TestCase):
    def test_full_pipeline(self):
        # 6 层嵌套聚合，内含跨单元位域
        def build(depth):
            if depth == 0:
                return ("rec", [("bf", "lo", 16, 9, None, 0),
                                ("bf", "hi", 16, 9, None, 1)])
            return ("rec", [("v", ("int", 8, 0)),
                            ("q", ("f", 32)),
                            ("inner", build(depth - 1))])

        def value(depth):
            if depth == 0:
                return {"lo": 255, "hi": -256}
            return {"v": depth, "q": depth * 0.5,
                    "inner": value(depth - 1)}

        nested_spec = build(6)
        nested_t = R.build_type(nested_spec)
        nested_v = value(6)

        # 32 个实参：混合标量 / 128 位 / 嵌套聚合 / 变参 / 空聚合
        specs = []
        specs += [("int", 64, 0)] * 6
        specs += [("int", 128, 1)]
        specs += [("f", 64), ("p",), ("int", 16, 1), ("f", 32)]
        specs += [nested_spec, nested_spec]
        specs += [("rec", [])]
        specs += [("int", 32, 0)] * 18
        self.assertEqual(len(specs), 32)

        types = [R.build_type(sp) for sp in specs]
        variadic = [False] * len(specs)
        for i in (9, 10):           # i16 与 f32 作变参（默认提升）
            variadic[i] = True
        params = [am.Param(t, variadic=v)
                  for t, v in zip(types, variadic)]

        vals = []
        for i, sp in enumerate(specs):
            if sp == nested_spec:
                vals.append(nested_v)
            elif sp == ("rec", []):
                vals.append({})
            elif i == 6:
                vals.append((1 << 127) - 1)
            elif sp[0] == "f":
                vals.append(3.25 if sp[1] == 64 else 1.5)
            elif sp[0] == "p":
                vals.append(0xDEADBEEFCAFE)
            else:
                vals.append(_rnd_val(random.Random(1000 + i), sp))

        images = {}
        for d in DESCS:
            img = am.pack_call_v2(params, vals, d)
            images[d.name] = img
            # 独立参考核对全部寄存器与帧字节
            ri, rf, fr, _rp, plan = R.ref_pack(
                list(zip(specs, variadic)), vals, d)
            self.assertEqual(list(img.int_regs), ri)
            fpats = [None if v is None else
                     v.pattern if isinstance(v, am.FloatBits)
                     else struct.unpack(">Q", struct.pack(">d", v))[0]
                     for v in img.float_regs]
            self.assertEqual(fpats, rf)
            self.assertEqual(img.frame, fr)
            # 规范值往返
            self.assertEqual(am.unpack_call_v2(params, img, d), vals)
            # 证书
            am.certify_layout(nested_t, d).raise_if_invalid()
            am.certify_frame(params, img.plan, d).raise_if_invalid()
            am.verify_image(params, vals, img, d).raise_if_invalid()

        # A -> B -> A 桥接，比较规范值（不比较填充字节）
        imgA = images["A"]
        imgB = am.bridge_call(imgA, params, am.LAYOUT_A, am.LAYOUT_B)
        imgA2 = am.bridge_call(imgB, params, am.LAYOUT_B, am.LAYOUT_A)
        self.assertEqual(am.unpack_call_v2(params, imgB, am.LAYOUT_B), vals)
        self.assertEqual(am.unpack_call_v2(params, imgA2, am.LAYOUT_A), vals)


if __name__ == "__main__":
    unittest.main()
