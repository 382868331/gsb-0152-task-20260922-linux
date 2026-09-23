"""abi_marshal v2 的扩展验证（默认跳过，设 ABI_EXTENDED=1 运行）。

与快速回归分开：
    python -m unittest discover -s tests                     # 快速回归
    ABI_EXTENDED=1 python -m unittest discover -s tests -v   # 含扩展验证

内容：
1. 小参考域完整枚举（位域两字段宽度组合、标量两两组合的安放分类）；
2. 固定随机种子的随机嵌套聚合（>=6 层，记录/数组/联合/位域混合），用
   tests/reference.py 的独立实现逐位核对映像与证书，并在 A/B 两种相反
   字节序间双向桥接，比较规范值与约定保留位（NaN 载荷、有符号边界）；
3. 主规模非退化例子：32 个参数 + 深嵌套大聚合（数十 KB 帧）。

参考验证不调用被测求解器产生期望值；NaN 按原始位模式比较。
"""

import os
import random
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import abi_marshal as am
import reference as ref

A, B = am.DESCRIPTOR_A, am.DESCRIPTOR_B
C = am.AbiDescriptor(
    name="C-big-nospan", endian="big", int_reg_count=4, float_reg_count=2,
    max_reg_bytes=16, align_int={1: 1, 2: 2, 4: 4, 8: 8, 16: 16},
    bitfield_span=False, i128_even_pair=True, sret_threshold=8)

RUN = os.environ.get("ABI_EXTERED", os.environ.get("ABI_EXTENDED")) == "1"
skip_if_off = unittest.skipUnless(RUN, "set ABI_EXTENDED=1 to run")


def rec(name, fields):
    r = am.RecordType(name)
    r.define(fields)
    return r


def uni(name, members, tag=am.UINT8):
    u = am.UnionType(name)
    u.define(members, tag_type=tag)
    return u


# ---------------------------------------------------------------------------
# 规范值比较（float 按位比较，NaN 载荷参与比较）
# ---------------------------------------------------------------------------

def canon_equal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, float) and isinstance(b, float)):
            return False
        return struct.pack("<d", a) == struct.pack("<d", b)
    if isinstance(a, am.Tagged) or isinstance(b, am.Tagged):
        return (isinstance(a, am.Tagged) and isinstance(b, am.Tagged)
                and a.member == b.member and canon_equal(a.value, b.value))
    if isinstance(a, am.CellPtr) or isinstance(b, am.CellPtr):
        return isinstance(a, am.CellPtr) and isinstance(b, am.CellPtr) \
            and a.cell == b.cell
    if isinstance(a, dict):
        return set(a) == set(b) and all(canon_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(canon_equal(x, y)
                                        for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# 随机类型 / 值生成（无环合法类型）
# ---------------------------------------------------------------------------

SCALARS = [am.UINT8, am.INT8, am.UINT16, am.INT16, am.UINT32, am.INT32,
           am.UINT64, am.INT64, am.DOUBLE, am.FLOAT32]


class Factory:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.n = 0

    def uniq(self, prefix):
        self.n += 1
        return f"{prefix}{self.n}"

    def gen_type(self, depth, allow_bf=True):
        r = self.rng.random()
        if depth <= 0 or r < 0.45:
            return self.rng.choice(SCALARS)
        if r < 0.62:
            # 数组（含空数组）
            count = self.rng.randrange(0, 4)
            return am.ArrayType(self.gen_type(depth - 1, False), count)
        if r < 0.80:
            return self.gen_record(depth)
        return self.gen_union(depth)

    def gen_record(self, depth):
        nfields = self.rng.randrange(1, 4)
        fields = []
        i = 0
        while i < nfields:
            # 约 1/3 概率插入连续位域段
            if self.rng.random() < 0.3:
                run_len = self.rng.randrange(1, 3)
                for _ in range(run_len):
                    w = self.rng.randrange(1, 13)
                    sgn = self.rng.random() < 0.5
                    fields.append((self.uniq("bf"), am.BitField(w, sgn)))
                    i += 1
                    if i >= nfields:
                        break
            else:
                fields.append((self.uniq("f"),
                               self.gen_type(depth - 1, False)))
                i += 1
        return rec(self.uniq("R"), fields)

    def gen_union(self, depth):
        members = []
        tags = self.rng.sample(range(256), self.rng.randrange(2, 4))
        for tag in tags:
            members.append((self.uniq("m"),
                            self.gen_type(depth - 1, False), tag))
        return uni(self.uniq("U"), members)

    def gen_value(self, t):
        if isinstance(t, (am.IntType, am.IntNType)):
            lo, hi = t.range
            choices = [lo, hi, 0, 1]
            inner_lo = max(lo, -(2 ** 40))
            inner_hi = min(hi, 2 ** 40)
            if inner_hi > inner_lo:
                choices.append(self.rng.randrange(inner_lo, inner_hi + 1))
            return self.rng.choice(choices)
        if isinstance(t, am.FloatType):
            return self._gen_float(64)
        if isinstance(t, am.Float32Type):
            # 合法 f32 值即能被 binary32 精确表示的 float（生成后舍入）
            v = self._gen_float(32)
            return struct.unpack("<f", struct.pack("<f", v))[0]
        if isinstance(t, am.PtrType):
            return self.rng.randrange(0, 2 ** 48)
        if isinstance(t, am.ArrayType):
            return [self.gen_value(t.elem) for _ in range(t.count)]
        if isinstance(t, am.RecordType):
            out = {}
            for name, ty, _pin in t.fields:
                if isinstance(ty, am.BitField):
                    lo = -(1 << (ty.width - 1)) if ty.signed else 0
                    hi = (1 << (ty.width - 1)) - 1 if ty.signed \
                        else (1 << ty.width) - 1
                    out[name] = self.rng.choice([lo, hi, 0, self.rng.randrange(
                        max(lo, 0), hi + 1)])
                else:
                    out[name] = self.gen_value(ty)
            return out
        if isinstance(t, am.UnionType):
            name, ty, _tag = self.rng.choice(t.members)
            return am.Tagged(name, self.gen_value(ty))
        raise AssertionError(t)

    def _gen_float(self, bits):
        kind = self.rng.random()
        fmt = "<d" if bits == 64 else "<f"
        if kind < 0.25:
            # 任意 NaN（含非标准载荷）
            exp = (1 << (11 if bits == 64 else 8)) - 1
            payload = self.rng.randrange(1, 1 << (52 if bits == 64 else 23))
            raw = (exp << (52 if bits == 64 else 23)) | payload
            return struct.unpack(fmt,
                                 raw.to_bytes(bits // 8, "little"))[0]
        if kind < 0.35:
            return self.rng.choice([float("inf"), float("-inf"), 0.0, -0.0])
        return self.rng.uniform(-1e12, 1e12)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class TestExhaustiveTinyDomain(unittest.TestCase):
    """完整枚举小规模域，逐位给确定答案。"""

    def test_bitfield_two_field_combinations(self):
        # 两字段宽度 (1..8)x(1..8) 共 64 组合 × 两种字节序 × 边界值，
        # 全部用独立参考逐字节核对
        for desc in (A, B):
            for w1 in range(1, 9):
                for w2 in range(1, 9):
                    r = rec(f"X{desc.name}{w1}_{w2}",
                            [("a", am.BitField(w1)),
                             ("b", am.BitField(w2, True))])
                    rr = ref.ref_layout(desc, r)
                    for va in (0, (1 << w1) - 1):
                        for vb in (-(1 << (w2 - 1)),
                                   (1 << (w2 - 1)) - 1):
                            val = {"a": va, "b": vb}
                            got = am.encode(r, val, abi=desc)
                            exp = ref.ref_encode(desc, r, rr, val)
                            self.assertEqual(got, exp,
                                             f"{desc.name} {w1},{w2} {val}")
                            self.assertEqual(am.decode(r, got, abi=desc), val)

    def test_scalar_pair_placement_enumeration(self):
        # 所有标量两两组合在 A 与 B 下的安放，与独立规划逐一核对
        scalars = SCALARS + [am.UINT128, am.INT128]
        fac = Factory(123)
        for desc in (A, B):
            for s1 in scalars:
                for s2 in scalars:
                    sig = am.Signature(args=(s1, s2))
                    plan = am.plan_call(desc, sig)
                    rp = ref.ref_plan(desc, sig)
                    for i, (pl, rr) in enumerate(
                            zip(plan.placements, rp["placements"])):
                        self.assertEqual(pl.kind, rr["kind"],
                                         f"{desc.name} {s1.name},{s2.name}[{i}]")
                        self.assertEqual(pl.int_regs, rr["iregs"])
                        self.assertEqual(pl.float_regs, rr["fregs"])
                    v = [fac.gen_value(s1), fac.gen_value(s2)]
                    img = am.encode_call(plan, v)
                    issues, _ = ref.verify_image(desc, sig, plan, img,
                                                 v, [], {}, None)
                    self.assertEqual(issues, [], "; ".join(issues))


class TestRandomNested(unittest.TestCase):
    N_CASES = 120

    def _run_one(self, seed, desc):
        fac = Factory(seed)
        t = fac.gen_type(6)
        # 控制单参数总尺寸（过大的顶层数组收缩为浅类型）
        if am.layout(t, abi=desc).size > 4000:
            t = fac.gen_type(3)
        val = fac.gen_value(t)

        n_args = fac.rng.randrange(1, 8)
        arg_ts = [fac.gen_type(4) for _ in range(n_args)]
        arg_vs = [fac.gen_value(x) for x in arg_ts]
        n_var = fac.rng.randrange(0, 4)
        var_ts = []
        for _ in range(n_var):
            var_ts.append(fac.rng.choice(
                [am.UINT8, am.INT16, am.FLOAT32, am.INT64, am.DOUBLE]))
        var_vs = [fac.gen_value(x) for x in var_ts]

        # 三者同现：sret + cell + save area
        ret_t = fac.gen_type(4)
        ret_v = fac.gen_value(ret_t)
        cell_t = am.ArrayType(am.UINT8, fac.rng.randrange(0, 24))
        cell_v = [fac.rng.randrange(256) for _ in range(cell_t.count)]
        arg_ts.append(am.PTR)
        arg_vs.append(am.CellPtr("buf"))

        sig = am.Signature(args=tuple(arg_ts), varargs=tuple(var_ts),
                           ret=ret_t, cells=(("buf", cell_t),))
        plan = am.plan_call(desc, sig)
        img = am.encode_call(plan, arg_vs, var_vs, {"buf": cell_v}, ret_v)
        issues, _ = ref.verify_image(desc, sig, plan, img, arg_vs, var_vs,
                                     {"buf": cell_v}, ret_v)
        self.assertEqual(issues, [], f"seed={seed} {desc.name}: " +
                         "; ".join(issues))
        back = am.decode_call(plan, img)
        self.assertTrue(canon_equal(back.args, arg_vs),
                        f"seed={seed} args canonical mismatch")
        self.assertTrue(canon_equal(back.varargs, var_vs),
                        f"seed={seed} varargs canonical mismatch")
        self.assertTrue(canon_equal(back.ret, ret_v),
                        f"seed={seed} ret canonical mismatch")
        return sig, plan, img, arg_vs, var_vs, {"buf": cell_v}, ret_v

    @skip_if_off
    def test_random_both_endians_and_bridge(self):
        for seed in range(self.N_CASES):
            sigA, pA, imgA, av, vv, cv, rv = self._run_one(seed, A)
            sigB, pB, imgB, av2, vv2, cv2, rv2 = self._run_one(seed, B)
            # A -> B
            bA = am.bridge_call(pA, imgA, B)
            issues, _ = ref.verify_image(B, sigA, bA.plan, bA, av, vv, cv, rv)
            self.assertEqual(issues, [], f"seed={seed} A->B: " +
                             "; ".join(issues))
            dB = am.decode_call(bA.plan, bA)
            self.assertTrue(canon_equal(dB.args, av))
            self.assertTrue(canon_equal(dB.varargs, vv))
            self.assertTrue(canon_equal(dB.ret, rv))
            self.assertTrue(canon_equal(dB.cells, cv))
            # B -> A
            bB = am.bridge_call(pB, imgB, A)
            dA = am.decode_call(bB.plan, bB)
            self.assertTrue(canon_equal(dA.args, av2))
            self.assertTrue(canon_equal(dA.varargs, vv2))
            self.assertTrue(canon_equal(dA.ret, rv2))
            # 证书双端自检
            self.assertEqual(am.check_certificate(pA.certificate), [])
            self.assertEqual(am.check_certificate(bA.plan.certificate), [])
            self.assertEqual(am.check_certificate(bB.plan.certificate), [])


class TestMainScale(unittest.TestCase):
    @skip_if_off
    def test_large_non_toy_call(self):
        # 非退化大例子：5 层结构，每层 3 路数组，底层联合（含位域记录），
        # 总叶子数 3^5=243；加上 32 个混合参数与多个可变参数。
        fac = Factory(20260924)
        bits = rec("Bits", [("lo", am.BitField(5)), ("mid", am.BitField(7)),
                            ("hi", am.BitField(4, True)),
                            ("x", am.UINT16)])
        u = uni("LeafU", [("b", bits, 0), ("v", am.INT64, 1),
                          ("f", am.FLOAT32, 2)])
        t = u
        for d in range(6):
            t = rec(f"Node{d}", [("tag", am.UINT8),
                                 ("kids", am.ArrayType(t, 3)),
                                 ("w", am.INT32)])
        val = fac.gen_value(t)
        self.assertGreater(am.layout(t, abi=A).size, 14000)

        arg_ts = [t]
        for i in range(30):
            arg_ts.append([am.INT64, am.DOUBLE, am.UINT32,
                           am.ArrayType(am.UINT16, 8)][i % 4])
        # 1 个大聚合 + 30 个混合参数 + 1 个 cell 指针 = 32（上限）
        arg_vs = [val] + [fac.gen_value(x) for x in arg_ts[1:]]
        var_ts = (am.UINT8, am.INT16, am.FLOAT32, am.DOUBLE)
        var_vs = [fac.gen_value(x) for x in var_ts]
        cell_t = am.ArrayType(am.UINT8, 64)
        cell_v = [fac.rng.randrange(256) for _ in range(64)]
        ret_t = am.ArrayType(am.UINT8, 20)  # A:20>16 sret；B 也 sret
        ret_v = [fac.rng.randrange(256) for _ in range(20)]
        arg_ts.append(am.PTR)
        arg_vs.append(am.CellPtr("buf"))

        sig = am.Signature(args=tuple(arg_ts), varargs=var_ts, ret=ret_t,
                           cells=(("buf", cell_t),))
        for desc in (A, B):
            plan = am.plan_call(desc, sig)
            img = am.encode_call(plan, arg_vs, var_vs, {"buf": cell_v},
                                 ret_v)
            issues, _ = ref.verify_image(desc, sig, plan, img, arg_vs, var_vs,
                                         {"buf": cell_v}, ret_v)
            self.assertEqual(issues, [], "; ".join(issues))
            back = am.decode_call(plan, img)
            self.assertTrue(canon_equal(back.args, arg_vs))
            self.assertTrue(canon_equal(back.ret, ret_v))
            # 桥到另一端再回来
            other = B if desc is A else A
            bimg = am.bridge_call(plan, img, other)
            again = am.bridge_call(bimg.plan, bimg, desc)
            self.assertTrue(canon_equal(
                am.decode_call(again.plan, again).args, arg_vs))
            self.assertGreaterEqual(plan.frame_size, 8000)


if __name__ == "__main__":
    unittest.main()
