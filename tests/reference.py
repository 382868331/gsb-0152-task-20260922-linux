"""独立参考实现（测试专用）。

本文件是 abi_marshal v2 的**平行独立实现**：只消费类型对象上的声明字段
（size/signed/fields/members/count/elem 等）与描述符字段，不调用
abi_marshal 的 layout/encode/classify/plan_call 等任何函数来取得期望值。
用途：

* ref_layout     —— 独立计算 size/align/字段偏移/位域物理位；
* ref_encode     —— 独立把规范值编码为字节；
* ref_plan       —— 独立计算寄存器/帧/sret/单元/保存区安放；
* verify_image   —— 独立重放：核对安放结构、映像逐位值与位证书归属。

检查器不读取 abi_marshal.check_certificate 的结论，也不信任求解器给出的
任何布尔标志；每一位的期望值都在这里重新推导。
"""

import struct

import abi_marshal as am

FRAME_BASE = am.FRAME_BASE
I, F, M = am.INTEGER, am.FLOAT, am.MEMORY


class R:
    __slots__ = ("size", "align", "kind", "fields", "members",
                 "tag_size", "tag_signed", "storage_off", "storage_size",
                 "elem", "count", "bits")

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))


def align_up(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# 独立布局
# ---------------------------------------------------------------------------

def scalar(t, desc):
    if isinstance(t, am.IntType):
        return t.size, desc.align_int[t.size]
    if isinstance(t, am.IntNType):
        return t.size, desc.align_int[t.size]
    if isinstance(t, am.FloatType):
        return 8, desc.align_float[8]
    if isinstance(t, am.Float32Type):
        return 4, desc.align_float[4]
    if isinstance(t, am.PtrType):
        return 8, desc.align_ptr
    raise AssertionError(f"not scalar: {t!r}")


def ref_layout(desc, t, visiting=frozenset()):
    if isinstance(t, (am.IntType, am.IntNType, am.FloatType,
                      am.Float32Type, am.PtrType)):
        size, al = scalar(t, desc)
        return R(kind="scalar", size=size, align=al, bits=size * 8)

    if isinstance(t, am.BitField):
        raise AssertionError("bare bitfield")

    if isinstance(t, am.ArrayType):
        key = ("a", id(t))
        if key in visiting:
            raise am.RecursiveTypeError("recursion")
        if t.count == 0:
            return R(kind="array", size=0, align=1, elem=t.elem, count=0,
                     bits=0)
        e = ref_layout(desc, t.elem, visiting | {key})
        size = e.size * t.count
        return R(kind="array", size=size,
                 align=min(e.align, desc.max_agg_align),
                 elem=t.elem, count=t.count, bits=size * 8)

    if isinstance(t, am.RecordType):
        key = ("r", id(t))
        if key in visiting:
            raise am.RecursiveTypeError("recursion")
        nv = visiting | {key}
        off = 0
        al = 1
        fields = []
        fdecl = t.fields
        i = 0
        while i < len(fdecl):
            name, ty, pin = fdecl[i]
            if not isinstance(ty, am.BitField):
                e = ref_layout(desc, ty, nv)
                a2 = min(e.align, desc.max_agg_align)
                off = align_up(off, a2)
                al = max(al, a2)
                fields.append((name, off, e))
                off += e.size
                i += 1
                continue
            run = []
            while i < len(fdecl) and isinstance(fdecl[i][1], am.BitField):
                run.append(fdecl[i])
                i += 1
            cursor = 0
            placed = []
            for rn, rb, rpin in run:
                g = cursor if rpin is None else rpin
                for pn, pb, pg, pw in placed:
                    if g < pg + pw and pg < g + rb.width:
                        raise am.BitFieldError("overlap", rn)
                if not desc.bitfield_span and g // 8 != (g + rb.width - 1) // 8:
                    raise am.BitFieldError("crossing forbidden", rn)
                placed.append((rn, rb, g, rb.width))
                cursor = max(cursor, g + rb.width)
            run_bytes = (cursor + 7) // 8
            er = R(kind="bitrun", size=run_bytes, align=1, bits=run_bytes * 8)
            for rn, rb, g, _w in placed:
                fields.append((rn, off, er, g, rb))
            off += run_bytes
            al = max(al, 1)
        size = align_up(off, al)
        return R(kind="record", size=size, align=al, fields=fields,
                 bits=size * 8)

    if isinstance(t, am.UnionType):
        key = ("u", id(t))
        if key in visiting:
            raise am.RecursiveTypeError("recursion")
        nv = visiting | {key}
        tag_size, tag_al = scalar(t.tag_type, desc)
        members = []
        storage = 0
        sal = 1
        for mn, mt, tag in t.members:
            e = ref_layout(desc, mt, nv)
            members.append((mn, tag, e))
            storage = max(storage, e.size)
            sal = max(sal, e.align)
        sal = min(sal, desc.max_agg_align)
        storage_off = align_up(tag_size, sal)
        size = align_up(storage_off + storage, sal)
        return R(kind="union", size=size, align=sal, members=members,
                 tag_size=tag_size, tag_signed=t.tag_type.signed,
                 storage_off=storage_off, storage_size=storage,
                 bits=size * 8)

    raise AssertionError(t)


# ---------------------------------------------------------------------------
# 独立编码
# ---------------------------------------------------------------------------

def put_bit(buf, byte, bit, v):
    if v:
        buf[byte] |= 1 << bit
    else:
        buf[byte] &= ~(1 << bit)


def ref_encode(desc, t, r, value, path="v", ptr_resolve=None):
    en = desc.endian
    if r.kind == "scalar":
        if isinstance(t, am.PtrType):
            if isinstance(value, am.CellPtr):
                value = ptr_resolve(value.cell)
            if isinstance(value, bool) or not isinstance(value, int) \
                    or not 0 <= value <= 2 ** 64 - 1:
                raise am.InvalidValueError("bad ptr", path)
            return value.to_bytes(8, en)
        if isinstance(t, (am.IntType, am.IntNType)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise am.InvalidValueError("bad int", path)
            bits = r.size * 8
            lo = -(1 << (bits - 1)) if t.signed else 0
            hi = (1 << (bits - 1)) - 1 if t.signed else (1 << bits) - 1
            if not lo <= value <= hi:
                raise am.IntegerOverflowError("range", path)
            return value.to_bytes(r.size, en, signed=t.signed)
        if isinstance(t, am.FloatType):
            if not isinstance(value, float):
                raise am.InvalidValueError("bad f64", path)
            return struct.pack(desc.endian_char + "d", value)
        if isinstance(t, am.Float32Type):
            if not isinstance(value, float):
                raise am.InvalidValueError("bad f32", path)
            return struct.pack(desc.endian_char + "f", value)
    if r.kind == "array":
        if not isinstance(value, (list, tuple)) or len(value) != t.count:
            raise am.InvalidValueError("bad array", path)
        e = ref_layout(desc, t.elem)
        return b"".join(ref_encode(desc, t.elem, e, v, f"{path}[{i}]",
                                   ptr_resolve)
                        for i, v in enumerate(value))
    if r.kind == "record":
        if not isinstance(value, dict):
            raise am.InvalidValueError("bad record", path)
        buf = bytearray(r.size)
        bit_runs = {}
        for entry in r.fields:
            name = entry[0]
            if name not in value:
                raise am.InvalidValueError(f"missing {name}", path)
        if set(value) != {e[0] for e in r.fields}:
            raise am.InvalidValueError("field set", path)
        for entry in r.fields:
            name, off, er = entry[0], entry[1], entry[2]
            if er.kind == "bitrun":
                _, _, _, g, bf = entry
                w = bf.width
                v = value[name]
                if isinstance(v, bool) or not isinstance(v, int):
                    raise am.InvalidValueError("bad bitfield", f"{path}.{name}")
                lo = -(1 << (w - 1)) if bf.signed else 0
                hi = (1 << (w - 1)) - 1 if bf.signed else (1 << w) - 1
                if not lo <= v <= hi:
                    raise am.IntegerOverflowError("bf range", f"{path}.{name}")
                uv = v & ((1 << w) - 1)
                for j in range(w):
                    lin = g + j
                    if en == "little":
                        byte, bit = lin // 8, lin % 8
                    else:
                        byte, bit = r_units(r, entry, lin)
                    put_bit(buf, off + byte, bit, (uv >> j) & 1)
            else:
                chunk = ref_encode(desc, dict_field_type(t, name), er,
                                   value[name], f"{path}.{name}", ptr_resolve)
                buf[off:off + len(chunk)] = chunk
        return bytes(buf)
    if r.kind == "union":
        if not isinstance(value, am.Tagged):
            raise am.InvalidValueError("bad tagged", path)
        buf = bytearray(r.size)
        m = next((x for x in r.members if x[0] == value.member), None)
        if m is None:
            raise am.TagError("no member", path)
        mn, tag, me = m
        buf[0:r.tag_size] = tag.to_bytes(r.tag_size, en, signed=r.tag_signed)
        chunk = ref_encode(desc, dict_member_type(t, mn), me, value.value,
                           f"{path}.<{mn}>", ptr_resolve)
        buf[r.storage_off:r.storage_off + len(chunk)] = chunk
        return bytes(buf)
    raise AssertionError(r.kind)


def r_units(rec_r, entry, lin):
    # 大端位域物理位（run 字节数从 entry 所属 run 推）：
    run_bytes = entry[2].size
    return run_bytes - 1 - lin // 8, 7 - lin % 8


def dict_field_type(record, name):
    for n, ty, _p in record.fields:
        if n == name:
            return ty
    raise KeyError(name)


def dict_member_type(union, name):
    for n, ty, _tag in union.members:
        if n == name:
            return ty
    raise KeyError(name)


# ---------------------------------------------------------------------------
# 独立分类 / 规划
# ---------------------------------------------------------------------------

def ref_spans(desc, t, r, base, out):
    if r.kind == "scalar":
        out.append((base, r.size,
                    F if isinstance(t, (am.FloatType, am.Float32Type)) else I))
    elif r.kind == "array":
        e = ref_layout(desc, t.elem)
        step = r.size // t.count if t.count else 0
        for i in range(t.count):
            ref_spans(desc, t.elem, e, base + i * step, out)
    elif r.kind == "record":
        for entry in r.fields:
            name, off, er = entry[0], entry[1], entry[2]
            if er.kind == "bitrun":
                _, _, _, g, bf = entry
                lo_bit, hi_bit = g, g + bf.width
                out.append((base + off + lo_bit // 8,
                            (hi_bit + 7) // 8 - lo_bit // 8, I))
            else:
                ref_spans(desc, dict_field_type(t, name), er, base + off, out)
    elif r.kind == "union":
        out.append((base, r.tag_size, I))
        for mn, tag, me in r.members:
            ref_spans(desc, dict_member_type(t, mn), me,
                      base + r.storage_off, out)


def ref_classify(desc, t, r):
    if r.size == 0:
        return ()
    if r.size > desc.max_reg_bytes:
        return (M,)
    spans = []
    ref_spans(desc, t, r, 0, spans)
    out = []
    for cs in range(0, r.size, 8):
        c = F
        for off, size, k in spans:
            if k == I and off < cs + 8 and off + size > cs:
                c = I
                break
        out.append(c)
    return tuple(out)


def ref_promote(desc, t):
    if isinstance(t, am.IntType) and t.size < 4:
        return am.IntType(4, t.signed)
    if isinstance(t, am.Float32Type):
        return am.FloatType()
    if isinstance(t, am.IntNType):
        raise am.PromotionError("i128 vararg", "vararg")
    return t


def ref_alloc(desc, t, r, frame, ni, nf, even_pair=False):
    cls = ref_classify(desc, t, r)
    if not cls:
        return ("none", (), (), (), None, 0, frame, ni, nf)
    if cls == (M,):
        off = align_up(frame, desc.stack_align)
        return ("stack", cls, (), (), off, r.size, off + r.size, ni, nf)
    need_i, need_f = cls.count(I), cls.count(F)
    si = ni
    if even_pair and isinstance(t, am.IntNType) and ni % 2 == 1:
        si = ni + 1
    if si + need_i <= desc.int_reg_count and nf + need_f <= desc.float_reg_count:
        return ("registers", cls,
                tuple(range(si, si + need_i)), tuple(range(nf, nf + need_f)),
                None, r.size, frame, si + need_i, nf + need_f)
    off = align_up(frame, desc.stack_align)
    return ("stack", cls, (), (), off, r.size, off + r.size, ni, nf)


def ref_plan(desc, sig):
    arg_r = [ref_layout(desc, t) for t in sig.args]
    vtypes = [ref_promote(desc, t) for t in sig.varargs]
    vr = [ref_layout(desc, t) for t in vtypes]
    ret_r = ref_layout(desc, sig.ret) if sig.ret else None

    if sig.ret is None:
        ret_kind = "none"
    elif sig.ret_cell is not None:
        ret_kind = "cell"
    else:
        cls = ref_classify(desc, sig.ret, ret_r)
        ret_kind = "sret" if (ret_r.size > desc.sret_threshold
                              or cls == (M,)) else "reg"

    frame, ni, nf = 0, 0, 0
    hidden = None
    if ret_kind == "sret":
        hidden, frame, ni, nf = _take(desc, am.PTR, ref_layout(desc, am.PTR),
                                      frame, ni, nf, even=False)

    placements = []
    for i, (t, r) in enumerate(zip(sig.args, arg_r)):
        pl, frame, ni, nf = _take(desc, t, r, frame, ni, nf,
                                  even=desc.i128_even_pair)
        placements.append(pl)

    vpls, si_slots, sf_slots = [], [], []
    for i, (t, r) in enumerate(zip(vtypes, vr)):
        pl, frame, ni, nf = _take(desc, t, r, frame, ni, nf,
                                  even=desc.i128_even_pair)
        vpls.append(pl)
        if pl["kind"] == "registers":
            si_slots.extend(pl["iregs"])
            sf_slots.extend(pl["fregs"])

    cells = {}
    cursor = align_up(frame, desc.stack_align)
    for cn, ct in sig.cells:
        cr = ref_layout(desc, ct)
        a = min(max(cr.align, desc.stack_align), desc.max_agg_align)
        cursor = align_up(cursor, a)
        cells[cn] = (cursor, cr)
        cursor += cr.size
    if ret_kind == "sret":
        a = min(max(ret_r.align, desc.stack_align), desc.max_agg_align)
        cursor = align_up(cursor, a)
        cells["$sret"] = (cursor, ret_r)
        cursor += ret_r.size

    save = None
    if si_slots or sf_slots:
        ioff = align_up(cursor, desc.stack_align)
        cursor = ioff + 8 * len(si_slots)
        foff = align_up(cursor, desc.stack_align)
        cursor = foff + 8 * len(sf_slots)
        save = (ioff, tuple(si_slots), foff, tuple(sf_slots))
    total = align_up(cursor, desc.stack_align)

    return {
        "arg_r": arg_r, "vtypes": vtypes, "vr": vr, "ret_r": ret_r,
        "ret_kind": ret_kind, "hidden": hidden,
        "placements": placements, "vpls": vpls,
        "cells": cells, "save": save, "frame_size": total,
    }


def _take(desc, t, r, frame, ni, nf, even):
    kind, cls, iregs, fregs, off, size, frame2, ni2, nf2 = ref_alloc(
        desc, t, r, frame, ni, nf, even_pair=even)
    return {"kind": kind, "cls": cls, "iregs": iregs, "fregs": fregs,
            "off": off, "size": size}, frame2, ni2, nf2


# ---------------------------------------------------------------------------
# 独立映像重放与证书核对
# ---------------------------------------------------------------------------

def slot_pack(desc, chunk):
    """与 abi_marshal._slot_raw 同一定义，但在此独立书写：
    槽是块的 8 字节内存别名，块数据在槽低 n 字节，槽原始位按物理 little
    位序编号，与描述符字节序无关。"""
    buf = bytearray(8)
    buf[:len(chunk)] = chunk
    return int.from_bytes(buf, "little")


def slot_unpack(raw, desc, n):
    buf = raw.to_bytes(8, "little")
    return buf if n == 8 else buf[:n]


def verify_image(desc, sig, plan, image, arg_vals, vararg_vals,
                 cell_vals, ret_val):
    """完整独立核对。返回 (问题列表, ref_plan)。问题为空即全部通过。"""
    issues = []
    rp = ref_plan(desc, sig)

    def expect(cond, msg):
        if not cond:
            issues.append(msg)

    # ---- 对齐约束（两种字节序下都必须成立）----
    expect(rp["frame_size"] % desc.stack_align == 0,
           f"frame_size {rp['frame_size']} not aligned to "
           f"{desc.stack_align}")
    for i, (pl, ref) in enumerate(zip(plan.placements, rp["placements"])):
        if ref["kind"] == "stack":
            rr = rp["arg_r"][i]
            a = min(rr.align, desc.max_agg_align)
            expect(ref["off"] % a == 0,
                   f"arg[{i}] stack offset {ref['off']} not aligned to {a}")
        if pl.kind == "registers":
            # i128 奇偶配对：槽连续，若描述符要求则起始为偶数
            if desc.i128_even_pair and isinstance(sig.args[i], am.IntNType):
                expect(pl.int_regs[0] % 2 == 0,
                       f"arg[{i}] i128 starts on odd slot")
    for i, (pl, ref) in enumerate(zip(plan.vararg_placements, rp["vpls"])):
        if ref["kind"] == "stack":
            a = min(rp["vr"][i].align, desc.max_agg_align)
            expect(ref["off"] % a == 0,
                   f"vararg[{i}] stack offset not aligned to {a}")
    for cn, (off, cr) in rp["cells"].items():
        a = min(max(cr.align, desc.stack_align), desc.max_agg_align)
        expect(off % a == 0, f"cell {cn} offset {off} not aligned to {a}")
    if rp["save"] is not None:
        ioff, _isl, foff, _fsl = rp["save"]
        expect(ioff % desc.stack_align == 0, "save int area not aligned")
        expect(foff % desc.stack_align == 0, "save float area not aligned")

    # ---- 结构：偏移 / 槽号 / 帧尺寸 / 单元 / 保存区 ----
    expect(plan.frame_size == rp["frame_size"],
           f"frame_size {plan.frame_size} != ref {rp['frame_size']}")
    for i, (pl, ref) in enumerate(zip(plan.placements, rp["placements"])):
        expect(pl.kind == ref["kind"], f"arg[{i}] kind {pl.kind}!={ref['kind']}")
        expect(pl.int_regs == ref["iregs"], f"arg[{i}] iregs")
        expect(pl.float_regs == ref["fregs"], f"arg[{i}] fregs")
        expect(pl.frame_offset == ref["off"], f"arg[{i}] off")
        expect(pl.size == ref["size"], f"arg[{i}] size")
    for i, (pl, ref) in enumerate(zip(plan.vararg_placements, rp["vpls"])):
        expect(pl.kind == ref["kind"], f"vararg[{i}] kind")
        expect(pl.int_regs == ref["iregs"], f"vararg[{i}] iregs")
        expect(pl.float_regs == ref["fregs"], f"vararg[{i}] fregs")
        expect(pl.frame_offset == ref["off"], f"vararg[{i}] off")
    expect(plan.ret.kind == rp["ret_kind"],
           f"ret kind {plan.ret.kind} != {rp['ret_kind']}")
    for cn, (off, cr) in rp["cells"].items():
        ci = next((c for c in plan.cells if c.name == cn), None)
        expect(ci is not None, f"cell {cn} missing")
        if ci:
            expect(ci.offset == off, f"cell {cn} off {ci.offset}!={off}")
            expect(ci.layout.size == cr.size, f"cell {cn} size")
    if rp["save"] is None:
        expect(plan.save is None, "save should be None")
    else:
        ioff, isl, foff, fsl = rp["save"]
        expect(plan.save.int_offset == ioff, "save int off")
        expect(plan.save.float_offset == foff, "save float off")
        expect(plan.save.int_slots == isl, "save int slots")
        expect(plan.save.float_slots == fsl, "save float slots")

    # ---- 独立重放映像 ----
    ptr_map = {cn: off for cn, (off, _cr) in rp["cells"].items()}

    def resolve(name):
        return FRAME_BASE + ptr_map[name]

    exp_ireg = [None] * desc.int_reg_count
    exp_freg = [None] * desc.float_reg_count
    exp_frame = bytearray(rp["frame_size"])

    def place(refpl, tt, rr, val, path):
        data = ref_encode(desc, tt, rr, val, path, resolve)
        if refpl["kind"] == "none":
            return
        if refpl["kind"] == "stack":
            exp_frame[refpl["off"]:refpl["off"] + len(data)] = data
            return
        ii = iter(refpl["iregs"])
        fi = iter(refpl["fregs"])
        n_full, tail = len(data) // 8, len(data) % 8
        for k, c in enumerate(refpl["cls"]):
            if k < n_full:
                chunk = data[8 * k:8 * k + 8]
            elif k == n_full and tail:
                chunk = data[8 * k:8 * k + tail]
            else:
                chunk = b"\x00" * 8
            raw = slot_pack(desc, chunk)
            if c == I:
                exp_ireg[next(ii)] = raw
            else:
                exp_freg[next(fi)] = raw

    h = rp["hidden"]
    if h is not None:
        addr = FRAME_BASE + ptr_map["$sret"]
        db = addr.to_bytes(8, desc.endian)
        if h["kind"] == "stack":
            exp_frame[h["off"]:h["off"] + 8] = db
        else:
            exp_ireg[h["iregs"][0]] = slot_pack(desc, db)

    for i, ((tt, rr), refpl, v) in enumerate(
            zip(zip(sig.args, rp["arg_r"]), rp["placements"], arg_vals)):
        place(refpl, tt, rr, v, f"arg[{i}]")
    for i, (t0, refpl, v) in enumerate(
            zip(sig.varargs, rp["vpls"], vararg_vals)):
        tt = rp["vtypes"][i]
        rr = rp["vr"][i]
        if isinstance(t0, am.IntType) and t0.size < 4:
            pass  # 值本身在 i32 域内
        elif isinstance(t0, am.Float32Type):
            v = struct.unpack(desc.endian_char + "f",
                              struct.pack(desc.endian_char + "f", v))[0]
        place(refpl, tt, rr, v, f"vararg[{i}]")

    for cn, ct in sig.cells:
        off, cr = rp["cells"][cn]
        val = ret_val if sig.ret_cell == cn else cell_vals[cn]
        exp_frame[off:off + cr.size] = ref_encode(
            desc, ct, cr, val, f"cell:{cn}", resolve)
    if rp["ret_kind"] == "sret":
        off, cr = rp["cells"]["$sret"]
        exp_frame[off:off + cr.size] = ref_encode(
            desc, sig.ret, cr, ret_val, "ret", resolve)

    # 保存区：寄存器原始位的按位拷贝（与字节序无关）
    if rp["save"] is not None:
        ioff, isl, foff, fsl = rp["save"]
        for j, s in enumerate(isl):
            exp_frame[ioff + 8 * j:ioff + 8 * j + 8] = \
                exp_ireg[s].to_bytes(8, "little")
        for j, s in enumerate(fsl):
            exp_frame[foff + 8 * j:foff + 8 * j + 8] = \
                exp_freg[s].to_bytes(8, "little")

    expect(bytes(exp_frame) == bytes(image.frame),
           f"frame bytes differ: "
           f"{bytes(exp_frame).hex()} != {bytes(image.frame).hex()}")
    for k in range(desc.int_reg_count):
        expect(exp_ireg[k] == image.int_regs[k],
               f"ireg:{k} {exp_ireg[k]} != {image.int_regs[k]}")
    for k in range(desc.float_reg_count):
        expect(exp_freg[k] == image.float_regs[k],
               f"freg:{k} {exp_freg[k]} != {image.float_regs[k]}")

    # 返回值（reg）
    if rp["ret_kind"] == "reg" and ret_val is not None:
        rr = rp["ret_r"]
        data = ref_encode(desc, sig.ret, rr, ret_val, "ret", resolve)
        ei, ef = [], []
        for k, c in enumerate(ref_classify(desc, sig.ret, rr)):
            chunk = data[8 * k:8 * k + 8] or b"\x00" * 8
            (ei if c == I else ef).append(slot_pack(desc, chunk))
        expect(tuple(ei) == image.ret_int, "ret_int differs")
        expect(tuple(ef) == image.ret_float, "ret_float differs")

    # ---- 独立证书核对：每一位恰好覆盖一次，pad 位在映像中为 0 ----
    cert = plan.certificate
    expect(cert.frame_bits == rp["frame_size"] * 8, "cert frame_bits")
    expect(cert.int_regs == desc.int_reg_count, "cert int_regs")
    expect(cert.float_regs == desc.float_reg_count, "cert float_regs")
    totals = {"frame": cert.frame_bits}
    for k in range(desc.int_reg_count):
        totals[f"ireg:{k}"] = 64
    for k in range(desc.float_reg_count):
        totals[f"freg:{k}"] = 64
    covered = {r: 0 for r in totals}
    for e in cert.entries:
        total = totals.get(e.region)
        if total is None:
            issues.append(f"cert bad region {e.region}")
            continue
        if not (0 <= e.start and e.length > 0 and e.start + e.length <= total):
            issues.append(f"cert oob {e.region}[{e.start},{e.start + e.length})")
            continue
        bits = ((1 << e.length) - 1) << e.start
        if covered[e.region] & bits:
            issues.append(f"cert overlap {e.region}[{e.start}..) {e.kind}")
        covered[e.region] |= bits
        # 与实际映像核对
        if e.region == "frame":
            av = _read_bits(bytes(image.frame), e.start, e.length)
        elif e.region.startswith("ireg:"):
            raw = image.int_regs[int(e.region.split(":")[1])]
            av = _reg_bits(raw, e.start, e.length)
        else:
            raw = image.float_regs[int(e.region.split(":")[1])]
            av = _reg_bits(raw, e.start, e.length)
        if e.kind == "pad":
            expect(av == 0, f"pad nonzero at {e.region}[{e.start}]")
        elif e.kind == "save-copy":
            src = e.src_region
            raw = image.int_regs[int(src.split(":")[1])] \
                if src.startswith("ireg") else \
                image.float_regs[int(src.split(":")[1])]
            # 保存区按槽原始位的逻辑（小端）位序逐位落盘
            expect(raw is not None and av == raw,
                   f"save-copy mismatch {e.region} -> {src}")
        # value/tag/union/cell/ret 的位值已经由上面的“独立重放映像”
        # 逐字节保证（image == exp_image），此处只需确认区间非空。
    for r, total in totals.items():
        want = (1 << total) - 1 if total else 0
        expect(covered[r] == want,
               f"cert incomplete {r}: "
               f"{(want & ~covered[r]).bit_count()} uncovered")

    return issues, rp


def _read_bits(data, start, width):
    out = 0
    for j in range(width):
        b = start + j
        out |= ((data[b // 8] >> (b % 8)) & 1) << j
    return out


def _reg_bits(raw, start, width):
    if raw is None:
        raw = 0  # 未使用槽：pad 位必须全为零
    if width >= 64:
        return raw & ((1 << 64) - 1)
    return (raw >> start) & ((1 << width) - 1)
