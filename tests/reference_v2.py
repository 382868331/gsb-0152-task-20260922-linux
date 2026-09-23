"""独立逐位参考实现（仅测试使用）。

本模块从同一份“类型规约 spec”出发，用与 abi_marshal 完全独立的写法计算
布局、编码、8 字节块分类、寄存器/栈分配与证据位位置。它**只**用
abi_marshal 的类型构造器（那是输入数据）和常量名，绝不调用
layout_v2/classify_v2/plan_call/pack_call_v2/encode_v2/decode_v2 等被测
函数来产生期望值。

spec 规约：
    ("int", nbits, signed)   nbits ∈ {8,16,32,64,128}
    ("f", 64|32)
    ("p",)
    ("arr", spec, count)
    ("rec", [field, ...])    field = (name, spec) 或 ("bf", name, base_bits,
                             width, lsb_or_None, base_signed)
    ("union", [(name, spec), ...], tagged: bool)
"""

import struct

import abi_marshal as am

LE = "little"
BE = "big"


# ---------------------------------------------------------------------------
# spec -> abi_marshal Type（仅构造输入对象）
# ---------------------------------------------------------------------------

def build_type(spec):
    tag = spec[0]
    if tag == "int":
        _, bits, signed = spec
        if bits == 128:
            return am.UINT128 if not signed else am.INT128
        return am.IntType(bits // 8, bool(signed))
    if tag == "f":
        return am.DOUBLE if spec[1] == 64 else am.FLOAT32
    if tag == "p":
        return am.PTR
    if tag == "arr":
        return am.ArrayType(build_type(spec[1]), spec[2])
    if tag == "rec":
        r = am.RecordType("Rec")
        items = []
        for f in spec[1]:
            if f[0] == "bf":
                _, name, base_bits, width, lsb, bsigned = f
                items.append((name, am.BitField(
                    name, am.IntType(base_bits // 8, bool(bsigned)),
                    width, lsb)))
            else:
                name, sub = f
                items.append((name, build_type(sub)))
        r.define(items)
        return r
    if tag == "union":
        tt = am.UINT8 if spec[2] else None
        u = am.UnionType("U", tt)
        u.define([(n, build_type(s)) for n, s in spec[1]])
        return u
    raise AssertionError(tag)


# ---------------------------------------------------------------------------
# 独立布局：返回 (size, align, info)
# info 携带记录成员/位域组、联合成员等，供编码与位段枚举复用。
# ---------------------------------------------------------------------------

def _align_up(x, a):
    return (x + a - 1) // a * a


def _scalar_size_align(spec, max_align):
    tag = spec[0]
    if tag == "int":
        n = spec[1] // 8
    elif tag == "f":
        n = 8 if spec[1] == 64 else 4
    elif tag == "p":
        n = 8
    else:
        raise AssertionError(tag)
    return n, min(n, max_align)


def ref_layout(spec, max_align):
    """返回 (size, align)。规则独立实现。"""
    tag = spec[0]
    if tag in ("int", "f", "p"):
        return _scalar_size_align(spec, max_align)
    if tag == "arr":
        ssz, sal = ref_layout(spec[1], max_align)
        return ssz * spec[2], (sal if spec[2] else 1)
    if tag == "rec":
        off = 0
        al = 1
        fields = spec[1]
        i = 0
        while i < len(fields):
            f = fields[i]
            if f[0] == "bf":
                base_bits = f[2]
                j = i + 1
                while j < len(fields) and fields[j][0] == "bf" \
                        and fields[j][2] == base_bits:
                    j += 1
                unit = base_bits // 8
                gsize = _bf_group_size(fields[i:j], unit)
                ual = min(unit, max_align)
                off = _align_up(off, ual)
                off += gsize
                al = max(al, ual)
                i = j
            else:
                ssz, sal = ref_layout(f[1], max_align)
                off = _align_up(off, sal)
                off += ssz
                al = max(al, sal)
                i += 1
        return _align_up(off, al), al
    if tag == "union":
        members = spec[1]
        tagged = spec[2]
        tag_bytes = 1 if tagged else 0
        ms = 0
        ma = 1
        for _n, ms2, ma2 in ((n, *ref_layout(s, max_align)) for n, s in members):
            ms = max(ms, ms2)
            ma = max(ma, ma2)
        moff = _align_up(tag_bytes, ma)
        size = _align_up(moff + ms, max(ma, (min(1, max_align)) if not tagged
                                        else min(1, max_align)))
        return size, ma
    raise AssertionError(tag)


def _bf_group_parts(run, unit):
    """独立位域分配：返回 {name:[(rel_byte, lsb, width),...]}, group_bytes。"""
    parts = {}
    units = 1
    cursor = 0
    occupied = []
    unit_bits = unit * 8

    def claim(abs_lsb, width, name):
        for lo, hi in occupied:
            if abs_lsb < hi and lo < abs_lsb + width:
                raise am.OverlappingBitFieldError(
                    f"overlap {name}", "record 'Rec'")
        occupied.append((abs_lsb, abs_lsb + width))

    for f in run:
        _, name, base_bits, width, lsb, _signed = f
        assert base_bits == unit * 8
        if lsb is not None:
            claim(lsb, width, name)
            local = lsb % unit_bits
            su = lsb // unit_bits
            remain = unit_bits - local
            if width <= remain:
                parts.setdefault(name, []).append((su * unit, local, width))
            else:
                parts.setdefault(name, []).append((su * unit, local, remain))
                parts.setdefault(name, []).append(
                    ((su + 1) * unit, 0, width - remain))
            units = max(units, -(-(lsb + width) // unit_bits))
            cursor = (lsb + width) % unit_bits
        else:
            abs_lsb = (units - 1) * unit_bits + cursor
            claim(abs_lsb, width, name)
            if cursor + width <= unit_bits:
                parts.setdefault(name, []).append(
                    ((units - 1) * unit, cursor, width))
                cursor += width
            else:
                remain = unit_bits - cursor
                parts.setdefault(name, []).append(
                    ((units - 1) * unit, cursor, remain))
                units += 1
                parts.setdefault(name, []).append(
                    ((units - 1) * unit, 0, width - remain))
                cursor = width - remain
    return parts, units * unit


def _bf_group_size(run, unit):
    return _bf_group_parts(run, unit)[1]


# ---------------------------------------------------------------------------
# 独立编码（逐标量 + 位域整数读写；填充与未激活成员写 0）
# ---------------------------------------------------------------------------

def _check_range(spec, value):
    tag = spec[0]
    if tag == "int":
        bits = spec[1]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AssertionError("bad int")
        lo = -(1 << (bits - 1)) if spec[2] else 0
        hi = (1 << (bits - 1)) - 1 if spec[2] else (1 << bits) - 1
        if not lo <= value <= hi:
            raise AssertionError("range")
    elif tag == "f":
        if not isinstance(value, (float, am.FloatBits)):
            raise AssertionError("bad float")
    elif tag == "p":
        if not isinstance(value, int) or not 0 <= value <= 2**64 - 1:
            raise AssertionError("bad ptr")


def _scalar_bytes(spec, value, endian):
    tag = spec[0]
    if tag == "p":
        return value.to_bytes(8, endian)
    if tag == "int":
        return value.to_bytes(spec[1] // 8, endian, signed=bool(spec[2]))
    if tag == "f":
        if isinstance(value, am.FloatBits):
            return value.pattern.to_bytes(spec[1] // 8, endian)
        return struct.pack(("<d" if endian == LE else ">d")
                           if spec[1] == 64 else
                           ("<f" if endian == LE else ">f"), value)
    raise AssertionError(tag)


def _union_tag_index(spec, value):
    members = spec[1]
    tag = value.tag
    if isinstance(tag, str):
        return [n for n, _ in members].index(tag)
    return tag


def ref_encode(spec, value, endian, max_align):
    size, _al = ref_layout(spec, max_align)
    buf = bytearray(size)
    _encode_into(spec, value, endian, max_align, buf, 0)
    return bytes(buf)


def _encode_into(spec, value, endian, max_align, buf, base):
    tag = spec[0]
    if tag in ("int", "f", "p"):
        _check_range(spec, value)
        b = _scalar_bytes(spec, value, endian)
        buf[base:base + len(b)] = b
        return
    if tag == "arr":
        ssz, _ = ref_layout(spec[1], max_align)
        for i, v in enumerate(value):
            _encode_into(spec[1], v, endian, max_align, buf, base + i * ssz)
        return
    if tag == "rec":
        off = 0
        al = 1
        fields = spec[1]
        i = 0
        while i < len(fields):
            f = fields[i]
            if f[0] == "bf":
                unit = f[2] // 8
                j = i + 1
                while j < len(fields) and fields[j][0] == "bf" \
                        and fields[j][2] == f[2]:
                    j += 1
                run = fields[i:j]
                ual = min(unit, max_align)
                gstart = base + _align_up(off, ual)
                parts, gsize = _bf_group_parts(run, unit)
                for bf in run:
                    _, name, base_bits, width, _lsb, bsigned = bf
                    raw = value[name]
                    lo = -(1 << (width - 1)) if bsigned else 0
                    hi = (1 << (width - 1)) - 1 if bsigned else (1 << width) - 1
                    if isinstance(raw, bool) or not lo <= raw <= hi:
                        raise AssertionError("bf range")
                    pat = raw & ((1 << width) - 1)
                    segs = parts[name]
                    shift = 0
                    for (rb, lsb, w) in segs:
                        piece = (pat >> shift) & ((1 << w) - 1)
                        uabs = gstart + rb
                        cur = int.from_bytes(buf[uabs:uabs + unit], endian)
                        cur = (cur & (((1 << (unit * 8)) - 1)
                                      ^ (((1 << w) - 1) << lsb))) \
                            | (piece << lsb)
                        buf[uabs:uabs + unit] = cur.to_bytes(unit, endian)
                        shift += w
                off = _align_up(off, ual) + gsize
                al = max(al, ual)
                i = j
            else:
                name, sub = f
                ssz, sal = ref_layout(sub, max_align)
                off = _align_up(off, sal)
                _encode_into(sub, value[name], endian, max_align, buf,
                             base + off)
                off += ssz
                al = max(al, sal)
                i += 1
        return
    if tag == "union":
        members = spec[1]
        tagged = spec[2]
        idx = _union_tag_index(spec, value)
        if tagged:
            buf[base] = idx
        _, ma = ref_layout(spec, max_align)
        moff = _align_up(1 if tagged else 0, ma)
        _n, ms = members[idx]
        _encode_into(ms, value.value, endian, max_align, buf, base + moff)
        return
    raise AssertionError(tag)


# ---------------------------------------------------------------------------
# 独立逻辑位段枚举：(unit_start, unit_size, logical_lsb, width, cls)
# 联合给静态并集（全部成员 + 标签），与 classify 的最坏口径一致。
# ---------------------------------------------------------------------------

def _cls(spec):
    return am.FLOAT if spec[0] == "f" else am.INTEGER


def ref_static_ranges(spec, max_align, base=0, union_all=False, value=None):
    """union_all=True 时联合展开全部成员（分类用）；否则仅活动成员。"""
    tag = spec[0]
    if tag in ("int",):
        n = spec[1] // 8
        yield (base, n, 0, spec[1], _cls(spec))
    elif tag == "f":
        n = 8 if spec[1] == 64 else 4
        yield (base, n, 0, spec[1], _cls(spec))
    elif tag == "p":
        yield (base, 8, 0, 64, am.INTEGER)
    elif tag == "arr":
        ssz, _ = ref_layout(spec[1], max_align)
        for i in range(spec[2]):
            yield from ref_static_ranges(spec[1], max_align,
                                         base + i * ssz, union_all,
                                         value[i] if value else None)
    elif tag == "rec":
        off = 0
        fields = spec[1]
        i = 0
        while i < len(fields):
            f = fields[i]
            if f[0] == "bf":
                unit = f[2] // 8
                j = i + 1
                while j < len(fields) and fields[j][0] == "bf" \
                        and fields[j][2] == f[2]:
                    j += 1
                run = fields[i:j]
                ual = min(unit, max_align)
                gstart_rel = _align_up(off, ual)
                parts, gsize = _bf_group_parts(run, unit)
                for bf in run:
                    name = bf[1]
                    for (rb, lsb, w) in parts[name]:
                        ustart = base + gstart_rel + (rb // unit) * unit
                        yield (ustart, unit, (rb % unit) * 8 + lsb, w,
                               am.INTEGER)
                off = gstart_rel + gsize
                i = j
            else:
                name, sub = f
                ssz, sal = ref_layout(sub, max_align)
                off = _align_up(off, sal)
                yield from ref_static_ranges(sub, max_align, base + off,
                                             union_all,
                                             value[name] if value else None)
                off += ssz
                i += 1
    elif tag == "union":
        members = spec[1]
        tagged = spec[2]
        _, ma = ref_layout(spec, max_align)
        moff = _align_up(1 if tagged else 0, ma)
        if tagged:
            yield (base, 1, 0, 8, am.INTEGER)
        if union_all or value is None:
            for n, ms in members:
                yield from ref_static_ranges(ms, max_align, base + moff, True)
        else:
            idx = _union_tag_index(spec, value)
            yield from ref_static_ranges(members[idx][1], max_align,
                                         base + moff, False, value.value)
    else:
        raise AssertionError(tag)


def _phys_byte(ustart, usize, logical, endian):
    if endian == LE:
        return ustart + logical // 8
    return ustart + (usize - 1 - logical // 8)


def ref_classify(spec, max_align, threshold, endian):
    size, _ = ref_layout(spec, max_align)
    if size == 0:
        return ()
    if size > threshold:
        return (am.MEMORY,)
    mark = bytearray(size)  # 1 float, 2 int
    for (ustart, usize, lsb, w, cls) in ref_static_ranges(
            spec, max_align, union_all=True):
        for k in range(w):
            logical = lsb + k
            pb = _phys_byte(ustart, usize, logical, endian)
            level = 2 if cls == am.INTEGER else 1
            if mark[pb] < level:
                mark[pb] = level
    out = []
    for k in range(0, size, 8):
        block = mark[k:k + 8]
        out.append(am.INTEGER if 2 in block
                   else am.FLOAT if 1 in block else am.INTEGER)
    return tuple(out)


# ---------------------------------------------------------------------------
# 独立帧分配（普通实参 + 变参；不含共享，共享在专门测试里核对）
# ---------------------------------------------------------------------------

def ref_plan(args, desc, ret_spec=None, max_align=None):
    """args: [(spec, variadic), ...]。返回 placements/regs/frame 布局信息。"""
    endian = desc.endian
    ma = max_align if max_align is not None else desc.max_align
    thr = desc.int_regs.max_aggregate_bytes
    nic, nfc = desc.int_regs.int_count, desc.int_regs.float_count
    ni = nf = 0
    cursor = 0
    regions = []
    placements = []
    save_slots = {"int": [], "float": []}
    ret_indirect = False
    if ret_spec is not None:
        rsz, _ = ref_layout(ret_spec, ma)
        if rsz > thr or ret_spec[0] == "int" and ret_spec[1] == 128:
            ret_indirect = True
            regions.append(("return", 0, _align_up(rsz, desc.stack_align)))
            cursor = _align_up(rsz, desc.stack_align)

    def promote(spec):
        if spec[0] == "f" and spec[1] == 32:
            return ("f", 64)
        if spec[0] == "int" and spec[1] in (8, 16):
            return ("int", 32, spec[2])
        if spec[0] in ("int", "f", "p"):
            return spec
        raise am.VariadicPromotionError("not promotable", None)

    for i, (spec, var) in enumerate(args):
        eff = promote(spec) if var else spec
        size, _ = ref_layout(eff, ma)
        cls = ref_classify(eff, ma, thr, endian)
        if not cls:
            placements.append(dict(i=i, kind="none", cls=cls, eff=eff))
            continue
        if cls == (am.MEMORY,):
            off = _align_up(cursor, desc.stack_align)
            slot = _align_up(size, desc.stack_align)
            regions.append((f"arg{i}", off, slot))
            cursor = off + slot
            placements.append(dict(i=i, kind="stack", cls=cls, off=off,
                                   size=size, eff=eff))
            continue
        need_i, need_f = cls.count(am.INTEGER), cls.count(am.FLOAT)
        if ni + need_i <= nic and nf + need_f <= nfc:
            iregs, fregs = [], []
            for c in cls:
                if c == am.INTEGER:
                    iregs.append(ni)
                    if var:
                        save_slots["int"].append(ni)
                    ni += 1
                else:
                    fregs.append(nf)
                    if var:
                        save_slots["float"].append(nf)
                    nf += 1
            placements.append(dict(i=i, kind="registers", cls=cls,
                                   iregs=tuple(iregs), fregs=tuple(fregs),
                                   size=size, eff=eff, var=var))
        else:
            off = _align_up(cursor, desc.stack_align)
            slot = _align_up(size, desc.stack_align)
            regions.append((f"arg{i}", off, slot))
            cursor = off + slot
            placements.append(dict(i=i, kind="stack", cls=cls, off=off,
                                   size=size, eff=eff))

    save_needed = any(v for _s, v in args)
    if save_needed:
        base = _align_up(cursor, desc.stack_align)
        ri = base
        rf = ri + desc.save_int_slots * 8
        regions.append(("save_int", ri, desc.save_int_slots * 8))
        regions.append(("save_float", rf, desc.save_float_slots * 8))
        cursor = rf + desc.save_float_slots * 8
    frame_size = _align_up(cursor, desc.stack_align)
    return dict(placements=placements, regions=regions, frame_size=frame_size,
                save_slots=save_slots, ret_indirect=ret_indirect, ma=ma,
                endian=endian, thr=thr, desc=desc)


def ref_pack(args, values, desc, ret_spec=None, ret_value=None):
    """完整独立编组：返回 (int_patterns, float_patterns, frame_bytes,
    ret_patterns, plan)。"""
    plan = ref_plan(args, desc, ret_spec)
    ma = plan["ma"]
    endian = plan["endian"]
    ints = [None] * desc.int_regs.int_count
    flts = [None] * desc.int_regs.float_count
    frame = bytearray(plan["frame_size"])

    for pl, (spec, _var), val in zip(plan["placements"], args, values):
        if pl["kind"] == "none":
            continue
        eff = pl["eff"]
        data = ref_encode(eff, val, endian, ma)
        if pl["kind"] == "stack":
            frame[pl["off"]:pl["off"] + pl["size"]] = data
        else:
            ii = iter(pl["iregs"]); ff = iter(pl["fregs"])
            for k, c in enumerate(pl["cls"]):
                chunk = data[8 * k:8 * k + 8].ljust(8, b"\x00")
                pat = int.from_bytes(chunk, endian)
                if c == am.INTEGER:
                    ints[next(ii)] = pat
                else:
                    flts[next(ff)] = pat

    # 保存区
    for kind, slots in plan["save_slots"].items():
        if not slots:
            continue
        base = next(r[1] for r in plan["regions"]
                    if r[0] == f"save_{kind}")
        for slot in set(slots):
            pat = ints[slot] if kind == "int" else flts[slot]
            if pat is not None:
                frame[base + slot * 8:base + slot * 8 + 8] = \
                    pat.to_bytes(8, endian)

    ret_ipats = ret_fpats = None
    if ret_spec is not None and ret_value is not None:
        data = ref_encode(ret_spec, ret_value, endian, ma)
        if plan["ret_indirect"]:
            frame[0:len(data)] = data
        else:
            ri, rf = [], []
            for k, c in enumerate(ref_classify(ret_spec, ma, plan["thr"],
                                               endian)):
                chunk = data[8 * k:8 * k + 8].ljust(8, b"\x00")
                (ri if c == am.INTEGER else rf).append(
                    int.from_bytes(chunk, endian))
            ret_ipats, ret_fpats = tuple(ri), tuple(rf)
    return ints, flts, bytes(frame), (ret_ipats, ret_fpats), plan
