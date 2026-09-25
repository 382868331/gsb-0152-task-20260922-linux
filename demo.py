"""demo.py — abi_marshal 固定输入演示（离线、确定性、约 1 秒内完成）。

演示内容：
1. 构造一组聚合类型并打印逐字段字节布局与 8 字节块分类；
2. pack_call 编组一次混合参数调用，打印寄存器与栈映像，并重建参数；
3. 参数侧越界拒绝（u8 赋值 300）与 NaN 拒绝；
4. 返回走寄存器：混合整浮记录逐块进入两个整数/两个浮点返回寄存器；
5. 大聚合间接返回：调用方有界缓冲区 + 隐藏指针先占整数参数寄存器，
   刚好把后面的聚合挤到栈；被调方写入，调用方按字节核对后读取；
6. 缓冲区边界拒绝（容量不足 / 未对齐）与返回值越界不写部分结果。
"""

import abi_marshal as am


def rec(name, fields):
    r = am.RecordType(name)
    r.define(fields)
    return r


def show_layout(t):
    lay = am.layout(t)
    print(f"  {t.name}: size={lay.size} align={lay.align} "
          f"classify={am.classify(t)}")
    for f in lay.fields:
        print(f"    +{f.offset:<3} {f.name}: {f.type.name} "
              f"(size={f.layout.size}, align={f.layout.align})")


def main():
    print("== 1. 类型布局与分类 ==")
    empty = rec("Empty", [])
    header = rec("Header", [("tag", am.UINT8), ("flags", am.UINT8),
                            ("count", am.UINT32)])
    point = rec("Point", [("x", am.DOUBLE), ("y", am.DOUBLE)])
    packet = rec("Packet", [("hdr", header),
                            ("pos", point),
                            ("samples", am.ArrayType(am.INT16, 4)),
                            ("reserved", empty)])
    mixed = rec("Mixed", [("weight", am.DOUBLE), ("id", am.INT32)])
    for t in (empty, header, point, packet, mixed):
        show_layout(t)

    print("\n== 2. pack_call：参数寄存器与栈映像 ==")
    args = [
        (am.UINT64, 0x1122334455667788),          # -> 整数寄存器 0
        (am.DOUBLE, 2.5),                          # -> 浮点寄存器 0
        (mixed, {"weight": 0.5, "id": 42}),        # 混合类：FLOAT+INTEGER
        (packet, {"hdr": {"tag": 1, "flags": 2, "count": 4},
                  "pos": {"x": 1.25, "y": -0.5},
                  "samples": [10, -20, 30, -40],
                  "reserved": {}}),                # size 32 > 16 -> 栈
        (empty, {}),                               # 空记录：不占任何资源
        (am.INT64, -1),                            # -> 整数寄存器
    ]
    image = am.pack_call(args)
    for i, (t, _) in enumerate(args):
        p = image.placements[i]
        print(f"  arg[{i}] {t.name}: kind={p.kind} eightbytes={p.eightbytes}"
              f" int_regs={p.int_regs} float_regs={p.float_regs}"
              f" stack_offset={p.stack_offset} size={p.size}")
    print(f"  int_regs   = {[hex(v) if v is not None else None for v in image.int_regs]}")
    print(f"  float_regs = {list(image.float_regs)}")
    print(f"  stack ({len(image.stack)} bytes) = {image.stack.hex()}")
    rebuilt = am.unpack_call([t for t, _ in args], image)
    print(f"  rebuilt == original : {rebuilt == [v for _, v in args]}")

    print("\n== 3. 参数侧拒绝：越界整数与 NaN ==")
    try:
        am.pack_call([(am.UINT8, 300)])
        print("  ERROR: 未触发拒绝！")
    except am.IntegerOverflowError as exc:
        print(f"  触发 IntegerOverflowError: {exc}")
    try:
        am.pack_call([(am.DOUBLE, float("nan"))])
        print("  ERROR: 未触发拒绝！")
    except am.NonFiniteFloatError as exc:
        print(f"  触发 NonFiniteFloatError: {exc}")

    # ---- 返回值 ABI ----
    print("\n== 4. 返回走寄存器：混合整浮 {f64, i32} ==")
    ret_mixed = rec("RetMix", [("ratio", am.DOUBLE), ("code", am.INT32)])
    inv = am.pack_invoke(ret_mixed, [(am.INT64, 3)])
    print(f"  返回约定: {inv.convention.kind} {inv.convention.eightbytes}")
    # 模拟被调方：解参、计算、编码返回
    n = am.unpack_call([am.INT64], inv.call_image)[0]
    ret = am.encode_return(inv, {"ratio": 1.0 / n, "code": n * 10})
    print(f"  返回寄存器: int={tuple(hex(v) if isinstance(v, int) else v for v in ret.int_regs)}"
          f" float={ret.float_regs}")
    print(f"  调用方读取: {am.decode_return(inv, ret)}")

    print("\n== 5. 大聚合间接返回：32 字节嵌套数组 Grid ==")
    two_i64 = rec("TwoI64", [("a", am.INT64), ("b", am.INT64)])
    grid = rec("Grid", [
        ("grid", am.ArrayType(am.ArrayType(am.INT16, 3), 2)),
        ("scale", am.DOUBLE),
        ("tag", am.UINT8),
    ])  # i16[2][3]@0(12) + pad + f64@16 + u8@24 + 尾填充 => 32 字节
    show_layout(grid)

    call_args = [(am.INT64, 0x1111), (am.INT64, 0x2222),
                 (two_i64, {"a": 0x3333, "b": 0x4444})]
    # 对照：若无间接返回，TwoI64 恰好占整数寄存器 2、3
    plain = am.pack_invoke(am.UINT64, call_args)
    print(f"  无隐藏指针: arg2.kind={plain.call_image.placements[2].kind} "
          f"int_regs={plain.call_image.int_regs[:4]}")

    # 调用方在合成内存中准备 32 字节、8 对齐的有界缓冲区
    mem = am.Memory(1 << 16)
    addr = 0x1000
    invoke = am.pack_invoke(grid, call_args, mem, addr, capacity=64)
    print(f"  隐藏指针 -> 整数参数寄存器0 = {hex(invoke.hidden_pointer)}")
    img = invoke.call_image
    print(f"  有隐藏指针: int_regs={tuple(hex(v) if v is not None else None for v in img.int_regs)}")
    p2 = img.placements[2]
    print(f"  arg2(TwoI64).kind={p2.kind} stack_offset={p2.stack_offset} "
          f"（被隐藏指针挤到栈）")
    print(f"  栈映像前16字节 = {img.stack[:16].hex()}")

    # 被调方：解参 + 模拟计算 + 编码返回（一次性写入调用方缓冲区）
    n1, n2, pair = am.unpack_call([am.INT64, am.INT64, two_i64], img)
    result = {"grid": [[n1, -n1, pair["a"]], [pair["b"], -3, 4]],
              "scale": 0.5, "tag": 7}
    ret = am.encode_return(invoke, result)
    raw = mem.read(addr, 32)
    print(f"  返回 kind={ret.kind}；缓冲区 32 字节（padding 写零）：")
    print(f"    {raw.hex()}")
    print(f"  调用方读取: {am.decode_return(invoke, ret)}")

    print("\n== 6. 缓冲区与返回值边界拒绝 ==")
    # 容量不足：先填哨兵，失败后哨兵原样（无部分写入）
    small = am.Memory(32)
    small.fill(0, 32, 0xCD)
    try:
        am.pack_invoke(grid, [], small, 0, capacity=31)
        print("  ERROR: 未触发容量拒绝！")
    except am.BufferError as exc:
        print(f"  容量不足 BufferError: {exc}")
    print(f"  哨兵保持原样（无写入）: {small.read(0, 32) == b'\xcd' * 32}")
    # 未对齐
    try:
        am.pack_invoke(grid, [], am.Memory(64), 4, capacity=60)
        print("  ERROR: 未触发对齐拒绝！")
    except am.BufferError as exc:
        print(f"  未对齐 BufferError: {exc}")
    # 返回值整数越界：发生在写入前，缓冲区保持哨兵
    mem2 = am.Memory(64)
    mem2.fill(0, 32, 0xEE)
    inv2 = am.pack_invoke(grid, [], mem2, 0, capacity=64)
    bad = {"grid": [[1, 32768, 3], [-4, 5, -6]], "scale": 1.0, "tag": 1}
    try:
        am.encode_return(inv2, bad)
        print("  ERROR: 未触发越界拒绝！")
    except am.IntegerOverflowError as exc:
        print(f"  返回值越界 IntegerOverflowError: {exc}")
    print(f"  缓冲区未被部分写入: {mem2.read(0, 32) == b'\xee' * 32}")

    print("\ndemo 完成。")


if __name__ == "__main__":
    main()
