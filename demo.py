"""Deterministic demo of abi_marshal: pack one call, show the byte layout,
register file and stack image, rebuild the arguments, then trigger one
real out-of-range rejection.  Runs in well under 8 seconds, no I/O.
"""

import abi_marshal as abi


def show_layout(name, typ):
    lay = abi.layout(typ)
    cls = abi.classify(typ)
    print(f"  {name:<10} size={lay.size:<3} align={lay.align}  class={cls}")
    for f in lay.fields:
        print(f"    +{f.offset:<3} {f.name}")


def show_image(img):
    for i, reg in enumerate(img.int_regs):
        if reg is not None:
            print(f"  int_reg[{i}]   = 0x{reg.hex()}")
    for i, reg in enumerate(img.float_regs):
        if reg is not None:
            print(f"  float_reg[{i}] = 0x{reg.hex()}")
    if img.stack:
        print(f"  stack ({len(img.stack)} bytes):")
        for off in range(0, len(img.stack), 8):
            print(f"    +{off:<3} 0x{img.stack[off:off + 8].hex()}")
    else:
        print("  stack: (empty)")
    for p in img.placements:
        print(f"  arg{p.index}: size={p.size} chunks={p.chunks}")


def main():
    # ---- types ------------------------------------------------------
    mixed = abi.RecordType("Mixed", [("tag", abi.I8), ("weight", abi.F64)])
    pair = abi.RecordType("Pair", [("a", abi.I64), ("b", abi.I64)])
    wide = abi.RecordType("Wide", [("x", abi.I64), ("y", abi.I64), ("z", abi.I64)])
    empty = abi.RecordType("Empty")
    samples = abi.ArrayType(abi.I16, 3)

    print("== byte layouts ==")
    for name, typ in [("Mixed", mixed), ("Pair", pair), ("Wide", wide),
                      ("Empty", empty), ("i16[3]", samples)]:
        show_layout(name, typ)

    # ---- one call ---------------------------------------------------
    # int regs: 3 scalars + Mixed.chunk0 = 4 (full).  Pair needs 2 int
    # regs but none remain -> whole Pair falls back to a 16-byte-aligned
    # stack slot and consumes no registers; the f64 still gets float
    # reg 1; Wide (>16 bytes) is MEMORY; i16[3] needs 1 int reg, finds
    # none, and spills to the next 16-byte-aligned stack slot.
    types = [abi.I64, abi.I64, abi.I64, mixed, empty, pair, abi.F64, wide, samples]
    values = [
        11, -22, 33,
        {"tag": 0x41, "weight": 2.5},
        {},
        {"a": 1000, "b": -2000},
        3.25,
        {"x": 7, "y": 8, "z": 9},
        [0x0201, 0x0403, -5],
    ]
    image = abi.pack_call(types, values)

    print("\n== packed call image ==")
    show_image(image)

    # ---- rebuild the arguments from the image -----------------------
    rebuilt = abi.unpack_call(types, image)
    print("\n== rebuilt arguments ==")
    for i, v in enumerate(rebuilt):
        print(f"  arg{i} = {v!r}")
    assert rebuilt == values, "round-trip mismatch"
    print("round-trip: OK (rebuilt arguments equal the originals)")

    # ---- one real out-of-range rejection ----------------------------
    print("\n== out-of-range rejection ==")
    try:
        abi.pack_call([abi.U8], [300])
    except abi.IntegerOverflowError as err:
        print(f"  rejected as expected: {type(err).__name__}: {err}")
    else:  # pragma: no cover
        raise SystemExit("expected IntegerOverflowError was not raised")


if __name__ == "__main__":
    main()
