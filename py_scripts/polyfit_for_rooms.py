#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import time
import polyfit

def reconstruct_one(vg_path: Path, out_path: Path, solver_enum, w_data: float, w_cover: float, w_complex: float) -> tuple[bool, int]:
    """Return (success, faces)."""
    pc = polyfit.read_point_set(str(vg_path))
    if not pc:
        print(f"[polyfit] FAIL read: {vg_path}", file=sys.stderr, flush=True)
        return False, -1

    mesh = polyfit.reconstruct(pc, solver_enum, w_data, w_cover, w_complex)
    if not mesh:
        print(f"[polyfit] FAIL reconstruct: {vg_path}", file=sys.stderr, flush=True)
        return False, -1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not polyfit.save_mesh(str(out_path), mesh):
        print(f"[polyfit] FAIL save: {out_path}", file=sys.stderr, flush=True)
        return False, -1

    faces = mesh.size_of_facets()
    print(f"[polyfit] OK: {vg_path.name} -> {out_path.name} (faces={faces})", flush=True)

    # help GC
    del mesh
    del pc
    return True, faces

def main():
    ap = argparse.ArgumentParser(description="Batch PolyFit: convert *.vg to *.obj")
    ap.add_argument("--in-dir", type=Path, required=True, help="Directory with .vg files")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write .obj files")
    ap.add_argument("--glob", type=str, default="*.bvg", help="Glob pattern for inputs")
    ap.add_argument("--solver", type=str, default="SCIP", choices=["SCIP","GUROBI"], help="PolyFit solver enum name")
    ap.add_argument("--w-data", type=float, default=0.43, help="Weight: data fitting")
    ap.add_argument("--w-cover", type=float, default=0.27, help="Weight: model coverage")
    ap.add_argument("--w-complex", type=float, default=0.30, help="Weight: model complexity")
    args = ap.parse_args()

    in_dir  = args.in_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vg_paths = sorted(in_dir.glob(args.glob))
    print(f"[polyfit] Found {len(vg_paths)} .bvg files in {in_dir}", flush=True)

    # Initialize once
    polyfit.initialize()
    ok_all = True
    manifest = out_dir / "obj_manifest.csv"

    try:
        with manifest.open("w", encoding="utf-8") as mf:
            mf.write("vg_path,obj_path,faces,success\n")
            t0 = time.perf_counter()
            # map solver string to enum safely
            solver_enum = getattr(polyfit, args.solver, polyfit.SCIP)

            for i, vg in enumerate(vg_paths, 1):
                obj = out_dir / (vg.stem + ".obj")
                print(f"[polyfit] ({i}/{len(vg_paths)}) {vg.name}", flush=True)
                try:
                    success, faces = reconstruct_one(vg, obj, solver_enum, args.w_data, args.w_cover, args.w_complex)
                except Exception as e:
                    print(f"[polyfit] EXCEPTION on {vg.name}: {e}", file=sys.stderr, flush=True)
                    success, faces = False, -1
                mf.write(f"{vg},{obj},{faces},{int(success)}\n")
                mf.flush()
                ok_all = ok_all and success

            dt = time.perf_counter() - t0
            print(f"[polyfit] Done {len(vg_paths)} files in {dt:.1f}s (ok_all={ok_all})", flush=True)
    finally:
        # Write marker last, so Snakemake can depend on it
        (out_dir / "_SUCCESS").write_text(("OK\n" if ok_all else "SOME_FAIL\n"), encoding="utf-8")
        ok_all = True
    return 0 if ok_all else 1

if __name__ == "__main__":
    sys.exit(main())
