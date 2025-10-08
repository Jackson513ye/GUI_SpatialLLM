configfile: "config.yaml"
import re

# --- config values ---
LAS        = config["input"]
OUTDIR     = config["outdir"]
ATTRIBUTE  = config.get("attribute", "Classification")
GROUPS     = config.get("groups", {})
IDS        = [int(i) for i in config["ids"]]

ROOMS_CFG  = config.get("rooms", {})
COMBINED_PLY = ROOMS_CFG.get("input_combined", "data/output/combined.ply")
ROOMS_DIR     = ROOMS_CFG.get("outdir", f"{OUTDIR}/rooms")
VOX   = float(ROOMS_CFG.get("voxel", 0.15))
ZMIN  = float(ROOMS_CFG.get("slice_min", 0.25))
ZMAX  = float(ROOMS_CFG.get("slice_max", 2.25))
GRID  = float(ROOMS_CFG.get("grid", 0.10))
MIN_A = float(ROOMS_CFG.get("min_area", 1.0))
SAVE_COLORED = bool(ROOMS_CFG.get("save_colored_all", True))

# --- class dictionary (id -> name) ---
CLASSES = {
    0: "unclassified", 1: "ceiling", 2: "floor", 3: "wall", 4: "wall_ext",
    5: "beam", 6: "column", 7: "window", 8: "door", 9: "door_leaf",
    10: "plant", 11: "curtain", 12: "stairs", 13: "clutter", 14: "noise",
    15: "person", 16: "kitchen_cabinet", 17: "lamp", 18: "bed", 19: "table",
    20: "chair", 21: "couch", 22: "monitor", 23: "cupboard", 24: "shelves",
    25: "builtin_cabinet", 26: "tree", 27: "ground", 28: "car", 29: "grass", 30: "other"
}

# Build name lists and inverse mapping
NAMES    = [CLASSES[i] for i in IDS]
NAME2ID  = {v: k for k, v in CLASSES.items()}

# Constrain {name} to the exact allowed names so it won't match "combined_*"
NAME_REGEX = "(" + "|".join(re.escape(n) for n in NAMES) + ")"
wildcard_constraints:
    name = NAME_REGEX

# Helper: only depend on per-class outputs for IDs that are actually in IDS
def group_name_deps(wc):
    gid_list = GROUPS[wc.gname]["ids"]
    present = [i for i in gid_list if i in IDS]
    return [f"{OUTDIR}/{CLASSES[i]}.ply" for i in present]

# --------------------------
# TARGETS
# --------------------------
rule all:
    input:
        # per-class outputs by name
        expand(f"{OUTDIR}/{{name}}.ply", name=NAMES),
        # combined group outputs
        expand(f"{OUTDIR}/combined_{{gname}}.ply", gname=list(GROUPS.keys())),
        # single combined file (from all IDS)
        COMBINED_PLY,
        # directory produced by room-splitting on the combined ply
        directory(ROOMS_DIR)

# --------------------------
# PER-CLASS (single-ID -> name.ply)
# --------------------------
rule per_class:
    input:
        las = LAS
    output:
        ply = f"{OUTDIR}/{{name}}.ply"
    params:
        attribute = ATTRIBUTE,
        id = lambda wc: NAME2ID[wc.name]   # map name -> id for the script
    shell:
        "python py_scripts/las_to_ply.py "
        "--input {input.las} --output {output.ply} "
        "--attribute {params.attribute} --ids {params.id} --mode separate"

# --------------------------
# COMBINED GROUPS (named combos, optional)
# --------------------------
rule combined_group:
    input:
        group_name_deps
    output:
        ply = f"{OUTDIR}/combined_{{gname}}.ply"
    params:
        attribute = ATTRIBUTE,
        ids = lambda wc: ",".join(str(i) for i in GROUPS[wc.gname]["ids"]),
        las = LAS
    shell:
        "python py_scripts/las_to_ply.py "
        "--input {params.las} --output {output.ply} "
        "--attribute {params.attribute} --ids {params.ids} --mode combined"

# --------------------------
# SINGLE COMBINED FILE (from all 'ids')
# --------------------------


# --------------------------
# ROOM-SPLITTING (Option A) on the single combined file
# --------------------------
rule split_rooms_from_combined:
    input:
        ply = COMBINED_PLY
    output:
        directory(ROOMS_DIR)
    params:
        voxel=VOX, slice_min=ZMIN, slice_max=ZMAX, grid=GRID, min_area=MIN_A,
        save_colored=("--save-colored-all" if SAVE_COLORED else "")
    shell:
        "python py_scripts/segment_rooms.py "
        "--input {input.ply} --outdir {output} "
        "--voxel {params.voxel} --slice-min {params.slice_min} "
        "--slice-max {params.slice_max} --grid {params.grid} "
        "--min-room-area {params.min_area} {params.save_colored}"
