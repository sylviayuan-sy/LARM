#!/usr/bin/env python3
import os, re, argparse, shutil, xml.etree.ElementTree as ET
from xml.dom import minidom
from urdfpy import URDF

# ==========================================================
# Utility
# ==========================================================

def find_joint_indices(base_path, obj_id, mode="sap"):
    pat = re.compile(rf"eval_{obj_id}_joint_(\d+)$")
    out = []
    for d in os.listdir(base_path):
        m = pat.match(d)
        if m:
            out.append((int(m.group(1)), d))
    return sorted(out)

def load_xml(path):
    return ET.parse(path).getroot()

def copy_mesh(src, dst_dir):
    if not os.path.exists(src):
        print(f"⚠️ Missing mesh: {src}")
        return None
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    return os.path.basename(dst)

def get_joint_data(urdf_path):
    root = load_xml(urdf_path)
    j = root.find("joint")
    if j is None:
        return ("revolute", "0 0 1", "0", "0", "0 0 0", "0 0 0")
    typ = j.attrib.get("type", "revolute")
    ax = j.find("axis")
    axis = ax.attrib.get("xyz", "0 0 1") if ax is not None else "0 0 1"
    lim = j.find("limit")
    low = lim.attrib.get("lower", "0") if lim is not None else "0"
    up = lim.attrib.get("upper", "0") if lim is not None else "0"
    o = j.find("origin")
    xyz = o.attrib.get("xyz", "0 0 0") if o is not None else "0 0 0"
    rpy = o.attrib.get("rpy", "0 0 0") if o is not None else "0 0 0"
    return typ, axis, low, up, xyz, rpy

def get_link_origins(urdf_path):
    root = load_xml(urdf_path)
    ln = root.find("./link[@name='part']")
    if ln is None:
        return ["0 0 0"] * 6
    def _g(sec):
        if sec is None: return "0 0 0", "0 0 0"
        o = sec.find("origin")
        return (o.attrib.get("xyz", "0 0 0"), o.attrib.get("rpy", "0 0 0")) if o is not None else ("0 0 0", "0 0 0")
    ix, ir = _g(ln.find("inertial"))
    vx, vr = _g(ln.find("visual"))
    cx, cr = _g(ln.find("collision"))
    return ix, ir, vx, vr, cx, cr

def make_link(name, mesh, ix, ir, vx, vr, cx, cr):
    L = ET.Element("link", name=name)
    inert = ET.SubElement(L, "inertial")
    ET.SubElement(inert, "origin", xyz=ix, rpy=ir)
    ET.SubElement(inert, "mass", value="0.0")
    ET.SubElement(inert, "inertia", ixx="1", ixy="0", ixz="0", iyy="1", iyz="0", izz="1")
    vis = ET.SubElement(L, "visual"); ET.SubElement(vis, "origin", xyz=vx, rpy=vr)
    gv = ET.SubElement(vis, "geometry"); ET.SubElement(gv, "mesh", filename=mesh)
    col = ET.SubElement(L, "collision"); ET.SubElement(col, "origin", xyz=cx, rpy=cr)
    gc = ET.SubElement(col, "geometry"); ET.SubElement(gc, "mesh", filename=mesh)
    return L

def make_joint(name, parent, child, typ, axis, lo, up, xyz, rpy):
    J = ET.Element("joint", name=name, type=typ)
    ET.SubElement(J, "origin", xyz=xyz, rpy=rpy)
    ET.SubElement(J, "parent", link=parent)
    ET.SubElement(J, "child", link=child)
    ET.SubElement(J, "axis", xyz=axis)
    ET.SubElement(J, "limit", effort="0", velocity="0", lower=str(lo), upper=str(up))
    return J

# ==========================================================
# Core logic
# ==========================================================

def create_base_link(obj_id, min_idx, load_dir, out_dir, mode="sap"):
    L = ET.Element("link", name="base_link")
    if mode == "sap":
        base_mesh = os.path.join(
            load_dir, "sap_urdf_final",
            f"eval_{obj_id}_joint_{min_idx}",
            f"eval_{obj_id}_joint_{min_idx}_multilink_base.ply")
    else:
        base_mesh = os.path.join(
            load_dir,
            f"eval_{obj_id}_joint_{min_idx}",
            f"{obj_id}_joint_{min_idx}_mesh_1.00_pred_multilink_base_clean.ply")

    if os.path.exists(base_mesh):
        fn = copy_mesh(base_mesh, out_dir)
        vis = ET.SubElement(L, "visual")
        ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
        g = ET.SubElement(vis, "geometry")
        ET.SubElement(g, "mesh", filename=fn)
        print(f"✅ Base mesh: {base_mesh}")
    else:
        print(f"⚠️ No base mesh found for {obj_id}: {base_mesh}")
    return L


def combine_urdfs(load_dir, obj_id, mode="sap"):
    base_dir = os.path.join(load_dir, "sap_urdf_final") if mode == "sap" else load_dir
    joints = find_joint_indices(base_dir, obj_id, mode)
    if not joints:
        print(f"❌ No joints for {obj_id}")
        return

    min_idx = joints[0][0]

    # --- SAP vs TSDF output directory ---
    if mode == "sap":
        out_dir = os.path.join(base_dir, f"eval_{obj_id}_joint_{min_idx}")
    else:
        out_dir = os.path.join(load_dir, f"eval_{obj_id}_joint_{min_idx}")
    os.makedirs(out_dir, exist_ok=True)

    robot = ET.Element("robot", name=f"combined_eval_{obj_id}")
    robot.append(create_base_link(obj_id, min_idx, load_dir, out_dir, mode))

    for jidx, jfolder in joints:
        if mode == "sap":
            urdfp = os.path.join(base_dir, jfolder, "mobility.urdf")
            part = os.path.join(base_dir, jfolder, f"eval_{obj_id}_joint_{jidx}_part.ply")
        else:
            urdfp = os.path.join(base_dir, jfolder, f"eval_{obj_id}_joint_{jidx}.urdf")
            part = os.path.join(base_dir, jfolder, f"{obj_id}_joint_{jidx}_mesh_1.00_pred_part.ply")

        if not os.path.exists(urdfp) or not os.path.exists(part):
            print(f"⚠️ Skipping joint {jidx}")
            continue

        mesh = copy_mesh(part, out_dir)
        jt, ax, lo, up, xyz, rpy = get_joint_data(urdfp)
        ix, ir, vx, vr, cx, cr = get_link_origins(urdfp)
        link = make_link(f"link_{jidx}", mesh, ix, ir, vx, vr, cx, cr)
        joint = make_joint(f"joint_{jidx}", "base_link", f"link_{jidx}", jt, ax, lo, up, xyz, rpy)
        robot.append(link)
        robot.append(joint)

    out_path = os.path.join(out_dir, "mobility_multilink.urdf")
    xml_str = minidom.parseString(ET.tostring(robot, "utf-8")).toprettyxml(indent="  ")
    with open(out_path, "w") as f:
        f.write(xml_str)
    print(f"✅ Wrote {out_path}")

    try:
        u = URDF.load(out_path)
        print(f"✅ Parsed OK: links={len(u.links)}, joints={len(u.joints)}")
    except Exception as e:
        print(f"❌ Parse error for {obj_id}: {e}")

# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Combine per-joint URDFs into multilink URDF (SAP or TSDF).")
    ap.add_argument("--load_dir", required=True)
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--mode", choices=["sap", "tsdf"], default="sap")
    args = ap.parse_args()

    ids = []
    for line in open(args.datalist):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(\d+)(?=_joint_\d+\.json$)", line)
        if m:
            ids.append(m.group(1))
        elif line.isdigit():
            ids.append(line)

    for oid in sorted(set(ids)):
        print(f"\n=== Processing object {oid} (mode={args.mode}) ===")
        combine_urdfs(args.load_dir, oid, mode=args.mode)
