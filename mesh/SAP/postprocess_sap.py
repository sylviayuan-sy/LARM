#!/usr/bin/env python3
import os, re, argparse, shutil, json, numpy as np, trimesh, tqdm, torch
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import calibur
from color_projection import multiview_color_projection
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, TexturesVertex

VALID_SUB_RE = re.compile(r"^eval_(\d+)_joint_(\d+)$")
QPOS_RX = re.compile(r"_(?P<qpos>[+-]?\d+(?:\.\d+)?)\.png$", re.IGNORECASE)

def as_mesh(o):
    if isinstance(o, trimesh.Trimesh): return o
    if isinstance(o, trimesh.Scene):
        g=[g for g in o.geometry.values() if isinstance(g,trimesh.Trimesh)]
        return trimesh.util.concatenate(g) if g else None
    if isinstance(o,(list,tuple)):
        g=[g for g in o if isinstance(g,trimesh.Trimesh)]
        return trimesh.util.concatenate(g) if g else None
    return None

def ensure_dir(p): os.makedirs(p,exist_ok=True);return p
def _first_existing_dir(cs):
    for c in cs:
        if c and os.path.isdir(c): return c
    return None
def _scan_for_off_dir(root):
    hits=glob(os.path.join(root,"**","*_base.off"),recursive=True)
    if not hits:return None
    counts={}
    for h in hits:
        d=os.path.dirname(h)
        counts[d]=counts.get(d,0)+1
    return max(counts.items(),key=lambda kv:kv[1])[0]

def infer_paths(ld,o=None):
    o=o or {}
    mesh=_first_existing_dir([o.get("mesh_dir"),os.path.join(ld,"sap_out","generation","meshes"),os.path.join(ld,"generation","meshes"),_scan_for_off_dir(ld)])
    sap=_first_existing_dir([o.get("sap_dir"),os.path.join(ld,"sap_in"),ld])
    src=o.get("urdf_src_dir") or ld
    return {
        "mesh_dir":mesh,"sap_dir":sap,"urdf_src_dir":src,
        "ply_out_dir":o.get("ply_out_dir")or os.path.join(ld,"sap_unnorm"),
        "urdf_swap_out_dir":o.get("urdf_swap_out_dir")or os.path.join(ld,"sap_urdf_swap"),
        "final_out_dir":o.get("final_out_dir")or os.path.join(ld,"sap_urdf_final")
    }

def unnormalize_mesh(m,b):
    mi,ma=b;v=(m.vertices+0.5)*(ma-mi)+mi
    return trimesh.Trimesh(vertices=v,faces=m.faces,process=False)

def stage1_unnormalize_to_ply(ld,sap,mesh,outd):
    ensure_dir(outd)
    subs=[s for s in os.listdir(ld) if VALID_SUB_RE.match(s)]
    for sub in tqdm.tqdm(subs,desc="[Stage1] OFF->PLY"):
        for part in["base","part","multilink_base"]:
            mp=os.path.join(mesh,f"{sub}_{part}.off")
            npz=os.path.join(sap,f"{sub}_{part}","pointcloud.npz")
            op=os.path.join(outd,f"{sub}_{part}.ply")
            if not(os.path.isfile(mp)and os.path.isfile(npz)):continue
            try:
                m=as_mesh(trimesh.load(mp,force='mesh'))
                if m is None:continue
                nb=np.load(npz)["nbox"]
                unnormalize_mesh(m,nb).export(op)
            except:continue
    return subs

def update_urdf_paths(txt,sub):
    out=[]
    for l in txt.splitlines():
        if "<mesh" in l and "filename" in l:
            if "base" in l or "empty_base" in l:
                nl=re.sub(r'filename="[^"]+"',f'filename="{sub}_base.ply"',l)
            elif "part" in l or "empty_part" in l:
                nl=re.sub(r'filename="[^"]+"',f'filename="{sub}_part.ply"',l)
            else:
                nl=re.sub(r'filename="[^"]+"',f'filename="{sub}_base.ply"',l)
            out.append(nl)
        else:out.append(l)
    u="\n".join(out)
    if u==txt:raise AssertionError(f"Mesh names not found in URDF for {sub}")
    return u

def stage2_swap_urdf_and_copy_meshes(src,ply,outd,hint=None):
    ensure_dir(outd)
    subs=[os.path.basename(p).replace("_base.ply","") for p in glob(os.path.join(ply,"*_base.ply"))]
    if hint:subs=[s for s in subs if s in set(hint)]
    for sub in tqdm.tqdm(subs,desc="[Stage2] URDF swap"):
        up=os.path.join(src,sub,f"{sub}.urdf")
        if not os.path.isfile(up):continue
        try:
            txt=open(up).read()
            new=update_urdf_paths(txt,sub)
            sd=ensure_dir(os.path.join(outd,sub))
            open(os.path.join(sd,"mobility.urdf"),"w").write(new)
            for part in["base","part","multilink_base"]:
                s=os.path.join(ply,f"{sub}_{part}.ply")
                d=os.path.join(sd,f"{sub}_{part}.ply")
                if os.path.isfile(s):shutil.copyfile(s,d)
        except:continue

def fps(x):
    ang=[[3,0,0],[-3,0,0],[0,3,0],[0,-3,0],[0,0,-3],[0,0,3]]
    ids=[]
    for r in ang:
        d=np.linalg.norm(x-np.array(r),axis=-1)
        ids.append(np.argmin(d))
    return ids

def _intrinsics_from_transforms(t,s):
    if all(k in t for k in("fx","fy","cx","cy")):
        fx,fy,cx,cy=t["fx"],t["fy"],t["cx"],t["cy"]
        im=plt.imread(s);H,W=im.shape[0],im.shape[1]
        return fx,fy,cx,cy,(H,W)
    im=plt.imread(s);H,W=im.shape[0],im.shape[1]
    fx=0.5*W/np.tan(0.5*t["camera_angle_x"]);fy=fx;cx,cy=W/2,H/2
    return fx,fy,cx,cy,(H,W)

def _resolve_frame_image(b,f):
    imd=os.path.join(b,"images")
    if "file_path" in f:
        fp=f["file_path"];fp=fp[2:] if fp.startswith("./") else fp
        c=os.path.join(b,fp)
        if os.path.isfile(c):return c
        if not c.lower().endswith(".png")and os.path.isfile(c+".png"):return c+".png"
        base=os.path.basename(fp);c2=os.path.join(imd,base)
        if os.path.isfile(c2):return c2
        if not base.lower().endswith(".png")and os.path.isfile(c2+".png"):return c2+".png"
    i=f.get("idx");q=f.get("qpos")
    if i is not None and q is not None:
        p=os.path.join(imd,f"{i}_{q:.02f}.png")
        if os.path.isfile(p):return p
    if i is not None:
        p=os.path.join(imd,f"{i}.png")
        if os.path.isfile(p):return p
    ps=sorted(glob(os.path.join(imd,"*.png")))
    if ps:return ps[0]
    raise FileNotFoundError

def _parse_qpos(p):
    m=QPOS_RX.search(os.path.basename(p))
    if not m:return None
    try:return float(m.group("qpos"))
    except:return None

def _filter_frames(b,fs,tgt=1.0,eps=1e-6):
    out=[]
    for f in fs:
        q=f.get("qpos")
        if q is not None:
            if abs(float(q)-tgt)<=eps:out.append(f);continue
        try:
            p=_resolve_frame_image(b,f);q=_parse_qpos(p)
            if q is not None and abs(q-tgt)<=eps:out.append(f)
        except:continue
    return out

def stage3_colorize_from_multiview(urdf_in_dir: str, load_dir: str, out_dir_root: str, rot_aug: bool = False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)

    groups = {}
    for sub in os.listdir(urdf_in_dir):
        m = VALID_SUB_RE.match(sub)
        if not m:
            continue
        oid, jid = map(int, m.groups())
        groups.setdefault(oid, []).append((jid, sub))

    for oid, items in tqdm.tqdm(groups.items(), desc="[Stage3] Color"):
        jid_min, sub_min = min(items)
        for jid, sub in items:
            eval_id = sub
            base_dir = os.path.join(load_dir, eval_id)
            out_dir = ensure_dir(os.path.join(out_dir_root, eval_id))
            urdf_dir = os.path.join(urdf_in_dir, eval_id)

            try:
                base_p = as_mesh(trimesh.load(f'{urdf_dir}/{eval_id}_base.ply', force='mesh'))
                part_p = as_mesh(trimesh.load(f'{urdf_dir}/{eval_id}_part.ply', force='mesh'))
                if base_p is None or part_p is None:
                    raise FileNotFoundError("Missing base/part PLY")

                multilink_path = None
                if jid == jid_min:
                    mlp = f'{urdf_dir}/{eval_id}_multilink_base.ply'
                    if os.path.isfile(mlp):
                        multilink_path = mlp
                        ml_p = as_mesh(trimesh.load(mlp, force='mesh'))
                        if ml_p is not None:
                            ml_p = ml_p.subdivide().subdivide()
                    else:
                        ml_p = None
                else:
                    ml_p = None

                base_p = base_p.subdivide().subdivide()
                part_p = part_p.subdivide().subdivide()

                tjson = os.path.join(base_dir, 'transforms.json')
                with open(tjson) as fi:
                    transforms = json.load(fi)
                frames_all = transforms.get('frames', [])
                frames_q1 = _filter_frames(base_dir, frames_all, tgt=1.0, eps=1e-6)
                if not frames_q1:
                    continue

                fx, fy, cx, cy, (H, W) = _intrinsics_from_transforms(transforms, _resolve_frame_image(base_dir, frames_q1[0]))

                scale_factor = 1.0
                if rot_aug:
                    circ_rad = 0.5
                    radius_scale = 1.5
                    fovy = 2 * np.arctan(H / (2 * fx))
                    natural_distance = (circ_rad / np.tan(fovy / 2.0)) * radius_scale
                    rot_aug_distance = 1.0
                    scale_factor = natural_distance / rot_aug_distance

                imgs, cams = [], []
                for frame in frames_q1:
                    rgb_path = _resolve_frame_image(base_dir, frame)
                    q_infer = _parse_qpos(rgb_path)
                    if (q_infer is None) or (abs(q_infer - 1.0) > 1e-6):
                        continue
                    rgb_i = plt.imread(rgb_path)
                    mask_path = os.path.splitext(rgb_path)[0] + "_mask.png"
                    if os.path.isfile(mask_path):
                        mask_i = plt.imread(mask_path)[..., None]
                        mask_i = np.clip(mask_i, 0, 1)
                    else:
                        mask_i = np.ones_like(rgb_i[..., :1])
                    rgba = np.concatenate([rgb_i[..., :3], mask_i], axis=-1)
                    imgs.append(Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8), mode='RGBA'))

                    cam_matrix_gl = np.array(frame['transform_matrix'], dtype=np.float32)
                    cam_matrix_p3d = np.linalg.inv(calibur.convert_pose(cam_matrix_gl, calibur.CC.GL, ('-X','Y','Z')))
                    cam_matrix_p3d[:3, 3] *= scale_factor
                    
                    cams.append(PerspectiveCameras(
                        focal_length=((fx, fy),),
                        principal_point=((cx, cy),),
                        R=torch.tensor(cam_matrix_p3d[:3, :3].T[None]).float().to(device),
                        T=torch.tensor(cam_matrix_p3d[:3, 3][None]).float().to(device),
                        in_ndc=False,
                        image_size=((H, W),),
                        device=device
                    ))

                if not imgs:
                    continue
                avg_color = np.mean([np.asarray(im)[..., :3].mean((0, 1)) / 255.0 for im in imgs], axis=0)
                verts = torch.from_numpy(np.vstack([base_p.vertices, part_p.vertices]).astype(np.float32)).to(device)
                f_base = torch.from_numpy(base_p.faces.astype(np.int64)).to(device)
                f_part = torch.from_numpy(part_p.faces.astype(np.int64)).to(device) + len(base_p.vertices)
                faces = torch.cat([f_base, f_part], dim=0)
                init_tex = torch.from_numpy(np.tile(avg_color[None, :], (verts.shape[0], 1)).astype(np.float32)).to(device)
                mesh = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(verts_features=[init_tex]))
                weights = np.ones(len(imgs), dtype=np.float32).tolist()
                mesh_col = multiview_color_projection(mesh, imgs, cams, camera_focal=fx/imgs[0].size[0], complete_unseen=True, resolution=512, weights=weights)
                tex = mesh_col.textures.verts_features_list()[0].detach().cpu().numpy()
                nb = len(base_p.vertices)
                base_p_c = trimesh.Trimesh(base_p.vertices, base_p.faces, vertex_colors=tex[:nb], process=False)
                part_p_c = trimesh.Trimesh(part_p.vertices, part_p.faces, vertex_colors=tex[nb:], process=False)

                shutil.copyfile(f'{urdf_dir}/mobility.urdf', f'{out_dir}/mobility.urdf')
                base_p_c.export(f'{out_dir}/{eval_id}_base.ply')
                part_p_c.export(f'{out_dir}/{eval_id}_part.ply')

                if ml_p is not None:
                    verts_m = torch.from_numpy(ml_p.vertices.astype(np.float32)).to(device)
                    faces_m = torch.from_numpy(ml_p.faces.astype(np.int64)).to(device)
                    tex_m = torch.from_numpy(np.tile(avg_color[None, :], (verts_m.shape[0], 1)).astype(np.float32)).to(device)
                    mesh_m = Meshes(verts=[verts_m], faces=[faces_m], textures=TexturesVertex(verts_features=[tex_m]))
                    weights_m = np.ones(len(imgs), dtype=np.float32).tolist()
                    mesh_col_m = multiview_color_projection(mesh_m, imgs, cams, camera_focal=fx/imgs[0].size[0], complete_unseen=True, resolution=512, weights=weights_m)
                    x0 = mesh_col_m.textures.verts_features_list()[0].detach().cpu().numpy()
                    ml_p_c = trimesh.Trimesh(ml_p.vertices, ml_p.faces, vertex_colors=x0, process=False)
                    ml_p_c.export(f'{out_dir}/{eval_id}_multilink_base.ply')

            except Exception as e:
                print(f"[WARN] Skipping {eval_id}: {e}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--load_dir",type=str,default="./output")
    p.add_argument("--sap_dir",type=str,default=None)
    p.add_argument("--mesh_dir",type=str,default=None)
    p.add_argument("--ply_out_dir",type=str,default=None)
    p.add_argument("--urdf_src_dir",type=str,default=None)
    p.add_argument("--urdf_swap_out_dir",type=str,default=None)
    p.add_argument("--final_out_dir",type=str,default=None)
    p.add_argument("--rot-aug",action="store_true",help="Apply camera distance correction for rotation augmentation mode")
    a=p.parse_args()
    paths=infer_paths(a.load_dir,{"sap_dir":a.sap_dir,"mesh_dir":a.mesh_dir,"ply_out_dir":a.ply_out_dir,"urdf_src_dir":a.urdf_src_dir,"urdf_swap_out_dir":a.urdf_swap_out_dir,"final_out_dir":a.final_out_dir})
    subs=stage1_unnormalize_to_ply(a.load_dir,paths["sap_dir"],paths["mesh_dir"],paths["ply_out_dir"])
    stage2_swap_urdf_and_copy_meshes(paths["urdf_src_dir"],paths["ply_out_dir"],paths["urdf_swap_out_dir"],subs)
    stage3_colorize_from_multiview(paths["urdf_swap_out_dir"],a.load_dir,paths["final_out_dir"],rot_aug=a.rot_aug)

if __name__=="__main__":main()