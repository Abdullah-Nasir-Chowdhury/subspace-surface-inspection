"""Stage 2. Frozen features and end-to-end scores of every window.

Per window: DentNet's own dent probability (mean and max over the window's pixels), its
1024-D bottleneck feature, the 128-D features of the three PointNet++ encoders (radius
0.3, 0.6, 2.0) and the radius-0.3 head's probability, and the DINOv2 feature exactly as on
Real3D-AD (r3_dinov2.Dino: the window resized to 224 px, hidden layers -12 to -18 averaged,
patch tokens pooled). `--synthetic` encodes the synthetic training and validation windows
of the paper (IDs 0 to 499 and 24000 to 24499, 340 windows each) the same per-window way,
because the fuselage stage encoded whole residual images and the two are not comparable.

Output: ~/accv_results_mvtec/feats_<proto>/<scan id>.npz, syn_dinov2_{train,val}.npz
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import mv_common as M
import os, glob, argparse
import numpy as np
from mv_common import C
from r3_dinov2 import Dino
import shift_analysis as RV7


class Enc:
    def __init__(self, device="cuda"):
        import torch
        from inference_ransac import load_model
        from pnpp_model import PatchClassifier
        from patches import normalize_patch
        self.torch, self.device, self.normalize_patch = torch, device, normalize_patch
        self.dentnet = load_model(C.DENTNET_CKPT, device=device)
        self.ext = C.dentnet_extractor(device)
        self.pnpp = {}
        for name, (tag, rad) in RV7.PNPP_RUNS.items():
            m = PatchClassifier(norm_radius=rad).to(device)
            m.load_state_dict(torch.load(os.path.expanduser("~/accv_results_real/%s/best.pth" % tag), map_location=device)); m.eval()
            self.pnpp[name] = m
        self.dino = Dino(device)
        ax = (np.arange(M.PATCH) - (M.PATCH - 1) / 2.0) * M.PITCH_MM
        Yg, Xg = np.meshgrid(ax, ax, indexing="ij")
        self.Xg, self.Yg = Xg.astype(np.float32), Yg.astype(np.float32)

    def __call__(self, R, rng):
        torch = self.torch
        R = np.asarray(R, np.float32); n = len(R)
        out = {}
        if n == 0:
            return out
        pm, px = [], []
        with torch.no_grad():
            for i in range(0, n, 256):
                x = torch.from_numpy(R[i:i + 256]).unsqueeze(1).to(self.device)
                pr = torch.softmax(self.dentnet(x), dim=1)[:, 1].cpu().numpy().reshape(-1, M.PATCH * M.PATCH)
                pm.append(pr.mean(1)); px.append(pr.max(1))
        out["dentnet_prob_mean"] = np.concatenate(pm); out["dentnet_prob_max"] = np.concatenate(px)
        out["dentnet_1024"] = np.asarray(self.ext.extract_features_batch([r for r in R], batch_size=256), np.float16)
        arr = np.empty((n, C.N_POINTS, 3), np.float32)
        for k in range(n):
            q = np.stack([self.Xg.ravel(), self.Yg.ravel(), R[k].ravel()], 1)
            sel = rng.choice(q.shape[0], C.N_POINTS, replace=False)
            arr[k] = self.normalize_patch(q[sel])
        for name, m in self.pnpp.items():
            prob, f = C.pnpp_forward(m, arr, self.device)
            out[name] = f.astype(np.float16)
            if name == "pnpp_r0.3_128":
                out["pnpp_prob"] = prob
        out["dinov2_1024"] = self.dino(R).astype(np.float16)
        return out


def run_scans(enc):
    for proto in ("metric", "sim"):
        src = os.path.join(M.OUT, "windows_" + proto); dst = os.path.join(M.OUT, "feats_" + proto)
        os.makedirs(dst, exist_ok=True)
        fps = sorted(glob.glob(os.path.join(src, "*.npz")))
        for k, fp in enumerate(fps, 1):
            o = os.path.join(dst, os.path.basename(fp))
            if os.path.exists(o):
                continue
            z = np.load(fp)
            rng = np.random.default_rng(abs(hash(os.path.basename(fp))) % (2 ** 32))
            out = enc(z["R"], rng)
            np.savez_compressed(o, frac=z["frac"], pos=z["pos"], cat=str(z["cat"]), split=str(z["split"]),
                                dtype=str(z["dtype"]), stem=str(z["stem"]), defective=bool(z["defective"]), **out)
            if k % 25 == 0:
                M.log(proto, "[%d/%d]" % (k, len(fps)), os.path.basename(fp), len(z["R"]), "windows")


def run_synthetic(enc):
    from patch_utils import PatchExtractor
    pe = PatchExtractor(patch_size=M.PATCH, stride=M.STRIDE, ignore_edge_patches=True)
    for tag, ids in (("train", range(0, 500)), ("val", range(24000, 24500))):
        o = os.path.join(os.path.expanduser("~/accv_results_mvtec"), "syn_dinov2_%s.npz" % tag)   # shared, not per MV_OUT
        if os.path.exists(o):
            continue
        ref = np.load(os.path.join(C.PBSS_RES, "features_%s.npz" % tag))
        feats = []
        for k, i in enumerate(ids):
            res = np.load("%s/%05d_input.npy" % (C.SYN, i)).astype(np.float32)
            info = pe.extract_patches(res); valid = np.array(info["valid_mask"])
            R = np.stack([p for j, p in enumerate(info["patches"]) if valid[j]])
            feats.append(enc.dino(R).astype(np.float16))
            if k % 50 == 0:
                M.log("synthetic", tag, k)
        F = np.concatenate(feats)
        assert len(F) == len(ref["labels"]), (len(F), len(ref["labels"]))
        np.savez_compressed(o, features=F, labels=ref["labels"], sample_idx=ref["sample_idx"])
        M.log("synthetic", tag, "done", F.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    enc = Enc()
    if a.synthetic:
        run_synthetic(enc)
    else:
        run_scans(enc)
    print("ALL_DONE_M2", flush=True)


if __name__ == "__main__":
    main()
