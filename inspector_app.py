# inspector_app.py
import os
import cv2
import json
import glob
import time
import threading
import traceback
import queue
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

import torch
from torchvision import transforms
from pymodbus.client.sync import ModbusSerialClient as ModbusClient

from resnet_feature_extractor import ResNetFeatureExtractor, embedding_concat, random_channel_indices, compute_gaussian_stats

# ---------------- CONFIG ----------------
DATA_ROOT = "data_variants"
MODEL_ROOT = "models"
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

VARIANTS = {
    "12ml_black": {"data": os.path.join(DATA_ROOT, "12ml_black"), "model": os.path.join(MODEL_ROOT, "padim_12ml_black.npz")},
    "12ml_white": {"data": os.path.join(DATA_ROOT, "12ml_white"), "model": os.path.join(MODEL_ROOT, "padim_12ml_white.npz")},
    "8ml_black":  {"data": os.path.join(DATA_ROOT, "8ml_black"),  "model": os.path.join(MODEL_ROOT, "padim_8ml_black.npz")},
    "8ml_white":  {"data": os.path.join(DATA_ROOT, "8ml_white"),  "model": os.path.join(MODEL_ROOT, "padim_8ml_white.npz")},
}
for v in VARIANTS.values():
    os.makedirs(v["data"], exist_ok=True)

DEFAULT_MODEL = list(VARIANTS.values())[0]["model"]

# Camera / inference settings
CAM_INDEX = 0
INFER_DOWNSCALE = 160
EMBED_DIMS = 80
DEFAULT_THRESHOLD = 3.5

# PLC / Modbus settings
SERIAL_PORT = "COM3"
BAUDRATE = 9600
SLAVE_ID = 1
DISCRETE_INPUT_ADDR = 1024
COIL_ADDR = 1280
REJECT_COIL_ADDR = 1281

# Polling & queue
MODBUS_POLL_INTERVAL = 0.01
EVENT_QUEUE_MAX = 400

# default timings (ms)
DEFAULT_PRE_ACT_MS = 0
DEFAULT_CAMERA_DELAY_MS = 50
DEFAULT_CAPTURE_OFFSET_MS = 40
DEFAULT_MAIN_ACTIVE_MS = 220
DEFAULT_REJECT_AFTER_MS = 50
DEFAULT_REJECT_ACTIVE_MS = 180

# Other
PREVIEW_LOOP_MS = 20

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat_net = ResNetFeatureExtractor().to(device)
channel_idx = None
stats = None

to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INFER_DOWNSCALE, INFER_DOWNSCALE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def load_padim(path):
    global stats, channel_idx
    if not path or not os.path.exists(path):
        print("Model not found:", path)
        return False
    d = np.load(path, allow_pickle=True)
    stats = {"means": d["means"], "inv_covs": d["inv_covs"], "H": int(d["H"]), "W": int(d["W"])}
    channel_idx = d["channel_idx"]
    print("Loaded model:", path)
    return True

def score_anomaly(crop_bgr):
    global stats, channel_idx
    if stats is None:
        return None, None
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feat_net(t)
        emb = embedding_concat(feats)
        emb = emb[:, channel_idx, :, :].squeeze(0).cpu().numpy()
    M,H,W = emb.shape
    emb2 = emb.reshape(M, H*W).T
    means = stats["means"]; invs = stats["inv_covs"]
    scores = np.zeros((H*W,), dtype=np.float32)
    for l in range(H*W):
        z = emb2[l]; mu = means[l]; inv = invs[l]; d = z - mu
        scores[l] = float(d @ inv @ d.T)
    amap = scores.reshape(H,W)
    return float(np.percentile(amap,95)), amap

# ---------------- Image enhancement helpers ----------------
def apply_clahe_bgr_lab(img_bgr, clip_limit=3.0, tile_grid_size=(8,8)):
    if img_bgr is None: return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def unsharp_mask(img, amount=0.6, radius=1.0):
    if img is None: return img
    k = max(3, int(radius*2)|1)
    blur = cv2.GaussianBlur(img, (k,k), sigmaX=radius)
    out = cv2.addWeighted(img, 1.0+amount, blur, -amount, 0)
    return np.clip(out,0,255).astype(np.uint8)

def build_gamma_lut(gamma):
    if gamma <= 0: gamma = 1.0
    inv = 1.0/gamma
    lut = np.array([((i/255.0)**inv)*255.0 for i in np.arange(256)]).astype("uint8")
    return lut

def capture_enhance(img, clahe_clip=3.0, alpha=1.0, beta=0.0, gamma=1.0, sharpen_amt=0.6, sharpen_rad=1.0):
    if img is None: return img
    img = apply_clahe_bgr_lab(img, clip_limit=clahe_clip)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    lut = build_gamma_lut(gamma)
    img = cv2.LUT(img, lut)
    img = unsharp_mask(img, amount=sharpen_amt, radius=sharpen_rad)
    return img

# ---------------- GUI App ----------------
class InspectorApp:
    def __init__(self, root):
        self.root = root; root.title("Inspector - Complete")
        # Camera
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", f"Cannot open camera index {CAM_INDEX}"); root.destroy(); return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Modbus client
        self.modbus = ModbusClient(method="rtu", port=SERIAL_PORT, baudrate=BAUDRATE, timeout=0.5)
        try: self.modbus.connect()
        except Exception as e: print("Modbus connect error:", e)

        # state
        self.frame = None; self.photo = None; self.running = True
        self.inspect_mode = False; self.auto_mode = False
        self.auto_target = 250; self.auto_count = 0
        self.roi = self.load_roi(); self.roi_selecting=False; self.roi_start=None; self.roi_current=None
        self.cam_lock = threading.Lock(); self.last_sensor=False

        # main solenoid refcount
        self.main_lock = threading.Lock(); self.main_hold_count = 0

        # event queue
        self.event_q = queue.Queue(maxsize=EVENT_QUEUE_MAX)

        # UI variables
        self.status_var = tk.StringVar(value="Idle"); self.bottle_var = tk.StringVar(value="Absent"); self.result_var = tk.StringVar(value="-")
        # enhancement vars
        self.alpha = tk.DoubleVar(value=1.0); self.beta = tk.DoubleVar(value=0.0); self.gamma = tk.DoubleVar(value=1.0)
        self.clahe = tk.DoubleVar(value=3.0); self.sharp_amt = tk.DoubleVar(value=0.6); self.sharp_rad = tk.DoubleVar(value=1.0)
        # delay/timing vars
        self.pre_act_ms = tk.IntVar(value=DEFAULT_PRE_ACT_MS)
        self.camera_delay_ms = tk.IntVar(value=DEFAULT_CAMERA_DELAY_MS)
        self.capture_offset_ms = tk.IntVar(value=DEFAULT_CAPTURE_OFFSET_MS)
        self.main_active_ms = tk.IntVar(value=DEFAULT_MAIN_ACTIVE_MS)
        self.reject_after_ms = tk.IntVar(value=DEFAULT_REJECT_AFTER_MS)
        self.reject_active_ms = tk.IntVar(value=DEFAULT_REJECT_ACTIVE_MS)
        # model/variant vars
        self.variant_choice = tk.StringVar(value=list(VARIANTS.keys())[0])
        self.loaded_model_label = tk.StringVar(value=os.path.basename(DEFAULT_MODEL))
        self.threshold_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)

        # build UI (canvas + controls)
        self.canvas = tk.Canvas(root, width=920, height=540, bg="black"); self.canvas.grid(row=0,column=0, padx=6, pady=6, columnspan=8)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down); self.canvas.bind("<B1-Motion>", self.on_mouse_drag); self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # top control row
        ttk.Button(root, text="Set ROI", command=self.enable_roi).grid(row=1,column=0, padx=2, pady=2)
        ttk.Button(root, text="Capture Good", command=self.capture_good).grid(row=1,column=1, padx=2)
        ttk.Button(root, text="Train", command=self.train_async).grid(row=1,column=2, padx=2)
        ttk.Button(root, text="Test Once", command=self.test_once).grid(row=1,column=3, padx=2)
        ttk.Button(root, text="Load Model File...", command=self.choose_model_file).grid(row=1,column=4, padx=2)
        ttk.Label(root, textvariable=self.loaded_model_label).grid(row=1,column=5, columnspan=3, sticky="w")

        # inspection controls row
        ttk.Button(root, text="Start Inspection", command=self.start_inspection).grid(row=2,column=0, padx=2, pady=4)
        ttk.Button(root, text="Stop Inspection", command=self.stop_inspection).grid(row=2,column=1, padx=2)
        ttk.Label(root, text="Auto N:").grid(row=2,column=2); self.auto_entry_var = tk.IntVar(value=250); ttk.Entry(root, textvariable=self.auto_entry_var, width=6).grid(row=2,column=3)
        ttk.Button(root, text="Start Auto N", command=self.start_auto).grid(row=2,column=4); ttk.Button(root, text="Stop Auto", command=self.stop_auto).grid(row=2,column=5)
        ttk.Label(root, text="Variant:").grid(row=2,column=6); ttk.OptionMenu(root, self.variant_choice, self.variant_choice.get(), *list(VARIANTS.keys())).grid(row=2,column=7)

        # enhancement sliders
        ttk.Label(root, text="Alpha").grid(row=3,column=0); tk.Scale(root, from_=0.5, to=3.0, resolution=0.05, orient="horizontal", variable=self.alpha, command=lambda v:self.on_enh()).grid(row=3,column=1,columnspan=2,sticky="ew")
        ttk.Label(root, text="Beta").grid(row=3,column=3); tk.Scale(root, from_=-80, to=80, resolution=1, orient="horizontal", variable=self.beta, command=lambda v:self.on_enh()).grid(row=3,column=4,columnspan=2,sticky="ew")
        ttk.Label(root, text="Gamma").grid(row=3,column=6); tk.Scale(root, from_=0.3, to=3.0, resolution=0.05, orient="horizontal", variable=self.gamma, command=lambda v:self.on_enh()).grid(row=3,column=7,columnspan=1,sticky="ew")

        ttk.Label(root, text="CLAHE").grid(row=4,column=0); tk.Scale(root, from_=1.0, to=8.0, resolution=0.1, orient="horizontal", variable=self.clahe, command=lambda v:self.on_enh()).grid(row=4,column=1,columnspan=2,sticky="ew")
        ttk.Label(root, text="Sharpen amount").grid(row=4,column=3); tk.Scale(root, from_=0.0, to=2.0, resolution=0.05, orient="horizontal", variable=self.sharp_amt, command=lambda v:self.on_enh()).grid(row=4,column=4,columnspan=2,sticky="ew")
        ttk.Label(root, text="Sharpen R").grid(row=4,column=6); tk.Scale(root, from_=0.5, to=3.0, resolution=0.05, orient="horizontal", variable=self.sharp_rad, command=lambda v:self.on_enh()).grid(row=4,column=7,columnspan=1,sticky="ew")

        # delays & timing fields (restored)
        ttk.Label(root, text="Pre-act (ms)").grid(row=5,column=0); ttk.Entry(root, textvariable=self.pre_act_ms, width=8).grid(row=5,column=1)
        ttk.Label(root, text="Camera delay (ms)").grid(row=5,column=2); ttk.Entry(root, textvariable=self.camera_delay_ms, width=8).grid(row=5,column=3)
        ttk.Label(root, text="Capture offset (ms)").grid(row=5,column=4); ttk.Entry(root, textvariable=self.capture_offset_ms, width=8).grid(row=5,column=5)
        ttk.Button(root, text="Apply Delays", command=self.apply_delays).grid(row=5,column=6)

        ttk.Label(root, text="Main active (ms)").grid(row=6,column=0); ttk.Entry(root, textvariable=self.main_active_ms, width=8).grid(row=6,column=1)
        ttk.Label(root, text="Reject after (ms)").grid(row=6,column=2); ttk.Entry(root, textvariable=self.reject_after_ms, width=8).grid(row=6,column=3)
        ttk.Label(root, text="Reject active (ms)").grid(row=6,column=4); ttk.Entry(root, textvariable=self.reject_active_ms, width=8).grid(row=6,column=5)
        ttk.Label(root, text="Threshold").grid(row=7,column=0); ttk.Entry(root, textvariable=self.threshold_var, width=8).grid(row=7,column=1)
        ttk.Label(root, textvariable=self.status_var).grid(row=7,column=2, columnspan=6, sticky="w")
        ttk.Label(root, text="Bottle").grid(row=8,column=0); ttk.Label(root, textvariable=self.bottle_var).grid(row=8,column=1)
        ttk.Label(root, text="Result").grid(row=8,column=2); ttk.Label(root, textvariable=self.result_var).grid(row=8,column=3)
        ttk.Button(root, text="Quit", command=self.on_quit).grid(row=8,column=7, sticky="ew")

        # internal
        self._gamma_lut = build_gamma_lut(self.gamma.get())
        self._last_preview_score = 0

        # start modbus poll & worker & preview update
        self.modbus_thread = threading.Thread(target=self.modbus_poll_loop, daemon=True)
        self.modbus_thread.start()
        self.worker_thread = threading.Thread(target=self.event_worker, daemon=True)
        self.worker_thread.start()
        self.root.after(PREVIEW_LOOP_MS, self.update_loop)

    # camera helpers
    def set_camera_exposure(self, val):
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(val))
            return True
        except Exception as e:
            print("set exposure err", e); return False

    def apply_camera_params(self):
        try:
            # placeholder if you add fields to change width/height/exposure
            pass
        except:
            pass

    def apply_delays(self):
        # ensure GUI values are consistent
        try:
            _ = int(self.capture_offset_ms.get()); _ = int(self.main_active_ms.get())
            _ = int(self.reject_after_ms.get()); _ = int(self.reject_active_ms.get())
            _ = int(self.pre_act_ms.get()); _ = int(self.camera_delay_ms.get())
            self.status_var.set("Delays applied")
        except Exception as e:
            self.status_var.set(f"Invalid delay input: {e}")

    # ROI functions
    def load_roi(self):
        try:
            if os.path.exists("roi_meta.json"):
                with open("roi_meta.json","r") as f:
                    return json.load(f)
        except:
            pass
        return None
    def save_roi(self, roi):
        try:
            with open("roi_meta.json","w") as f: json.dump(roi, f)
        except: pass
    def enable_roi(self):
        self.roi_selecting=True; self.roi_start=None; self.roi_current=None; self.status_var.set("ROI select - drag on preview")

    def on_mouse_down(self, e):
        if not self.roi_selecting or self.frame is None: return
        x,y = self.canvas_to_frame_coords(e.x, e.y); self.roi_start=(x,y); self.roi_current=(x,y)
    def on_mouse_drag(self, e):
        if not self.roi_selecting or self.roi_start is None: return
        x,y = self.canvas_to_frame_coords(e.x, e.y); self.roi_current=(x,y)
    def on_mouse_up(self, e):
        if not self.roi_selecting or self.roi_start is None: return
        x2,y2 = self.canvas_to_frame_coords(e.x, e.y); x1,y1 = self.roi_start
        x0,y0 = min(x1,x2), min(y1,y2); w,h = abs(x2-x1), abs(y2-y1)
        if w>5 and h>5:
            self.roi = {"x":int(x0), "y":int(y0), "w":int(w), "h":int(h)}
            self.save_roi(self.roi)
            self.status_var.set(f"ROI set {self.roi}")
        else:
            self.status_var.set("ROI too small")
        self.roi_selecting=False; self.roi_start=None; self.roi_current=None

    def canvas_to_frame_coords(self, cx, cy):
        if self.frame is None: return 0,0
        fh,fw = self.frame.shape[:2]; cw = int(self.canvas.winfo_width()); ch = int(self.canvas.winfo_height())
        scale = min((cw/fw) if fw>0 else 1.0, (ch/fh) if fh>0 else 1.0)
        disp_w = int(fw*scale); disp_h = int(fh*scale); off_x=(cw-disp_w)//2; off_y=(ch-disp_h)//2
        x = (cx-off_x)/scale; y = (cy-off_y)/scale
        x = max(0, min(fw-1, int(x))); y = max(0, min(fh-1, int(y)))
        return x,y

    def on_enh(self):
        try: self._gamma_lut = build_gamma_lut(float(self.gamma.get()))
        except: pass
        self.status_var.set("Enhancement updated")

    # capture helper
    def capture_frame(self):
        with self.cam_lock:
            ret, frame = self.cap.read()
            if not ret or frame is None: return None
            return frame.copy()

    # Modbus poll - immediate actuation on rising edge, enqueue event
    def modbus_poll_loop(self):
        while self.running:
            try:
                resp = None
                try:
                    resp = self.modbus.read_discrete_inputs(address=DISCRETE_INPUT_ADDR, count=1, unit=SLAVE_ID)
                except Exception:
                    resp = None
                current = bool(resp.bits[0]) if (resp and hasattr(resp,"bits")) else False
                self.bottle_var.set("Present" if current else "Absent")
                if current and not self.last_sensor and (self.inspect_mode or self.auto_mode):
                    # pre-act wait
                    pre = int(self.pre_act_ms.get()) if self.pre_act_ms.get() is not None else 0
                    if pre > 0:
                        time.sleep(pre/1000.0)
                    # immediate main solenoid ON (refcounted)
                    with self.main_lock:
                        self.main_hold_count += 1
                        if self.main_hold_count == 1:
                            try:
                                self.modbus.write_coil(address=COIL_ADDR, value=True, unit=SLAVE_ID)
                                self.status_var.set("Main solenoid ON")
                            except Exception as e:
                                print("Main on err:", e)
                    # enqueue event with detection timestamp and variant selection
                    ev = {"t_detect": time.time(), "variant": self.variant_choice.get()}
                    try:
                        self.event_q.put_nowait(ev)
                        self.status_var.set("Event queued")
                    except queue.Full:
                        print("Event queue full - dropping event")
                self.last_sensor = current
            except Exception as e:
                print("Modbus poll err:", e)
            time.sleep(MODBUS_POLL_INTERVAL)

    # event worker
    def event_worker(self):
        while self.running:
            try:
                ev = self.event_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                t_detect = ev.get("t_detect", time.time()); variant = ev.get("variant", self.variant_choice.get())
                # capture offset (ms)
                offset = int(self.capture_offset_ms.get()) if self.capture_offset_ms.get() is not None else DEFAULT_CAPTURE_OFFSET_MS
                cam_delay = int(self.camera_delay_ms.get()) if self.camera_delay_ms.get() is not None else DEFAULT_CAMERA_DELAY_MS
                if offset > 0:
                    time.sleep(offset/1000.0)
                # synchronized capture
                frame = self.capture_frame()
                if frame is None:
                    self.status_var.set("Capture failed")
                else:
                    # enhance & crop
                    frame_cap = capture_enhance(frame.copy(), clahe_clip=float(self.clahe.get()), alpha=float(self.alpha.get()), beta=float(self.beta.get()), gamma=float(self.gamma.get()), sharpen_amt=float(self.sharp_amt.get()), sharpen_rad=float(self.sharp_rad.get()))
                    if self.roi:
                        x,y,w,h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
                        crop = frame_cap[y:y+h, x:x+w].copy()
                    else:
                        crop = frame_cap.copy()
                    # if auto_mode: save to variant folder
                    if self.auto_mode:
                        save_dir = VARIANTS.get(variant, {})["data"]
                        if save_dir:
                            os.makedirs(save_dir, exist_ok=True)
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            path = os.path.join(save_dir, f"auto_{ts}.png"); cv2.imwrite(path, crop)
                            self.auto_count += 1; self.status_var.set(f"Auto {self.auto_count}/{self.auto_target}")
                            if self.auto_count >= self.auto_target:
                                self.auto_mode = False; self.status_var.set("Auto finished")
                    # scoring using model for selected variant
                    model_path = VARIANTS.get(variant, {}).get("model", DEFAULT_MODEL)
                    ok = load_padim(model_path)
                    if not ok:
                        self.status_var.set("Model load failed")
                    else:
                        start = time.time()
                        score, amap = score_anomaly(crop)
                        dur_ms = (time.time()-start)*1000.0
                        if score is None:
                            self.result_var.set("Model error")
                        else:
                            thr = float(self.threshold_var.get())
                            decision = "OK" if score < thr else "NG"
                            self.result_var.set(f"{decision} ({score:.2f})"); self.status_var.set(f"Score {dur_ms:.1f} ms")
                            # if NG -> pulse reject after configured delay
                            if decision == "NG":
                                after = int(self.reject_after_ms.get()) if self.reject_after_ms.get() is not None else DEFAULT_REJECT_AFTER_MS
                                active = int(self.reject_active_ms.get()) if self.reject_active_ms.get() is not None else DEFAULT_REJECT_ACTIVE_MS
                                def reject_pulse():
                                    if after>0: time.sleep(after/1000.0)
                                    try:
                                        self.modbus.write_coil(address=REJECT_COIL_ADDR, value=True, unit=SLAVE_ID)
                                        time.sleep(active/1000.0)
                                        self.modbus.write_coil(address=REJECT_COIL_ADDR, value=False, unit=SLAVE_ID)
                                        self.status_var.set("Reject pulse done")
                                    except Exception as e:
                                        print("Reject write err:", e)
                                threading.Thread(target=reject_pulse, daemon=True).start()
                # hold main solenoid for per-event main_active ms relative to detection
                main_active = int(self.main_active_ms.get()) if self.main_active_ms.get() is not None else DEFAULT_MAIN_ACTIVE_MS
                elapsed = int((time.time() - t_detect) * 1000.0); rem = max(0, main_active - elapsed)
                if rem > 0: time.sleep(rem/1000.0)
                # release refcount and physically release if no more holds
                with self.main_lock:
                    if self.main_hold_count > 0: self.main_hold_count -= 1
                    if self.main_hold_count == 0:
                        try:
                            self.modbus.write_coil(address=COIL_ADDR, value=False, unit=SLAVE_ID); self.status_var.set("Main solenoid OFF")
                        except Exception as e:
                            print("Main off err:", e)
                    else:
                        self.status_var.set(f"Main held by others ({self.main_hold_count})")
            except Exception as e:
                print("Event worker err:", e); traceback.print_exc()
            finally:
                try: self.event_q.task_done()
                except: pass

    # training and model handling
    def train_async(self):
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        global channel_idx, stats
        try:
            variant = self.variant_choice.get()
            data_dir = VARIANTS.get(variant, {}).get("data")
            model_file = VARIANTS.get(variant, {}).get("model")
            if not data_dir or not model_file:
                self.status_var.set("Variant paths invalid"); return
            img_paths = sorted(glob.glob(os.path.join(data_dir,"*.png"))+glob.glob(os.path.join(data_dir,"*.jpg")))
            if len(img_paths) < 20:
                self.status_var.set(f"Need >=20 imgs in {data_dir} (found {len(img_paths)})"); return
            embs = []
            with torch.no_grad():
                for p in img_paths:
                    im = cv2.imread(p)
                    if im is None: continue
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    t = to_tensor(im).unsqueeze(0).to(device)
                    feats = feat_net(t); emb = embedding_concat(feats); embs.append(emb.cpu())
            if len(embs)==0:
                self.status_var.set("No valid images"); return
            embs = torch.cat(embs, dim=0)
            C = embs.shape[1]
            channel_idx = random_channel_indices(C, EMBED_DIMS, seed=42)
            embs = embs[:, channel_idx, :, :]
            means, invs, H, W = compute_gaussian_stats(embs)
            np.savez(model_file, means=means, inv_covs=invs, H=H, W=W, channel_idx=channel_idx)
            stats = {"means":means, "inv_covs":invs, "H":H, "W":W}
            self.loaded_model_label.set(os.path.basename(model_file))
            self.status_var.set(f"Trained {variant} -> {model_file}")
        except Exception as e:
            traceback.print_exc(); self.status_var.set(f"Train error: {e}")

    def choose_model_file(self):
        p = filedialog.askopenfilename(title="Select .npz model", filetypes=[("NPZ","*.npz"),("All","*.*")])
        if not p: return
        ok = load_padim(p)
        if ok: self.loaded_model_label.set(os.path.basename(p)); self.status_var.set(f"Loaded {p}")
        else: messagebox.showerror("Model", "Failed to load model")

    def test_once(self):
        if self.frame is None or self.roi is None:
            self.status_var.set("Set ROI first"); return
        with self.cam_lock:
            crop = self.frame[self.roi["y"]:self.roi["y"]+self.roi["h"], self.roi["x"]:self.roi["x"]+self.roi["w"]].copy()
        crop = capture_enhance(crop, clahe_clip=float(self.clahe.get()), alpha=float(self.alpha.get()), beta=float(self.beta.get()), gamma=float(self.gamma.get()), sharpen_amt=float(self.sharp_amt.get()), sharpen_rad=float(self.sharp_rad.get()))
        model_path = VARIANTS.get(self.variant_choice.get(), {}).get("model", DEFAULT_MODEL)
        if not load_padim(model_path):
            self.status_var.set("Model not loaded"); return
        start = time.time(); score, amap = score_anomaly(crop); dur=(time.time()-start)*1000.0
        if score is None: self.status_var.set("Score failed"); return
        thr = float(self.threshold_var.get()); decision = "OK" if score < thr else "NG"
        self.result_var.set(f"{decision} ({score:.2f})"); self.status_var.set(f"Test score {dur:.1f} ms")

    # auto capture controls
    def start_auto(self):
        try:
            n = int(self.auto_entry_var.get()); self.auto_target = n; self.auto_count = 0; self.auto_mode = True; self.status_var.set(f"Auto started ({n})")
        except Exception:
            self.status_var.set("Invalid auto N")

    def stop_auto(self):
        self.auto_mode = False; self.status_var.set("Auto stopped")

    # preview loop (lightweight)
    def update_loop(self):
        if not self.running: return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            with self.cam_lock:
                self.frame = frame.copy()
            disp = frame.copy()
            if self.roi:
                x,y,w,h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
                cv2.rectangle(disp, (x,y), (x+w, y+h), (0,255,255), 2)
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            fh,fw = disp_rgb.shape[:2]; cw = int(self.canvas.winfo_width()); ch = int(self.canvas.winfo_height())
            if fw>0 and fh>0 and cw>0 and ch>0:
                scale = min(cw/fw, ch/fh); new_w = max(1,int(fw*scale)); new_h = max(1,int(fh*scale))
                disp_rgb = cv2.resize(disp_rgb, (new_w, new_h))
                img = Image.fromarray(disp_rgb); self.photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all"); self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor="center")
        self.root.after(PREVIEW_LOOP_MS, self.update_loop)

    def capture_good(self):
        if self.frame is None or self.roi is None:
            self.status_var.set("Set ROI & camera"); return
        x,y,w,h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
        with self.cam_lock:
            crop = self.frame[y:y+h, x:x+w].copy()
        if crop is None or crop.size==0: self.status_var.set("Empty crop"); return
        variant = self.variant_choice.get(); save_dir = VARIANTS.get(variant, {}).get("data")
        if not save_dir: self.status_var.set("Invalid variant data dir"); return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f"); path = os.path.join(save_dir, f"good_{ts}.png"); cv2.imwrite(path, crop); self.status_var.set(f"Saved {path}")

    def start_inspection(self):
        self.inspect_mode = True; self.status_var.set("Inspection ON")

    def stop_inspection(self):
        self.inspect_mode = False; self.status_var.set("Inspection OFF"); self.result_var.set("-")

    def on_quit(self):
        self.running = False
        try: self.modbus.close()
        except: pass
        self.root.after(100, self.cleanup)

    def cleanup(self):
        try: self.cap.release()
        except: pass
        self.root.destroy()