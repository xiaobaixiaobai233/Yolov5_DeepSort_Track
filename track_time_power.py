# ==============================================
# PyTorch 2.6+ Weight Safe Loading Patch - Top of File!
# ==============================================
import torch.serialization
import numpy as np
from numpy import dtype

torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    dtype,
])

# ==================== PyTorch Upsample Compatibility Patch ====================
import torch
import torch.nn as nn
import torch.nn.functional as F

def patched_upsample_forward(self, input):
    return nn.functional.interpolate(
        input, size=self.size, scale_factor=self.scale_factor,
        mode=self.mode, align_corners=self.align_corners
    )

nn.Upsample.forward = patched_upsample_forward


# ==================== Perception Module ====================
class PerceptionModule:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def compute_weights(self, person_norm, accel_norm, power_norm):
        imp_person = person_norm ** 1.6
        imp_accel  = accel_norm ** 1.45
        imp_power  = power_norm ** 1.25

        importance = torch.tensor([imp_person, imp_accel, imp_power], dtype=torch.float32)
        weights = F.softmax(importance / self.temperature, dim=0)

        return weights[0].item(), weights[1].item(), weights[2].item()


# ==================== Main Code ====================
import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch.backends.cudnn as cudnn
import numpy as np
import csv
import threading
from datetime import datetime
import math
import pandas as pd

# ---------------------- Power Consumption Monitoring Module ----------------------
import subprocess
import re

class PowerMonitor:
    def __init__(self, csv_path="power_trace.csv", sample_interval=0.5):
        self.csv_path = csv_path
        self.sample_interval = sample_interval
        self.running = False
        self.thread = None
        self.current_frame = -1
        self.process = None
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "total_power_mw", "gpu_cpu_power_mw", "frame_idx"])

    def start(self):
        self.running = True
        interval_ms = int(self.sample_interval * 1000)
        cmd = ["sudo", "tegrastats", "--interval", str(interval_ms)]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, universal_newlines=True)
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread:
            self.thread.join()

    def _monitor(self):
        power_pattern = re.compile(r'VDD_IN\s*(\d+)mW/\d+mW.*?VDD_CPU_GPU_CV\s*(\d+)mW/\d+mW')
        for line in iter(self.process.stdout.readline, ''):
            if not self.running:
                break
            match = power_pattern.search(line)
            if match:
                total_power = int(match.group(1))
                gpu_cpu_power = int(match.group(2))
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, total_power, gpu_cpu_power, self.current_frame])


# ---------------------- Frame Statistics Recorder ----------------------
class FrameStatistics:
    def __init__(self, csv_path="frame_statistics.csv"):
        self.csv_path = csv_path
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_idx", "timestamp", "model_size",
                "person_count", "avg_acceleration", "power_mw", "score",
                "w1_dynamic", "w2_dynamic", "w3_dynamic"
            ])

    def record(self, frame_idx, model_size, person_count, avg_acceleration, power_mw,
               score, w1, w2, w3):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_idx, timestamp, model_size, person_count,
                f"{avg_acceleration:.3f}", f"{power_mw:.0f}", f"{score:.4f}",
                f"{w1:.4f}", f"{w2:.4f}", f"{w3:.4f}"
            ])


def get_latest_power(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if not df.empty:
            return float(df['gpu_cpu_power_mw'].iloc[-1])
    except:
        pass
    return 0.0


def load_yolov5_model(size='s', device=None, half=False):
    if device is None:
        raise ValueError("Device parameter must be provided")
    weights_path = f'yolov5/weights/yolov5{size}.pt'
    if not os.path.exists(weights_path):
        print(f"ERROR: Weight file not found {weights_path}")
        return None
    print(f"Loading model: yolov5{size}")
    model = torch.load(weights_path, map_location=device, weights_only=False)['model'].float()
    model.to(device).eval()
    if half:
        model.half()
    return model


# ---------------------- Visualization Functions ----------------------
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    return x_c, y_c, bbox_w, bbox_h

def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]; x2 += offset[0]
        y1 += offset[1]; y2 += offset[1]
        id_ = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id_)
        label = f'{id_}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def xyxy2tlwh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


# ====================== Normalization Configuration ======================
NORM_CONFIG = {
    'person_max': 25.0,
    'accel_max': 22000.0,
    'power_max': 4500.0
}

def normalize(value, max_val):
    if max_val == 0:
        return 0.0
    return max(0.0, min(1.0, value / max_val))


# ---------------------- Main Detection Function ----------------------
def detect(opt, save_img=False):
    power_monitor = PowerMonitor(csv_path="power_trace.csv", sample_interval=0.5)
    power_monitor.start()

    stats_recorder = FrameStatistics(csv_path="frame_statistics.csv")

    accel_csv_path = "accelsum.csv"
    with open(accel_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp", "accel_sum_pixel_per_s2"])

    # ==================== Fix Unpacking Error ====================
    source = opt.source
    view_img = opt.view_img
    save_txt = opt.save_txt
    imgsz = opt.img_size

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    device = select_device(opt.device)
    half = device.type != 'cpu'

    current_model_size = 's'
    model = load_yolov5_model(current_model_size, device=device, half=half)
    if model is None:
        power_monitor.stop()
        return

    if webcam:
        view_img = False
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
        fps = 30.0
    else:
        view_img = False
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
        fps = dataset.cap.get(cv2.CAP_PROP_FPS) if hasattr(dataset, 'cap') else 30.0

    dt = 1.0 / fps
    names = model.module.names if hasattr(model, 'module') else model.names

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    if os.path.exists(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)

    txt_path = str(Path(opt.output) / 'results.txt')

    dict_box = dict()
    vid_path = None
    vid_writer = None
    frame_check_counter = 0

    last_person_count = 0
    last_sum_accel = 0.0
    last_power = 0.0

    # Initialize perception module
    perception = PerceptionModule(temperature=1.0)

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        power_monitor.current_frame = frame_idx

        # Adaptive model selection every 5 frames
        if frame_check_counter % 5 == 0:
            current_power = get_latest_power(power_monitor.csv_path)
            avg_accel = last_sum_accel / last_person_count if last_person_count > 0 else 0.0

            person_norm = normalize(last_person_count, NORM_CONFIG['person_max'])
            accel_norm  = normalize(avg_accel, NORM_CONFIG['accel_max'])
            power_norm  = normalize(current_power, NORM_CONFIG['power_max'])

            # Compute dynamic weights using perception module
            w1, w2, w3 = perception.compute_weights(person_norm, accel_norm, power_norm)

            score = w1 * person_norm + w2 * accel_norm + w3 * power_norm

            print(f"\n=== Perception Module Decision === Frame {frame_idx}")
            print(f" Person count: {last_person_count:3d} → {person_norm:.4f}")
            print(f" Acceleration: {avg_accel:8.1f} → {accel_norm:.4f}")
            print(f" Power usage : {current_power:5.0f} mW → {power_norm:.4f}")
            print(f" Dynamic wts : w1={w1:.3f} | w2={w2:.3f} | w3={w3:.3f}")
            print(f" Weight score: {score:.4f}")

            # Select model size based on complexity score
            if score < 0.35:
                target_size = 's'
            elif score < 0.55:
                target_size = 'm'
            elif score < 0.8:
                target_size = 'l'
            else:
                target_size = 'x'

            print(f" Selected model: Switch to yolov5{target_size}")

            if target_size != current_model_size:
                new_model = load_yolov5_model(target_size, device=device, half=half)
                if new_model is not None:
                    model = new_model
                    current_model_size = target_size
                    print(f"✓ Successfully switched to yolov5{target_size}")
                else:
                    print(f"⚠ Switch failed, continue using yolov5{current_model_size}")
            else:
                print(f" Keep current model yolov5{current_model_size}")

        frame_check_counter += 1

        # ==================== Inference Pipeline ====================
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        person_count = 0
        for det in pred:
            if det is not None:
                person_count += len(det)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0 = path[i], f'%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]
            save_path_img = str(Path(opt.output) / Path(p).name)

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                bbox_xywh = []
                confs = []
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    bbox_xywh.append([x_c, y_c, bbox_w, bbox_h])
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                outputs = deepsort.update(xywhs, confss, im0)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                    box_xywh = xyxy2tlwh(bbox_xyxy)
                    for j in range(len(box_xywh)):
                        x_center = box_xywh[j][0] + box_xywh[j][2] / 2
                        y_center = box_xywh[j][1] + box_xywh[j][3] / 2
                        id_ = outputs[j][-1]
                        dict_box.setdefault(id_, []).append([x_center, y_center])

                    # Draw trajectory lines
                    if frame_idx > 2:
                        max_trajectory_length = 50
                        temp_dict_box = dict_box.copy()
                        for key, value in temp_dict_box.items():
                            if key in identities and len(value) > max_trajectory_length:
                                value = value[-max_trajectory_length:]
                            for a in range(len(value) - 1):
                                color = compute_color_for_labels(key)
                                pt1 = tuple(map(int, value[a]))
                                pt2 = tuple(map(int, value[a + 1]))
                                cv2.line(im0, pt1, pt2, color, thickness=5, lineType=8)

                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left, bbox_top, bbox_w, bbox_h = output[:4]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
            else:
                deepsort.increment_ages()

            # Calculate total acceleration of all tracked persons
            sum_accel = 0.0
            if 'outputs' in locals() and len(outputs) > 0 and 'identities' in locals():
                for id_ in identities:
                    trajectory = dict_box.get(id_, [])
                    if len(trajectory) >= 3:
                        p1 = trajectory[-3]
                        p2 = trajectory[-2]
                        p3 = trajectory[-1]
                        vx1 = (p2[0] - p1[0]) / dt
                        vy1 = (p2[1] - p1[1]) / dt
                        vx2 = (p3[0] - p2[0]) / dt
                        vy2 = (p3[1] - p2[1]) / dt
                        ax = (vx2 - vx1) / dt
                        ay = (vy2 - vy1) / dt
                        a = math.sqrt(ax**2 + ay**2)
                        sum_accel += a

            avg_acceleration = sum_accel / person_count if person_count > 0 else 0.0

            current_power = get_latest_power(power_monitor.csv_path)
            w1, w2, w3 = perception.compute_weights(
                normalize(last_person_count, NORM_CONFIG['person_max']),
                normalize(avg_acceleration, NORM_CONFIG['accel_max']),
                normalize(current_power, NORM_CONFIG['power_max'])
            )

            # Record frame statistics
            stats_recorder.record(
                frame_idx=frame_idx,
                model_size=current_model_size,
                person_count=person_count,
                avg_acceleration=avg_acceleration,
                power_mw=current_power,
                score=score if 'score' in locals() else 0.0,
                w1=w1, w2=w2, w3=w3
            )

            # Update state variables for next frame
            last_person_count = person_count
            last_sum_accel = sum_accel
            last_power = current_power

            # Save acceleration data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(accel_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([frame_idx, timestamp, f"{sum_accel:.3f}"])

            print('%sDone. (%.3fs) Person:%d Accel:%.3f' % (s, t2 - t1, person_count, sum_accel))

            # Save output image or video
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path_img, im0)
                else:
                    if vid_path != save_path_img:
                        vid_path = save_path_img
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        fps_write = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else fps
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap else im0.shape[1]
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else im0.shape[0]
                        vid_writer = cv2.VideoWriter(save_path_img, cv2.VideoWriter_fourcc(*opt.fourcc), fps_write, (w, h))
                    vid_writer.write(im0)

    power_monitor.stop()

    print("\n" + "="*70)
    print("Processing completed! Perception Module enabled")
    print("Statistics saved to: frame_statistics.csv")
    print("="*70)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./MOT01.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
