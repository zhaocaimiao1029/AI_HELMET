import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import time
import pathlib
import numpy as np

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

# Linux / 树莓派兼容 Windows 训练出的路径对象
pathlib.WindowsPath = pathlib.PosixPath

try:
    import onnxruntime as ort
except Exception:
    ort = None


# =====================
# 基础路径
# =====================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PERSON_ONNX = os.path.join(BASE_DIR, "yolov5n_int8.onnx")
HELMET_ONNX = os.path.join(BASE_DIR, "helmetv2_int8.onnx")


# =====================
# 参数（稳定优先）
# =====================
SOURCE = 0

CAMERA_WIDTH = 512
CAMERA_HEIGHT = 384

PERSON_INFER_SIZE = 256
HELMET_INFER_SIZE = 320

PERSON_CONF = 0.40
HELMET_CONF = 0.10
NMS_IOU = 0.45

MAX_DET_PERSON = 3
MAX_DET_HELMET = 5

HEAD_REGION_RATIO = 0.60
# 只收紧左右，不动上下
HEAD_REGION_X_SHRINK = 0.12
HEAD_CENTER_X_MARGIN = 0.22
HEAD_CENTER_Y_MARGIN_TOP = 0.05
HEAD_CENTER_Y_MARGIN_BOTTOM = 0.64
HEAD_REGION_Y_EXTRA = 0.04
EDGE_TOUCH_PX = 20
EDGE_X_SHRINK_RELAX = 0.07
EDGE_CENTER_MARGIN_EXTRA = 0.075
SIDE_RATIO_HARD_LIMIT = 0.1675
SIDE_RATIO_EDGE_LIMIT = 0.215
HELMET_BOTTOM_MIN_RATIO = 0.06
HELMET_TOP_GAP_MAX_RATIO = 0.075
DRAW_HELMET_BOX = True
DRAW_HEAD_BOX = True
DRAW_MATCH_LINE = True
HELMET_CENTER_MIN_RATIO = 0.05
HELMET_CENTER_TO_HEADBOX_MAX_EXTRA = 0.05
HELMET_BOTTOM_TO_HEADBOX_MAX_EXTRA = 0.08
TOP_HELMET_SIDE_RELAX = 0.055
TOP_HELMET_RELAX_CENTER_MAX_RATIO = 0.24
TOP_HELMET_RELAX_BOTTOM_MAX_RATIO = 0.30
NOHELMET_HEAD_OVERLAP_MIN = 0.16
NOHELMET_HEAD_CENTER_Y_MAX_RATIO = 0.34
NOHELMET_SUPPRESS_CONF = 0.20

# 稳定优先：人体更慢，头盔稍快
PERSON_INFER_INTERVAL_SEC = 0.70
HELMET_INFER_INTERVAL_SEC = 0.25

SHOW_LOCAL_WINDOW = os.environ.get("SHOW_LOCAL_WINDOW", "1").strip() == "1"
PRINT_STATUS_INTERVAL_SEC = 1.0


# =====================
# 摄像头（树莓派）
# =====================
class PiCameraCapture:
    def __init__(self, size=(512, 384)):
        if Picamera2 is None:
            raise RuntimeError("picamera2 未安装")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)

    def isOpened(self):
        return True

    def read(self):
        try:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None
            return True, frame
        except Exception:
            return False, None

    def release(self):
        try:
            self.picam2.stop()
        except Exception:
            pass


def open_camera(source=0):
    print(f"👉 正在打开摄像头 source={source} ...", flush=True)

    if source == 0:
        try:
            cap = PiCameraCapture(size=(CAMERA_WIDTH, CAMERA_HEIGHT))
            print(f"✅ Picamera2 摄像头打开成功: {CAMERA_WIDTH}x{CAMERA_HEIGHT}", flush=True)
            return cap
        except Exception as e:
            print(f"⚠ Picamera2 打开失败：{e}", flush=True)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        return None
    return cap


# =====================
# ONNX YOLOv5 推理
# =====================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class YOLOv5ONNX:
    def __init__(self, model_path, input_size=640, conf_thres=0.25, iou_thres=0.45, max_det=100, class_names=None):
        if ort is None:
            raise RuntimeError("onnxruntime 未安装")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        self.model_path = model_path
        self.input_size = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.class_names = class_names or []

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, image):
        img, ratio, dwdh = letterbox(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, ratio, dwdh

    def infer(self, image):
        inp, ratio, dwdh = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: inp})
        pred = outputs[0]
        return self.postprocess(pred, image.shape[:2], ratio, dwdh)

    def postprocess(self, pred, orig_shape, ratio, dwdh):
        h0, w0 = orig_shape
        pred = np.squeeze(pred, axis=0)

        boxes = []
        confidences = []
        class_ids = []

        if pred.ndim != 2 or pred.shape[1] < 6:
            return []

        for det in pred:
            obj_conf = float(det[4])
            if obj_conf < 1e-6:
                continue

            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cls_id])
            conf = obj_conf * cls_conf

            if conf < self.conf_thres:
                continue

            x, y, w, h = det[:4]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            dw, dh = dwdh
            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio

            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))

            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

            boxes.append([float(x1), float(y1), float(bw), float(bh)])
            confidences.append(conf)
            class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)
        if len(indices) == 0:
            return []

        results = []
        for idx in indices.flatten()[:self.max_det]:
            x, y, w, h = boxes[idx]
            results.append({
                "box": [x, y, x + w, y + h],
                "conf": float(confidences[idx]),
                "cls": int(class_ids[idx]),
                "name": self.class_names[class_ids[idx]] if 0 <= class_ids[idx] < len(self.class_names) else str(class_ids[idx])
            })
        return results


def filter_by_name(dets, target_name):
    out = []
    for d in dets:
        if d["name"] == target_name:
            x1, y1, x2, y2 = d["box"]
            out.append([x1, y1, x2, y2, d["conf"]])
    return out


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def iou(box1, box2):
    inter = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - inter
    return inter / union if union > 0 else 0.0


def head_region(person_box, head_ratio=0.70):
    px1, py1, px2, py2 = person_box[:4]
    pw = px2 - px1
    ph = py2 - py1

    shrink_left = HEAD_REGION_X_SHRINK
    shrink_right = HEAD_REGION_X_SHRINK
    if px1 <= EDGE_TOUCH_PX:
        shrink_left = max(0.04, HEAD_REGION_X_SHRINK - EDGE_X_SHRINK_RELAX)
    if px2 >= CAMERA_WIDTH - EDGE_TOUCH_PX:
        shrink_right = max(0.04, HEAD_REGION_X_SHRINK - EDGE_X_SHRINK_RELAX)

    head_x1 = px1 + pw * shrink_left
    head_x2 = px2 - pw * shrink_right
    if head_x2 <= head_x1:
        head_x1 = px1 + pw * 0.05
        head_x2 = px2 - pw * 0.05

    head_y1 = py1
    head_y2 = min(py2, py1 + ph * head_ratio + ph * HEAD_REGION_Y_EXTRA)
    return [head_x1, head_y1, head_x2, head_y2]


def head_face_region(person_box):
    px1, py1, px2, py2 = person_box[:4]
    pw = px2 - px1
    ph = py2 - py1
    return [
        px1 + pw * 0.20,
        py1 + ph * 0.04,
        px2 - pw * 0.20,
        py1 + ph * 0.34,
    ]


def nohelmet_conflicts(person_box, nohelmets):
    if not nohelmets:
        return None

    face_box = head_face_region(person_box)
    px1, py1, px2, py2 = person_box[:4]
    ph = py2 - py1
    fx1, fy1, fx2, fy2 = face_box
    best = None

    for nb in nohelmets:
        conf = nb[4] if len(nb) >= 5 else 0.0
        if conf < NOHELMET_SUPPRESS_CONF:
            continue
        box = nb[:4]
        inter = intersection_area(face_box, box)
        area = box_area(box)
        if area <= 0:
            continue
        overlap = inter / area
        ncx = (box[0] + box[2]) / 2.0
        ncy = (box[1] + box[3]) / 2.0
        if overlap >= NOHELMET_HEAD_OVERLAP_MIN and fx1 <= ncx <= fx2 and ncy <= py1 + ph * NOHELMET_HEAD_CENTER_Y_MAX_RATIO:
            item = {"box": box, "conf": conf, "overlap": overlap, "cx": ncx, "cy": ncy}
            if best is None or conf * overlap > best["conf"] * best["overlap"]:
                best = item
    return best


def helmet_person_score(person_box, helmet_box, nohelmets=None):
    head_box = head_region(person_box, head_ratio=HEAD_REGION_RATIO)
    helmet_box = helmet_box[:4]
    inter = intersection_area(head_box, helmet_box)
    if inter <= 0.0:
        return 0.0

    helmet_area = box_area(helmet_box)
    if helmet_area <= 0.0:
        return 0.0

    overlap_ratio = inter / helmet_area
    head_iou = iou(head_box, helmet_box)

    hcx = (helmet_box[0] + helmet_box[2]) / 2.0
    hcy = (helmet_box[1] + helmet_box[3]) / 2.0
    htop = helmet_box[1]
    hbot = helmet_box[3]

    px1, py1, px2, py2 = person_box[:4]
    pw = px2 - px1
    ph = py2 - py1
    pcx = (px1 + px2) / 2.0
    hx1, hy1_box, hx2, hy2_box = head_box
    head_cx = (hx1 + hx2) / 2.0

    near_left = px1 <= EDGE_TOUCH_PX
    near_right = px2 >= CAMERA_WIDTH - EDGE_TOUCH_PX
    near_image_edge = near_left or near_right

    top_gap_ratio = max(0.0, (htop - py1) / max(1.0, ph))
    top_gap_limit = HELMET_TOP_GAP_MAX_RATIO + (0.015 if near_image_edge else 0.0)
    if top_gap_ratio > top_gap_limit:
        return 0.0

    left_margin = pw * HEAD_CENTER_X_MARGIN
    right_margin = pw * HEAD_CENTER_X_MARGIN
    if near_left:
        left_margin += pw * EDGE_CENTER_MARGIN_EXTRA
    if near_right:
        right_margin += pw * EDGE_CENTER_MARGIN_EXTRA

    # 倾头/侧脸时更依赖头部区域中心，而不是整个人体框中心
    center_ref = head_cx if near_image_edge else (pcx * 0.40 + head_cx * 0.60)

    # 顶部附近的 helmet 给一点横向容忍，避免手臂外展或倾头导致误杀
    if hcy <= py1 + ph * TOP_HELMET_RELAX_CENTER_MAX_RATIO and hbot <= py1 + ph * TOP_HELMET_RELAX_BOTTOM_MAX_RATIO:
        left_margin += pw * TOP_HELMET_SIDE_RELAX
        right_margin += pw * TOP_HELMET_SIDE_RELAX

    if not (center_ref - left_margin <= hcx <= center_ref + right_margin):
        return 0.0

    if not (py1 - ph * HEAD_CENTER_Y_MARGIN_TOP <= hcy <= py1 + ph * HEAD_CENTER_Y_MARGIN_BOTTOM):
        return 0.0
    if hcy < py1 + ph * HELMET_CENTER_MIN_RATIO:
        return 0.0
    if hbot < py1 + ph * HELMET_BOTTOM_MIN_RATIO:
        return 0.0
    if hcy > hy2_box + ph * HELMET_CENTER_TO_HEADBOX_MAX_EXTRA:
        return 0.0
    if hbot > hy2_box + ph * HELMET_BOTTOM_TO_HEADBOX_MAX_EXTRA:
        return 0.0

    dx = abs(hcx - center_ref)
    center_span = max(left_margin, right_margin, 1.0)
    side_ratio = dx / max(1.0, pw)

    if side_ratio > SIDE_RATIO_EDGE_LIMIT and near_image_edge:
        return 0.0
    if side_ratio > SIDE_RATIO_HARD_LIMIT and not near_image_edge:
        return 0.0

    conflict = nohelmet_conflicts(person_box, nohelmets or [])

    # 有 no_helmet 冲突时，不再一刀切强力抑制；
    # 对真正覆盖头部较好的 helmet 只轻度降权，减少倾头戴帽误判。
    suppress = 1.0
    if conflict is not None:
        strong_helmet_cover = (overlap_ratio >= 0.30 or head_iou >= 0.14 or hbot >= py1 + ph * 0.18)
        suppress = 0.72 if strong_helmet_cover else 0.22

    center_bonus = max(0.0, 1.0 - dx / center_span)
    vertical_bonus = max(0.0, 1.0 - abs(hcy - (py1 + ph * 0.23)) / max(1.0, ph * 0.30))

    tilt_cover_bonus = 0.0
    if overlap_ratio >= 0.26 and head_iou >= 0.10 and hbot >= py1 + ph * 0.16:
        tilt_cover_bonus = 0.06

    return (max(overlap_ratio, head_iou * 1.5) + center_bonus * 0.30 + vertical_bonus * 0.12 + tilt_cover_bonus) * suppress


def assign_helmets_to_persons(persons, helmets, nohelmets=None):
    if not persons or not helmets:
        return [-1] * len(persons)

    matches = []
    head_centers = []
    for person in persons:
        hx1, hy1, hx2, hy2 = head_region(person, head_ratio=HEAD_REGION_RATIO)
        head_centers.append((hx1 + hx2) / 2.0)

    for pi, person in enumerate(persons):
        for hi, helmet in enumerate(helmets):
            score = helmet_person_score(person, helmet, nohelmets=nohelmets)
            if score <= 0.0:
                continue

            hcx = (helmet[0] + helmet[2]) / 2.0
            this_dist = abs(hcx - head_centers[pi])
            min_dist = min(abs(hcx - cx) for cx in head_centers) if head_centers else this_dist
            if this_dist > min_dist + 1.0:
                score *= 0.55

            matches.append((score, pi, hi))

    matches.sort(reverse=True, key=lambda x: x[0])
    assigned_person = [-1] * len(persons)
    assigned_helmet = [False] * len(helmets)

    for score, pi, hi in matches:
        if assigned_person[pi] == -1 and not assigned_helmet[hi]:
            assigned_person[pi] = hi
            assigned_helmet[hi] = True

    return assigned_person


def draw_overlay(frame, persons, helmets, nohelmets=None, helmet_debug_dets=None):
    helmet_person = 0
    no_helmet_person = 0

    assigned = assign_helmets_to_persons(persons, helmets, nohelmets=nohelmets)
    for idx, (px1, py1, px2, py2, pconf) in enumerate(persons):
        has_helmet = (idx < len(assigned) and assigned[idx] >= 0)

        if has_helmet:
            helmet_person += 1
            color = (0, 255, 0)
            label = "HELMET"
        else:
            no_helmet_person += 1
            color = (0, 0, 255)
            label = "NO HELMET"

        cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), color, 2)
        cv2.putText(frame, label, (int(px1), max(0, int(py1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)

        if DRAW_HEAD_BOX:
            hx1, hy1, hx2, hy2 = head_region([px1, py1, px2, py2], head_ratio=HEAD_REGION_RATIO)
            cv2.rectangle(frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 0, 255), 1)

        if DRAW_MATCH_LINE and has_helmet:
            hi = assigned[idx]
            if 0 <= hi < len(helmets):
                hb = helmets[hi][:4]
                pcx = int((px1 + px2) / 2.0)
                pcy = int(py1 + (py2 - py1) * 0.18)
                hcx = int((hb[0] + hb[2]) / 2.0)
                hcy = int((hb[1] + hb[3]) / 2.0)
                cv2.line(frame, (pcx, pcy), (hcx, hcy), (0, 255, 255), 1)

    if DRAW_HELMET_BOX and helmet_debug_dets:
        for d in helmet_debug_dets:
            try:
                x1, y1, x2, y2 = d["box"]
                conf = float(d.get("conf", 0.0))
                name = str(d.get("name", "unknown"))

                if name == "helmet":
                    dbg_color = (255, 255, 0)
                elif name == "no_helmet":
                    dbg_color = (0, 255, 255)
                else:
                    dbg_color = (255, 0, 0)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), dbg_color, 2)
                cv2.putText(
                    frame,
                    f"HDBG {name} {conf:.2f}",
                    (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    dbg_color,
                    1
                )
            except Exception:
                pass

    total_person = len(persons)
    return helmet_person, no_helmet_person, total_person


# =====================
# 主推理循环
# =====================
def infer_loop():
    if ort is None:
        print("❌ 未安装 onnxruntime", flush=True)
        return

    if not os.path.exists(PERSON_ONNX):
        print(f"❌ 人体模型不存在: {PERSON_ONNX}", flush=True)
        return

    if not os.path.exists(HELMET_ONNX):
        print(f"❌ 头盔模型不存在: {HELMET_ONNX}", flush=True)
        return

    print("👉 加载人体 INT8 ONNX 模型...", flush=True)
    person_model = YOLOv5ONNX(
        PERSON_ONNX,
        input_size=PERSON_INFER_SIZE,
        conf_thres=PERSON_CONF,
        iou_thres=NMS_IOU,
        max_det=MAX_DET_PERSON,
        class_names=[
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
            "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
            "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
            "scissors","teddy bear","hair drier","toothbrush"
        ]
    )

    print("👉 加载头盔 INT8 ONNX 模型...", flush=True)
    helmet_model = YOLOv5ONNX(
        HELMET_ONNX,
        input_size=HELMET_INFER_SIZE,
        conf_thres=HELMET_CONF,
        iou_thres=NMS_IOU,
        max_det=MAX_DET_HELMET,
        class_names=["helmet", "no_helmet"]
    )

    cap = open_camera(SOURCE)
    if cap is None:
        print("❌ 摄像头打开失败", flush=True)
        return

    last_person_infer_time = 0.0
    last_helmet_infer_time = 0.0
    last_console_print = 0.0

    fps_last_time = time.time()
    fps_frame_count = 0
    display_fps = 0.0

    infer_last_stat_time = time.time()
    infer_count = 0
    display_infer_fps = 0.0

    cached_persons = []
    cached_helmets = []
    cached_nohelmets = []
    cached_helmet_debug_dets = []
    cached_frame_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        fps_frame_count += 1
        now_fps = time.time()
        if now_fps - fps_last_time >= 1.0:
            display_fps = fps_frame_count / (now_fps - fps_last_time)
            fps_frame_count = 0
            fps_last_time = now_fps

        now = time.time()
        did_infer = False
        t0 = time.time()

        # 同一轮最多只跑一个模型
        if now - last_person_infer_time >= PERSON_INFER_INTERVAL_SEC:
            try:
                person_dets = person_model.infer(frame)
                cached_persons = filter_by_name(person_dets, "person")
                last_person_infer_time = now
                did_infer = True
            except Exception as e:
                print("⚠ 人体推理异常：", e, flush=True)

        elif len(cached_persons) > 0 and (now - last_helmet_infer_time >= HELMET_INFER_INTERVAL_SEC):
            try:
                helmet_dets = helmet_model.infer(frame)

                cached_helmets = filter_by_name(helmet_dets, "helmet")
                cached_nohelmets = filter_by_name(helmet_dets, "no_helmet")
                cached_helmet_debug_dets = helmet_dets

                last_helmet_infer_time = now
                did_infer = True
            except Exception as e:
                print("⚠ 头盔推理异常：", e, flush=True)
        else:
            if len(cached_persons) == 0:
                cached_helmets = []
                cached_nohelmets = []
                cached_helmet_debug_dets = []

        if did_infer:
            cached_frame_ms = int((time.time() - t0) * 1000)
            infer_count += 1
            now_infer_stat = time.time()
            if now_infer_stat - infer_last_stat_time >= 1.0:
                display_infer_fps = infer_count / (now_infer_stat - infer_last_stat_time)
                infer_count = 0
                infer_last_stat_time = now_infer_stat

        helmet_person, no_helmet_person, total_person = draw_overlay(
            frame,
            cached_persons,
            cached_helmets,
            cached_nohelmets,
            cached_helmet_debug_dets
        )
        frame_ms = cached_frame_ms

        now_report = time.time()
        if now_report - last_console_print >= PRINT_STATUS_INTERVAL_SEC:
            print(
                f"识别状态 total={total_person} helmet={helmet_person} "
                f"no_helmet={no_helmet_person} infer={frame_ms}ms fps={display_fps:.1f}",
                flush=True
            )
            last_console_print = now_report

        cv2.putText(
            frame,
            f"Stable INT8 | P:{PERSON_INFER_SIZE}/{PERSON_INFER_INTERVAL_SEC:.1f}s H:{HELMET_INFER_SIZE}/{HELMET_INFER_INTERVAL_SEC:.2f}s | FPS:{display_fps:.1f} | Infer:{display_infer_fps:.1f}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        if SHOW_LOCAL_WINDOW:
            cv2.imshow("Helmet Check - Stable SplitRate", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# =====================
# 主程序
# =====================
if __name__ == "__main__":
    print("======================================", flush=True)
    print("安全帽识别启动：本地识别", flush=True)
    print(f"PERSON_ONNX={PERSON_ONNX}", flush=True)
    print(f"HELMET_ONNX={HELMET_ONNX}", flush=True)
    print("======================================", flush=True)

    infer_loop()
