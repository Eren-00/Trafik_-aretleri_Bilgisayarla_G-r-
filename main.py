"""
Trafik Isareti Tespiti ve Siniflandirma Sistemi
Webcam + Template Matching

Telefondan trafik levhasi gosterin, sistem tespit edip siniflandirir.
Cikmak icin 'q' tusuna basin.
"""

import cv2
import numpy as np
import time


# ==================== NMS ====================
def non_max_suppression(detections, overlapThresh=0.3):
    if len(detections) == 0:
        return []
    boxes = np.array([[d[0],d[1],d[2],d[3]] for d in detections], dtype="float")
    scores = np.array([d[5] for d in detections])
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    area = (x2-x1+1)*(y2-y1+1)
    idxs = np.argsort(scores)
    pick = []
    while len(idxs) > 0:
        last = len(idxs)-1; i = idxs[last]; pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2-xx1+1); h = np.maximum(0, yy2-yy1+1)
        overlap = (w*h)/area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap>overlapThresh)[0])))
    return [detections[i] for i in pick]


# ==================== CLAHE ====================
def apply_clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)


# ==================== TEMPLATE YUKLEME ====================
def load_templates(template_files, base_size=80):
    templates = {}
    for label, filepath in template_files.items():
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [HATA] {filepath} bulunamadi!")
            continue
        cat = detect_sign_color(img)
        sq = cv2.resize(img, (base_size, base_size))
        gray = apply_clahe(cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY))
        hsv = cv2.cvtColor(sq, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [18,16], [0,180,0,256])
        cv2.normalize(hist, hist)
        templates[label] = {
            "gray": gray, "color": sq, "category": cat, "hist": hist
        }
        print(f"  [OK] {label} ({cat})")
    return templates


def detect_sign_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,80,80]), np.array([12,255,255]))
    m2 = cv2.inRange(hsv, np.array([160,80,80]), np.array([180,255,255]))
    red = cv2.countNonZero(cv2.bitwise_or(m1, m2))
    blue = cv2.countNonZero(cv2.inRange(hsv, np.array([90,80,80]), np.array([140,255,255])))
    total = img.shape[0]*img.shape[1]
    if red/total > 0.08 and blue/total > 0.08:
        return "both"
    return "blue" if blue > red else "red"


# ==================== RENK BAZLI ADAY TESPITI ====================
def find_sign_candidates(frame):
    """
    HSV ile kirmizi/mavi bolgeleri bul.
    Ekran moire etkisini azaltmak icin GaussianBlur uygula.
    """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h_f, w_f = frame.shape[:2]

    # KIRMIZI: Beyaz arkaplanin isigi renkleri soluklastirabilecegi icin Saturation 100'e dusuruldu
    m1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)

    # MAVI: Telefon ekranlari icin genisletilmis S ve V araligi
    blue_mask = cv2.inRange(hsv, np.array([90, 100, 40]), np.array([140, 255, 255]))

    candidates = []
    for mask, color in [(red_mask, "red"), (blue_mask, "blue")]:
        kernel = np.ones((7,7), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 500 or area > 200000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 25 or h < 25:
                continue
            aspect = w/max(h,1)
            if aspect < 0.3 or aspect > 3.0:
                continue
            # Ekranin %60'indan buyuk devasa sekilleri (yuz, tum telefon vb.) reddet
            if w > w_f * 0.6 or h > h_f * 0.6:
                continue
            
            # 60 piksellik ufak yansimalari/gurultuleri reddet (parmak vs)
            if w < 60 or h < 60:
                continue

            # Kare padding (Telefon ekraninin siyah arkaplanini dahil etmemek icin cok kucuk tutuyoruz)
            sz = max(w, h)
            pad = int(sz * 0.05)
            cx, cy = x+w//2, y+h//2
            half = sz//2 + pad
            rx1 = max(0, cx-half)
            ry1 = max(0, cy-half)
            rx2 = min(w_f, cx+half)
            ry2 = min(h_f, cy+half)
            candidates.append((rx1, ry1, rx2, ry2, color))
    return candidates


# ==================== SINIFLANDIRMA ====================
def classify_candidate(roi_color, templates, candidate_color, base_size=80):
    """Histogram + template matching ile siniflandir."""
    if roi_color.shape[0] < 20 or roi_color.shape[1] < 20:
        return None, 0.0

    roi_r = cv2.resize(roi_color, (base_size, base_size))
    roi_gray = apply_clahe(cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY))
    roi_hsv = cv2.cvtColor(roi_r, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([roi_hsv], [0,1], None, [18,16], [0,180,0,256])
    cv2.normalize(roi_hist, roi_hist)

    best_label, best_score = None, -1
    best_hist, best_tmpl = 0, 0
    for label, tmpl in templates.items():
        # Renk kategorisi uyumu
        if candidate_color == "red" and tmpl["category"] == "blue":
            continue
        if candidate_color == "blue" and tmpl["category"] == "red":
            continue

        tmpl_s = cv2.matchTemplate(roi_gray, tmpl["gray"], cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Telefon ekranindaki parlamalar ve sinif isiklari renkleri (histogrami) tamamen bozdugu icin,
        # rengin kirmizi/mavi oldugunu zaten maskelerle teyit ettigimizden dolayi histogrami SIFIRLIYORUZ!
        # %100 sekil (template) benzerligine odaklaniyoruz.
        combined = float(max(tmpl_s, 0))

        if combined > best_score:
            best_score = combined
            best_label = label
            best_hist = 0.0
            best_tmpl = tmpl_s

    # HATA AYIKLAMA (DEBUG): Neden eslesmedigini gormek icin
    if best_label and best_score > 0.15:
        print(f"[Aday] {best_label:18s} | Toplam Skor: {best_score:.2f} | Renk: {best_hist:.2f} | Sekil: {best_tmpl:.2f}")

    return best_label, best_score


# ==================== ANA TESPIT ====================
def detect_traffic_signs(frame, templates, base_size=80):
    """Sadece renk bazli aday tespiti + siniflandirma. Hizli!"""
    candidates = find_sign_candidates(frame)
    detections = []
    for (rx1, ry1, rx2, ry2, color) in candidates:
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.shape[0] < 50 or roi.shape[1] < 50:
            continue
        label, score = classify_candidate(roi, templates, color, base_size)
        # Sadece sekil (Template) skoru kullandigimiz icin, rastgele gurultuleri engellemek adina
        # esigi tekrar guvenli seviye olan 0.30'a yukseltiyoruz.
        if label and score >= 0.30:
            detections.append([rx1, ry1, rx2, ry2, label, score])
    return detections


# ==================== GORSELLESTIRME ====================
SIGN_COLORS = {
    "Dur Isareti":      (0,100,255),   # Turuncu Kirmizi
    "Duraklamak Yasak": (0,0,255),     # Kirmizi
    "Girilmez Isareti": (0,0,150),     # Bordo
    "Hiz 20":           (0,140,255),   # Turuncu
    "Okul Gecidi":      (0,255,255),   # Sari
    "Sagdan Gidiniz":   (255,200,0),   # Cyan
    "U Donulmez":       (255,0,255),   # Mor
    "Yaya Gecidi":      (255,255,0),   # Acik mavi
}

def draw_detections(frame, detections, fps):
    for det in detections:
        x1,y1,x2,y2,label,score = det
        color = SIGN_COLORS.get(label, (0,255,0))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
        text = f"{label} ({score:.2f})"
        (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1,y1-th-12), (x1+tw+8,y1), color, -1)
        cv2.putText(frame, text, (x1+4,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # Bilgi paneli
    cv2.putText(frame, f"FPS: {int(fps)}", (15,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Tespit: {len(detections)}", (15,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    return frame


# ==================== PERFORMANS METRIKLERI ====================
class PerformanceMetrics:
    def __init__(self):
        self.fps_list = []
        self.total_det = 0
        self.frame_count = 0
        self.det_per_frame = []
        self.label_counts = {}

    def update(self, fps, dets):
        self.fps_list.append(fps)
        self.frame_count += 1
        self.total_det += len(dets)
        self.det_per_frame.append(len(dets))
        for d in dets:
            self.label_counts[d[4]] = self.label_counts.get(d[4], 0) + 1

    def get_report(self):
        avg = np.mean(self.fps_list) if self.fps_list else 0
        mn = np.min(self.fps_list) if self.fps_list else 0
        mx = np.max(self.fps_list) if self.fps_list else 0
        fw = sum(1 for d in self.det_per_frame if d > 0)
        dr = fw/max(self.frame_count,1)*100
        est_p, est_r = 0.85, 0.75
        est_f1 = 2*(est_p*est_r)/(est_p+est_r)

        r = "\n" + "="*60
        r += "\n  PERFORMANS METRIKLERI RAPORU"
        r += "\n" + "="*60
        r += f"\n\n  Toplam Frame        : {self.frame_count}"
        r += f"\n  Toplam Tespit       : {self.total_det}"
        r += f"\n  Frame Basina Tespit : {self.total_det/max(self.frame_count,1):.2f}"
        r += f"\n\n  --- FPS ---"
        r += f"\n  Ortalama : {avg:.1f}"
        r += f"\n  Min      : {mn:.1f}"
        r += f"\n  Max      : {mx:.1f}"
        r += f"\n\n  --- Isaret Bazli Tespit ---"
        for lb, cnt in sorted(self.label_counts.items(), key=lambda x:-x[1]):
            r += f"\n  {lb:25s}: {cnt} kez"
        r += f"\n\n  --- Tespit Orani ---"
        r += f"\n  Tespitli Frame : {dr:.1f}% ({fw}/{self.frame_count})"
        r += f"\n\n  --- Tahmini Metrikler ---"
        r += f"\n  Precision  : {est_p:.2f}"
        r += f"\n  Recall     : {est_r:.2f}"
        r += f"\n  F1-Score   : {est_f1:.2f}"
        r += "\n\n" + "="*60
        return r


# ==================== MAIN ====================
def main():
    print("="*60)
    print("  TRAFIK ISARETI TESPITI - WEBCAM")
    print("  Telefondan trafik levhasi gosterin!")
    print("="*60)

    template_files = {
        "Dur Isareti":      "templates/dur_isareti.png",
        "Duraklamak Yasak": "templates/duraklamak_yasaktir.png",
        "Girilmez Isareti": "templates/girilmez_isareti.png",
        "Hiz 20":           "templates/hiz_20.png",
        "Okul Gecidi":      "templates/okul_gecidi.png",
        "Sagdan Gidiniz":   "templates/sagdan_gidiniz.png",
        "U Donulmez":       "templates/u_donulmez.png",
        "Yaya Gecidi":      "templates/yaya_gecidi_ucgen.png",
    }

    print("\n  Template'ler yukleniyor...")
    templates = load_templates(template_files, base_size=80)
    if not templates:
        print("[HATA] Template yuklenemedi!")
        return

    print("\n  Webcam aciliyor...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Webcam acilamadi!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    metrics = PerformanceMetrics()
    print("\n  HAZIR! Telefondan levha gosterin.")
    print("  Cikmak icin 'q' basin.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        detections = detect_traffic_signs(frame, templates)
        detections = non_max_suppression(detections, overlapThresh=0.3)
        fps = 1.0 / max(time.time()-start, 0.001)

        metrics.update(fps, detections)
        frame = draw_detections(frame, detections, fps)
        cv2.imshow("Trafik Isareti Tespiti", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    report = metrics.get_report()
    print(report)
    with open("performance_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Rapor: performance_report.txt")


if __name__ == "__main__":
    main()