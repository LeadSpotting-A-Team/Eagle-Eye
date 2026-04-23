"""
Eagle-Eye – Module 2 v2: Illumination Invariant Pre-processing (UPGRADED)
==========================================================================
Upgrades over v1:
  - Multi color space: LAB (default), YCbCr, HSV
  - Vectorized NMS via np.roll (no pixel loops)
  - Adaptive shadow mask: low luminance AND dilated NMS edges
  - Entropy-driven compensation per LBP class, per BGR channel
      ent < 5.5  -> Texture-Priority (additive offset)
      ent >= 5.5 -> Color-Priority  (multiplicative ratio)
  - Final output: shadow-neutralized RGB image (not a gradient map)
  - process_pair: handles two images with all new logic

ProcessedImage.final is now a uint8 BGR shadow-neutralized image.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
from skimage.feature import local_binary_pattern
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Color space registry
# ---------------------------------------------------------------------------

_CS_MAP = {
    "LAB":   (cv2.COLOR_BGR2LAB,   0),
    "YCbCr": (cv2.COLOR_BGR2YCrCb, 0),
    "HSV":   (cv2.COLOR_BGR2HSV,   2),
}

# ---------------------------------------------------------------------------
# ProcessedImage data container
# ---------------------------------------------------------------------------

class ProcessedImage:
    """
    All intermediate and final outputs for one image.

    Attributes
    ----------
    original       : uint8 BGR  – input image
    L_raw          : float32    – luminance before CLAHE  [0,1]
    L_normalized   : float32    – luminance after CLAHE   [0,1]
    L_blurred      : float32    – luminance after blur    [0,1]
    gradient       : float32    – NMS-thinned gradient mag [0,1]
    lbp            : float32    – LBP texture map          [0,1]
    shadow_mask    : uint8      – binary shadow mask {0,255}
    final          : uint8 BGR  – shadow-neutralized image  ← UPGRADED
    weberface      : float32    – Weberface map [0,1] or None
    """

    def __init__(
        self,
        original: np.ndarray,
        L_raw: np.ndarray,
        L_normalized: np.ndarray,
        L_blurred: np.ndarray,
        gradient: np.ndarray,
        lbp: np.ndarray,
        shadow_mask: Optional[np.ndarray],
        final: np.ndarray,
        weberface: Optional[np.ndarray] = None,
    ) -> None:
        self.original     = original
        self.L_raw        = L_raw
        self.L_normalized = L_normalized
        self.L_blurred    = L_blurred
        self.gradient     = gradient
        self.lbp          = lbp
        self.shadow_mask  = shadow_mask
        self.final        = final
        self.weberface    = weberface


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _f2u(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Module2Processor (v2)
# ---------------------------------------------------------------------------

class Module2Processor:
    """
    Upgraded Illumination Invariant Pre-processor for drone image pairs.

    Parameters
    ----------
    color_space : str
        'LAB' (default), 'YCbCr', or 'HSV'.
    clahe_clip_limit : float
    clahe_tile_grid : tuple
    blur_kernel : int   (must be odd)
    lbp_radius : int
    lbp_n_points : int
    lbp_method : str    skimage method, default 'uniform'
    shadow_threshold_ratio : float
        Pixels with L < ratio*mean(L) are shadow candidates.
    entropy_threshold : float
        Boundary between Texture-Priority and Color-Priority compensation.
    enable_weberface : bool
    """

    _LBP_N = 256   # lookup table size

    def __init__(
        self,
        color_space: str = "LAB",
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        blur_kernel: int = 5,
        lbp_radius: int = 1,
        lbp_n_points: int = 8,
        lbp_method: str = "uniform",
        shadow_threshold_ratio: float = 0.6,
        entropy_threshold: float = 5.5,
        enable_weberface: bool = True,
    ) -> None:
        if color_space not in _CS_MAP:
            raise ValueError(f"color_space must be one of {list(_CS_MAP)}")
        self._cv2_code, self._lum_ch = _CS_MAP[color_space]
        self.color_space           = color_space
        self.clahe                 = cv2.createCLAHE(
                                         clipLimit=clahe_clip_limit,
                                         tileGridSize=clahe_tile_grid,
                                     )
        self.blur_kernel           = blur_kernel | 1   # ensure odd
        self.lbp_radius            = lbp_radius
        self.lbp_n_points          = lbp_n_points
        self.lbp_method            = lbp_method
        self.shadow_threshold_ratio = shadow_threshold_ratio
        self.entropy_threshold     = entropy_threshold
        self.enable_weberface      = enable_weberface
        self._lbp_table            = self._build_lbp_table(lbp_n_points)
        print(f"[Module 2] color_space={color_space}  entropy_thresh={entropy_threshold}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pair(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> Tuple[ProcessedImage, ProcessedImage]:
        return self._run(img_a), self._run(img_b)

    def process_single(self, img_bgr: np.ndarray) -> ProcessedImage:
        return self._run(img_bgr)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _run(self, img_bgr: np.ndarray) -> ProcessedImage:
        img_f32 = img_bgr.astype(np.float32) / 255.0

        # 1 – Luminance extraction
        L_raw = self._extract_L(img_bgr)

        # 2 – CLAHE
        L_norm = self._clahe(L_raw)

        # 3 – Gaussian blur
        L_blur = self._blur(L_norm)

        # 4 – Sobel + vectorized NMS
        grad_mag, grad_dir = self._sobel(L_blur)
        nms = self._nms(grad_mag, grad_dir)
        gradient = _norm01(nms)

        # 5 – LBP (skimage)
        lbp_map = self._lbp(L_blur)

        # 8 – Weberface (computed early to aid shadow detection/compensation)
        weberface = self._weberface(L_blur) if self.enable_weberface else None

        # 6 – Adaptive shadow mask (now uses morphological closing & weberface)
        shadow_mask = self._adaptive_shadow_mask(L_raw, nms, weberface)

        # 7 – Entropy-driven compensation (all BGR channels, cross-referenced with weberface)
        compensated = self._compensate(img_f32, L_raw, lbp_map, shadow_mask, weberface)
        final_bgr = np.clip(compensated * 255.0, 0, 255).astype(np.uint8)

        return ProcessedImage(
            original=img_bgr,
            L_raw=L_raw,
            L_normalized=L_norm,
            L_blurred=L_blur,
            gradient=gradient,
            lbp=lbp_map,
            shadow_mask=(shadow_mask.astype(np.uint8) * 255),
            final=final_bgr,
            weberface=weberface,
        )

    # ------------------------------------------------------------------
    # Step 1 – Luminance extraction
    # ------------------------------------------------------------------

    def _extract_L(self, img_bgr: np.ndarray) -> np.ndarray:
        converted = cv2.cvtColor(img_bgr, self._cv2_code).astype(np.float32) / 255.0
        return converted[:, :, self._lum_ch]

    # ------------------------------------------------------------------
    # Step 2 – CLAHE
    # ------------------------------------------------------------------

    def _clahe(self, L: np.ndarray) -> np.ndarray:
        return self.clahe.apply(_f2u(L)).astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Step 3 – Gaussian blur
    # ------------------------------------------------------------------

    def _blur(self, L: np.ndarray) -> np.ndarray:
        k = self.blur_kernel
        return cv2.GaussianBlur(_f2u(L), (k, k), 0).astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Step 4 – Sobel + vectorized NMS
    # ------------------------------------------------------------------

    @staticmethod
    def _sobel(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L_u8 = (L * 255).astype(np.uint8)
        Gx = cv2.Sobel(L_u8, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(L_u8, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(Gx ** 2 + Gy ** 2).astype(np.float32)
        direction = np.arctan2(Gy, Gx).astype(np.float32)
        return mag, direction

    @staticmethod
    def _nms(mag: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Vectorized Non-Maximum Suppression via np.roll (no pixel loops)."""
        angle = np.rad2deg(direction) % 180.0
        bin_idx = np.zeros_like(angle, dtype=np.uint8)
        bin_idx[(angle >= 22.5)  & (angle < 67.5)]  = 1
        bin_idx[(angle >= 67.5)  & (angle < 112.5)] = 2
        bin_idx[(angle >= 112.5) & (angle < 157.5)] = 3

        # (shift_r1, shift_c1), (shift_r2, shift_c2) per orientation bin
        shifts = {
            0: ((0, -1), (0,  1)),
            1: ((-1, 1), (1, -1)),
            2: ((-1, 0), (1,  0)),
            3: ((-1,-1), (1,  1)),
        }
        suppressed = np.zeros_like(mag)
        for k, ((r1, c1), (r2, c2)) in shifts.items():
            nb1 = np.roll(np.roll(mag, r1, axis=0), c1, axis=1)
            nb2 = np.roll(np.roll(mag, r2, axis=0), c2, axis=1)
            mask = (bin_idx == k) & (mag >= nb1) & (mag >= nb2)
            suppressed[mask] = mag[mask]
        return suppressed.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 5 – LBP (skimage)
    # ------------------------------------------------------------------

    def _lbp(self, L: np.ndarray) -> np.ndarray:
        raw = local_binary_pattern(
            _f2u(L), P=self.lbp_n_points,
            R=self.lbp_radius, method=self.lbp_method,
        ).astype(np.float32)
        return _norm01(raw)

    # ------------------------------------------------------------------
    # Step 6 – Adaptive shadow mask
    # ------------------------------------------------------------------

    def _adaptive_shadow_mask(
        self, L: np.ndarray, nms: np.ndarray, weberface: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Clean, simple, and highly accurate shadow masking.
        """
        avg_L = float(L.mean())
        # Shadow pixels are significantly darker than the mean
        low_lum = L < (self.shadow_threshold_ratio * avg_L)

        # 1. Start with purely dark pixels
        base_shadow = low_lum.astype(np.uint8) * 255

        # 2. Clean up small noise (e.g., dark spots that aren't shadows)
        kernel = np.ones((3, 3), np.uint8)
        shadow_cleaned = cv2.morphologyEx(base_shadow, cv2.MORPH_OPEN, kernel)

        # 3. Fill small holes inside the shadow
        close_kernel = np.ones((9, 9), np.uint8)
        shadow_filled = cv2.morphologyEx(shadow_cleaned, cv2.MORPH_CLOSE, close_kernel)
        
        return shadow_filled > 0

    # ------------------------------------------------------------------
    # Step 7 – Entropy-driven shadow compensation
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        hist, _ = np.histogram(values, bins=256, range=(0.0, 1.0))
        hist = hist[hist > 0].astype(np.float64)
        return float(scipy_entropy(hist / hist.sum(), base=2))

    def _compensate(
        self,
        img_f32: np.ndarray,
        L: np.ndarray,
        lbp_map: np.ndarray,
        shadow_mask: np.ndarray,
        weberface: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Direct Channel-Wise Gain in BGR space.
        Naturally corrects intensity, color shift, and preserves texture contrast.
        """
        compensated = img_f32.copy()
        shadow_u8 = shadow_mask.astype(np.uint8)
        
        if not shadow_mask.any():
            return compensated
            
        # Sharp feathering
        soft_shadow = cv2.GaussianBlur(shadow_u8.astype(np.float32), (7, 7), 0)
        
        # Process individual shadow regions
        num_labels, labels = cv2.connectedComponents(shadow_u8)
        
        for i in range(1, num_labels + 1):
            s_mask = labels == i
            
            # Find the immediate lit neighborhood around this specific shadow
            s_dilated = cv2.dilate(s_mask.astype(np.uint8), np.ones((15, 15), np.uint8))
            s_neighbors = (s_dilated > 0) & (~shadow_mask)
            
            if not s_neighbors.any():
                continue
                
            # Apply channel-wise gain to correct both brightness and ambient color shift
            for ch in range(3):
                val_s = img_f32[:, :, ch][s_mask].mean()
                val_n = img_f32[:, :, ch][s_neighbors].mean()
                
                # Avoid division by zero
                val_s = max(val_s, 1e-4)
                
                # Direct Multiplicative Gain
                gain = val_n / val_s
                
                # Limit the gain to prevent noise amplification in deep shadows
                gain = np.clip(gain, 1.0, 4.0)
                
                compensated[:, :, ch][s_mask] = img_f32[:, :, ch][s_mask] * gain

        # Alpha blend with the sharp soft shadow mask
        result = img_f32.copy()
        for ch in range(3):
            result[:, :, ch] = img_f32[:, :, ch] * (1.0 - soft_shadow) + compensated[:, :, ch] * soft_shadow

        return np.clip(result, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Bonus – Weberface
    # ------------------------------------------------------------------

    @staticmethod
    def _weberface(L: np.ndarray) -> np.ndarray:
        Y   = L.astype(np.float64)
        d   = Y + 1e-9
        acc = np.zeros_like(Y)
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            nb  = np.roll(np.roll(Y, dr, axis=0), dc, axis=1)
            acc += (Y - nb) / d
        return _norm01(np.arctan(acc).astype(np.float32))

    # ------------------------------------------------------------------
    # LBP lookup table (for internal uniform mapping)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lbp_table(n_points: int = 8) -> np.ndarray:
        table = np.full(256, 58, dtype=np.int32)
        idx = 0
        for code in range(256):
            bits = [(code >> i) & 1 for i in range(n_points)]
            if sum(bits[i] != bits[(i+1) % n_points] for i in range(n_points)) <= 2:
                table[code] = idx
                idx += 1
        return table

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_pair(
        self,
        proc_a: ProcessedImage,
        proc_b: ProcessedImage,
        label_a: str = "A",
        label_b: str = "B",
        scale: float = 1.0,
    ) -> np.ndarray:
        # Build two rows for A and two rows for B (4 panels per row)
        rows_a = self._build_rows_split(proc_a, label_a, scale)
        rows_b = self._build_rows_split(proc_b, label_b, scale)
        
        all_rows = rows_a + rows_b
        
        # Pad all rows to equal width
        max_w = max(r.shape[1] for r in all_rows)
        padded = []
        for r in all_rows:
            dw = max_w - r.shape[1]
            if dw > 0:
                r = np.hstack([r, np.zeros((r.shape[0], dw, 3), np.uint8)])
            padded.append(r)
            
        div = np.full((6, max_w, 3), 60, dtype=np.uint8)
        canvas = padded[0]
        for r in padded[1:]:
            canvas = np.vstack([canvas, div, r])
            
        return canvas

    def _build_rows_split(self, proc: ProcessedImage, label: str, scale: float) -> list[np.ndarray]:
        def g2b(arr):
            u = _f2u(arr) if arr.dtype != np.uint8 else arr
            return cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)

        def rsz(img):
            h, w = img.shape[:2]
            target_h = 220  # Comfortable height
            new_w = int(w * (target_h / h) * scale)
            return cv2.resize(img, (new_w, target_h))

        all_panels = [
            (rsz(proc.original), "Original"),
            (rsz(g2b(proc.L_raw)), "L-raw"),
            (rsz(g2b(proc.L_normalized)), "L-CLAHE"),
            (rsz(g2b(proc.gradient)), "Gradient(NMS)"),
            (rsz(g2b(proc.lbp)), "LBP"),
            (rsz(g2b(proc.shadow_mask)), "Shadow"),
            (rsz(proc.final), "Compensated"),
        ]
        if proc.weberface is not None:
            all_panels.append((rsz(g2b(proc.weberface)), "Weberface"))

        # Split into rows of max 4
        chunk_size = 4
        rows = []
        for i in range(0, len(all_panels), chunk_size):
            chunk = all_panels[i:i + chunk_size]
            
            h_max = max(p[0].shape[0] for p in chunk)
            aligned = []
            for p_img, p_lbl in chunk:
                dh = h_max - p_img.shape[0]
                aligned.append(np.vstack([p_img, np.zeros((dh, p_img.shape[1], 3), np.uint8)]) if dh > 0 else p_img)
            
            div = np.full((h_max, 2, 3), 40, dtype=np.uint8)
            row_img = aligned[0]
            for p in aligned[1:]:
                row_img = np.hstack([row_img, div, p])
            
            # Add labels and banner
            banner_w = 40
            banner = np.full((h_max, banner_w, 3), 30, dtype=np.uint8)
            row_lbl = f"{label} (P{i//chunk_size + 1})"
            cv2.putText(banner, row_lbl, (2, h_max//2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1)
            row_img = np.hstack([banner, row_img])
            
            # Put column labels
            curr_x = banner_w + 3
            for j, (p_img, p_lbl) in enumerate(chunk):
                cv2.putText(row_img, p_lbl, (curr_x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1)
                curr_x += aligned[j].shape[1] + 2
                
            rows.append(row_img)
        return rows


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os

    def _safe_load(path, name):
        if not path or not os.path.exists(path):
            print(f"[WARN] Could not load '{path}' -- generating synthetic {name}")
            H, W = 256, 384
            base = np.tile(np.linspace(60, 210, W, dtype=np.uint8), (H, 1))
            img  = cv2.merge([base, base, base])
            img[70:150, 100:270] = (img[70:150, 100:270] * 0.25).astype(np.uint8)
            if name == "B":
                img = np.clip(img.astype(np.int32) + 40, 0, 255).astype(np.uint8)
            return img
        
        # Safe read for paths with Hebrew/Unicode
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _safe_save(path, img):
        # Safe save for paths with Hebrew/Unicode
        ext = os.path.splitext(path)[1]
        is_success, buffer = cv2.imencode(ext, img)
        if is_success:
            buffer.tofile(path)
            return True
        return False

    img_a = _safe_load(sys.argv[1] if len(sys.argv) > 1 else "", "A")
    img_b = _safe_load(sys.argv[2] if len(sys.argv) > 2 else "", "B")
    print(f"[INFO] A: {img_a.shape}  B: {img_b.shape}")

    proc = Module2Processor(
        color_space="LAB",
        shadow_threshold_ratio=0.6,
        entropy_threshold=5.5,
        enable_weberface=True,
    )

    pa, pb = proc.process_pair(img_a, img_b)

    canvas = proc.visualize_pair(pa, pb, label_a="Img A", label_b="Img B")
    
    # Use default window sizing to prevent smearing
    cv2.imshow("Eagle-Eye Module 2 v2", canvas)
    
    print("[INFO] Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # Save outputs to a dedicated Output folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "Output")
    os.makedirs(out_dir, exist_ok=True)

    _safe_save(os.path.join(out_dir, "m2v2_A_final.png"), pa.final)
    _safe_save(os.path.join(out_dir, "m2v2_B_final.png"), pb.final)
    _safe_save(os.path.join(out_dir, "m2v2_canvas.png"),  canvas)
    print(f"[INFO] Outputs saved -> {out_dir}")


