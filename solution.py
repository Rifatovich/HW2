from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / "hw2"
OUT = BASE / "results"
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def to_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать {path}")
    return img


def norm(gx, gy):
    mag = cv2.magnitude(gx, gy)
    hi = np.percentile(mag, 99)
    return np.clip(mag / max(hi, 1e-6) * 255, 0, 255).astype(np.uint8)


def sobel(img):
    return norm(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))


def scharr(img):
    return norm(cv2.Scharr(img, cv2.CV_32F, 1, 0),
                cv2.Scharr(img, cv2.CV_32F, 0, 1))


def prewitt(img):
    kx = np.array([[-1, 0, 1]] * 3, dtype=np.float32)
    ky = kx.T
    return norm(cv2.filter2D(img, cv2.CV_32F, kx),
                cv2.filter2D(img, cv2.CV_32F, ky))


def canny(img):
    """Пороги подбираются по самому снимку: верхний - 92-й перцентиль градиента."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    nz = mag[mag > 0]
    high = float(np.percentile(nz, 92)) if nz.size else 1.0
    return cv2.Canny(img, 0.4 * high, high, L2gradient=True)


FILTERS = {
    "median": lambda im: cv2.medianBlur(im, 7),
    "gaussian": lambda im: cv2.GaussianBlur(im, (7, 7), 1.5),
}
OPERATORS = {"sobel": sobel, "scharr": scharr, "prewitt": prewitt, "canny": canny}


def process_image(img):
    """Принимает grayscale uint8, возвращает словарь из 8 результатов."""
    res = {}
    for fn, f in FILTERS.items():
        smooth = f(img)
        for on, op in OPERATORS.items():
            res[f"{fn}_{on}"] = op(smooth)
    return res


def boundary_score(edges, mask):
    """Доля энергии границ, попавшая в полосу вокруг контура почки."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = (mask > 0).astype(np.uint8)
    band = cv2.dilate(m, k) - cv2.erode(m, k)
    total = edges.astype(np.float64).sum()
    return float(edges[band > 0].astype(np.float64).sum() / total) if total else 0.0


def find_mask(img_path):
    stem = img_path.stem
    names = [stem.replace("img", "mask", 1), f"{stem}_mask", f"mask_{stem}"]
    for n in names:
        for e in EXT:
            p = img_path.with_name(n + e)
            if p.is_file() and p != img_path:
                return p
    return None


def tile(img, title, size=300):
    h, w = img.shape
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    t = np.zeros((size + 25, size), np.uint8)
    t[25:25 + r.shape[0], :r.shape[1]] = r
    cv2.putText(t, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    return t


def save_grid(img, mask, res, path):
    blank = np.zeros_like(tile(img, ""))
    rows = [[tile(img, "original"), tile(mask, "mask"), blank, blank]]
    for fn in FILTERS:
        rows.append([tile(res[f"{fn}_{on}"], f"{fn}+{on}") for on in OPERATORS])
    cv2.imwrite(str(path), np.vstack([np.hstack(r) for r in rows]))


def main():
    zip_path = BASE / "hw2.zip"
    if not DATA.is_dir() and zip_path.is_file():
        with ZipFile(zip_path) as z:
            z.extractall(DATA)
    if not DATA.is_dir():
        raise SystemExit("Нет папки hw2 или hw2.zip рядом со скриптом")

    imgs = sorted(p for p in DATA.rglob("*")
                  if p.suffix.lower() in EXT and "mask" not in p.stem.lower())
    if not imgs:
        raise SystemExit("В hw2 нет изображений")

    OUT.mkdir(exist_ok=True)
    scores = {}
    for p in imgs:
        img = to_gray(p)
        res = process_image(img)
        d = OUT / p.stem
        d.mkdir(exist_ok=True)
        for name, r in res.items():
            cv2.imwrite(str(d / f"{name}.png"), r)

        mp = find_mask(p)
        if mp is None:
            print(f"{p.name}: маска не найдена, сравнение пропущено")
            continue
        mask = to_gray(mp)
        if mask.shape != img.shape:
            print(f"{p.name}: размер маски не совпадает, сравнение пропущено")
            continue
        save_grid(img, mask, res, d / "comparison.png")
        scores[p.stem] = {k: boundary_score(v, mask) for k, v in res.items()}

    if scores:
        names = list(next(iter(scores.values())))
        print("\nДоля границ вокруг контура почки, больше значит лучше")
        print(f"{'':20}" + "".join(f"{s[-6:]:>10}" for s in scores))
        for n in names:
            print(f"{n:20}" + "".join(f"{scores[s][n]:>10.3f}" for s in scores))
    print(f"\nРезультаты в {OUT}")


if __name__ == "__main__":
    main()
