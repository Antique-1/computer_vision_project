import os, glob, shutil, random, argparse, yaml
from pathlib import Path

DEFAULT_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
CLASS_MAP = {
    "금속캔": 0,
    "종이": 1,
    "페트병": 2,
    "플라스틱": 3,
    "스티로폼": 4,
    "비닐": 5,
    "유리병": 6
}

def find_all_images(root):
    imgs = []
    for ext in DEFAULT_IMAGE_EXT:
        imgs.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(imgs)

def corresponding_label_path(img_path, images_root, labels_root):
    rel = os.path.relpath(img_path, images_root)
    base = Path(rel).stem
    # 1) same relative dir
    rel_dir = os.path.dirname(rel)
    candidate1 = os.path.join(labels_root, rel_dir, base + ".txt")
    if os.path.exists(candidate1):
        return candidate1
    # 2) global search - if multiple matches, pick the one in same folder name as image parent if possible
    matches = glob.glob(os.path.join(labels_root, "**", base + ".txt"), recursive=True)
    if not matches:
        return None
    # try match by parent's folder name
    img_parent = os.path.basename(os.path.dirname(img_path))
    for m in matches:
        if os.path.basename(os.path.dirname(m)) == img_parent:
            return m
    return matches[0]

def safe_copy(src, dst, overwrite=False):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and not overwrite:
        return dst
    shutil.copy2(src, dst)
    return dst

def prepare_dataset(images_root, labels_root, out_root,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                    seed=42, path_type="relative", overwrite=False, exclude_no_label=False):
    random.seed(seed)
    images = find_all_images(images_root)
    if not images:
        raise SystemExit("images_root에서 이미지가 없습니다: " + images_root)

    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]

    print(f"총 이미지 {n} -> train:{len(train_imgs)}, val:{len(val_imgs)}, test:{len(test_imgs)}")

    imgs_out_train = os.path.join(out_root, "images", "train")
    imgs_out_val = os.path.join(out_root, "images", "val")
    imgs_out_test = os.path.join(out_root, "images", "test")
    lbls_out_train = os.path.join(out_root, "labels", "train")
    lbls_out_val = os.path.join(out_root, "labels", "val")
    lbls_out_test = os.path.join(out_root, "labels", "test")
    for d in [imgs_out_train, imgs_out_val, imgs_out_test, lbls_out_train, lbls_out_val, lbls_out_test]:
        os.makedirs(d, exist_ok=True)

    bad_log = []
    def process_list(img_list, dst_img_dir, dst_lbl_dir):
        saved_paths = []
        for img in img_list:
            label = corresponding_label_path(img, images_root, labels_root)
            if not label and exclude_no_label:
                bad_log.append(f"No label => excluded: {img}")
                continue
            dst_img = os.path.join(dst_img_dir, os.path.basename(img))
            safe_copy(img, dst_img, overwrite=overwrite)
            dst_lbl = os.path.join(dst_lbl_dir, Path(img).stem + ".txt")
            if label:
                safe_copy(label, dst_lbl, overwrite=overwrite)
            else:
                open(dst_lbl, "w", encoding="utf-8").close()
                bad_log.append(f"No label, created empty txt: {img}")
            if path_type == "absolute":
                saved_paths.append(os.path.abspath(dst_img))
            else:
                saved_paths.append(os.path.relpath(dst_img, start=out_root))
        return saved_paths

    train_list = process_list(train_imgs, imgs_out_train, lbls_out_train)
    val_list = process_list(val_imgs, imgs_out_val, lbls_out_val)
    test_list = process_list(test_imgs, imgs_out_test, lbls_out_test)

    # write lists
    def write_list(path, lst):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lst) + ("\n" if lst else ""))

    write_list(os.path.join(out_root, "train.txt"), train_list)
    write_list(os.path.join(out_root, "val.txt"), val_list)
    write_list(os.path.join(out_root, "test.txt"), test_list)

    names = [None] * (max(CLASS_MAP.values()) + 1)
    for k,v in CLASS_MAP.items():
        names[v] = k
    for i in range(len(names)):
        if names[i] is None:
            names[i] = f"class_{i}"

    yaml_content = {
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "test": os.path.join("images", "test"),
        "nc": len(names),
        "names": names
    }
    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, allow_unicode=True)
    bad_log_path = os.path.join(out_root, "bad_files.log")
    with open(bad_log_path, "w", encoding="utf-8") as f:
        if bad_log:
            f.write("\n".join(bad_log) + "\n")
        else:
            f.write("OK - no issues\n")

    print("Dataset prepared at:", out_root)
    print("data.yaml:", yaml_path)
    print("bad_files.log:", bad_log_path)
    return yaml_path

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", required=True)
    p.add_argument("--labels_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--path_type", choices=["relative","absolute"], default="relative")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--exclude_no_label", action="store_true",
                   help="라벨 없는 이미지를 dataset에서 제외하려면 사용")
    args = p.parse_args()
    prepare_dataset(args.images_root, args.labels_root, args.out_root,
                    train_ratio=args.train, val_ratio=args.val, test_ratio=args.test,
                    seed=args.seed, path_type=args.path_type, overwrite=args.overwrite,
                    exclude_no_label=args.exclude_no_label)

if __name__ == "__main__":
    cli()