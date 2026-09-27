#!/usr/bin/env python
"""把 Oxford 102 Flowers 整理结果转成 FloraQwen 训练用 labels.csv。

输入:  data/raw/flowers102/{labels.csv, splits.json}  (由 download_flowers102.py 生成)
输出:  data/raw/labels.csv   (prepare_data.py 期望格式: image,species_zh,species_en,family,features,habitat)

物种名映射:
    - 标签顺序: imagelabels.mat 的编号 1-102 对应 TensorFlow Datasets 官方
      oxford_flowers102 builder 中的类名顺序 (本脚本 CANONICAL_NAMES 逐条照录)。
      ⚠️ 注意: VGG 官网 categories.html 是【字母序】, 与标签编号【不一致】,
      不能直接拿来按序号映射 (2026-09-17 踩坑记录, 见 docs 工作记录)。
    - 中文名/科属: 按通用园艺名与植物分类填写; 少数园艺品种 (bolero deep blue、
      cape flower) 无法可靠定科, 统一标 "园艺栽培品种", 见 SRC_NOTES 逐条说明。
    - features: 采用"科级"通用形态描述 (有分类学依据, 不做逐种编造)。

用法:
    python scripts/make_flowers102_labels.py            # 只输出官方 train 划分 (1020 张)
    python scripts/make_flowers102_labels.py --all      # 输出全部 8189 张
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# imagelabels.mat 编号 1-102 对应的官方类名 (顺序来源: TFDS oxford_flowers102 builder)
CANONICAL_NAMES = [
    "Pink Primrose", "Hard-leaved Pocket Orchid", "Canterbury Bells", "Sweet Pea",
    "English Marigold", "Tiger Lily", "Moon Orchid", "Bird of Paradise",
    "Monkshood", "Globe Thistle", "Snapdragon", "Colt's Foot",
    "King Protea", "Spear Thistle", "Yellow Iris", "Globe-flower",
    "Purple Coneflower", "Peruvian Lily", "Balloon Flower", "Giant White Arum Lily",
    "Fire Lily", "Pincushion Flower", "Fritillary", "Red Ginger",
    "Grape Hyacinth", "Corn Poppy", "Prince of Wales Feathers", "Stemless Gentian",
    "Artichoke", "Sweet William", "Carnation", "Garden Phlox",
    "Love in the Mist", "Mexican Aster", "Alpine Sea Holly", "Ruby-lipped Cattleya",
    "Cape Flower", "Great Masterwort", "Siam Tulip", "Lenten Rose",
    "Barbeton Daisy", "Daffodil", "Sword Lily", "Poinsettia",
    "Bolero Deep Blue", "Wallflower", "Marigold", "Buttercup",
    "Oxeye Daisy", "Common Dandelion", "Petunia", "Wild Pansy",
    "Primula", "Sunflower", "Pelargonium", "Bishop of Llandaff",
    "Gaura", "Geranium", "Orange Dahlia", "Pink-yellow Dahlia?",
    "Cautleya Spicata", "Japanese Anemone", "Black-eyed Susan", "Silverbush",
    "Californian Poppy", "Osteospermum", "Spring Crocus", "Bearded Iris",
    "Windflower", "Tree Poppy", "Gazania", "Azalea",
    "Water Lily", "Rose", "Thorn Apple", "Morning Glory",
    "Passion Flower", "Lotus", "Toad Lily", "Anthurium",
    "Frangipani", "Clematis", "Hibiscus", "Columbine",
    "Desert-rose", "Tree Mallow", "Magnolia", "Cyclamen",
    "Watercress", "Canna Lily", "Hippeastrum", "Bee Balm",
    "Ball Moss", "Foxglove", "Bougainvillea", "Camellia",
    "Mallow", "Mexican Petunia", "Bromelia", "Blanket Flower",
    "Trumpet Creeper", "Blackberry Lily",
]

# 英文名 -> (中文名, 科, 科级特征描述)
NAME_INFO = {
    "Pink Primrose": ("粉花月见草", "柳叶菜科", "花四瓣，子房下位"),
    "Hard-leaved Pocket Orchid": ("硬叶兜兰", "兰科", "花两侧对称，具特化唇瓣"),
    "Canterbury Bells": ("坎特伯雷风铃草", "桔梗科", "花冠钟形，多为蓝色系"),
    "Sweet Pea": ("香豌豆", "豆科", "蝶形花冠，攀援草本"),
    "English Marigold": ("金盏花", "菊科", "头状花序，舌状花橙黄色"),
    "Tiger Lily": ("卷丹", "百合科", "花被六枚反卷，具深色斑点"),
    "Moon Orchid": ("蝴蝶兰", "兰科", "花两侧对称，具特化唇瓣"),
    "Bird of Paradise": ("鹤望兰", "鹤望兰科", "花序形似鸟首，具橙色苞片"),
    "Monkshood": ("乌头", "毛茛科", "花两侧对称，上萼片盔状"),
    "Globe Thistle": ("蓝刺头", "菊科", "头状花序球形，蓝色"),
    "Snapdragon": ("金鱼草", "车前科", "总状花序，花冠二唇形"),
    "Colt's Foot": ("款冬", "菊科", "头状花序，先花后叶"),
    "King Protea": ("帝王花", "山龙眼科", "头状花序，具大型木质总苞"),
    "Spear Thistle": ("翼蓟", "菊科", "头状花序，茎叶具刺"),
    "Yellow Iris": ("黄菖蒲", "鸢尾科", "花被六枚，外轮下垂"),
    "Globe-flower": ("金莲花", "毛茛科", "花多具多数雄蕊与雌蕊，花冠球状"),
    "Purple Coneflower": ("紫锥菊（松果菊）", "菊科", "头状花序，管状花凸起如松果"),
    "Peruvian Lily": ("六出花（秘鲁百合）", "六出花科", "花被六枚，内轮具斑纹"),
    "Balloon Flower": ("桔梗", "桔梗科", "花冠钟形，多为蓝紫色"),
    "Giant White Arum Lily": ("白马蹄莲", "天南星科", "具白色佛焰苞与肉穗花序"),
    "Fire Lily": ("嘉兰（火焰百合）", "秋水仙科", "花被六枚，花瓣强烈反卷"),
    "Pincushion Flower": ("蓝盆花（松虫草）", "川续断科", "头状花序，总苞片明显"),
    "Fritillary": ("花格贝母", "百合科", "花被六枚，具网格状斑纹"),
    "Red Ginger": ("红姜花", "姜科", "花序具覆瓦状苞片，苞片红色"),
    "Grape Hyacinth": ("葡萄风信子", "百合科", "总状花序，小花坛状簇生"),
    "Corn Poppy": ("虞美人", "罂粟科", "花瓣四枚，雄蕊多数，花瓣薄纸质"),
    "Prince of Wales Feathers": ("千穗谷（威尔士王子羽毛）", "苋科", "穗状花序，花序色彩艳丽"),
    "Stemless Gentian": ("无茎龙胆", "龙胆科", "花冠钟形，多为蓝色"),
    "Artichoke": ("菜蓟（洋蓟）", "菊科", "头状花序，总苞片肉质"),
    "Sweet William": ("美国石竹（须苞石竹）", "石竹科", "花瓣先端常具锯齿，聚伞花序"),
    "Carnation": ("康乃馨（香石竹）", "石竹科", "花瓣先端常具锯齿，节部膨大"),
    "Garden Phlox": ("宿根福禄考", "花荵科", "伞房状聚伞花序，花冠五裂"),
    "Love in the Mist": ("黑种草", "毛茛科", "花被片多枚，具羽状分裂苞叶"),
    "Mexican Aster": ("波斯菊", "菊科", "头状花序，舌状花八枚左右"),
    "Alpine Sea Holly": ("高山刺芹", "伞形科", "头状花序，苞片具刺"),
    "Ruby-lipped Cattleya": ("红唇卡特兰", "兰科", "花两侧对称，唇瓣色深"),
    "Cape Flower": ("好望角花（园艺品种）", "园艺栽培品种", "观赏栽培品种"),
    "Great Masterwort": ("大星芹", "伞形科", "复伞形花序，具纸质苞片"),
    "Siam Tulip": ("姜荷花", "姜科", "花序具覆瓦状苞片，粉红色"),
    "Lenten Rose": ("铁筷子（圣诞玫瑰）", "毛茛科", "花多具多数雄蕊，花被片宿存"),
    "Barbeton Daisy": ("巴伯顿雏菊", "菊科", "头状花序，具舌状花与管状花"),
    "Daffodil": ("黄水仙", "石蒜科", "花被六枚，具杯状副花冠"),
    "Sword Lily": ("唐菖蒲（剑兰）", "鸢尾科", "穗状花序，花被六枚"),
    "Poinsettia": ("一品红", "大戟科", "杯状聚伞花序，苞片色彩艳丽"),
    "Bolero Deep Blue": ("博莱罗深蓝（园艺品种）", "园艺栽培品种", "观赏栽培品种"),
    "Wallflower": ("桂竹香", "十字花科", "花瓣四枚排成十字形"),
    "Marigold": ("万寿菊", "菊科", "头状花序，橙黄色"),
    "Buttercup": ("毛茛", "毛茛科", "花多具多数离生雄蕊与雌蕊，花瓣具光泽"),
    "Oxeye Daisy": ("滨菊（牛眼菊）", "菊科", "头状花序，舌状花白色"),
    "Common Dandelion": ("蒲公英", "菊科", "头状花序全为舌状花"),
    "Petunia": ("矮牵牛", "茄科", "花冠合瓣，漏斗状，五裂"),
    "Wild Pansy": ("三色堇（野堇）", "堇菜科", "花两侧对称，具距"),
    "Primula": ("报春花", "报春花科", "花冠合瓣五裂，伞形花序"),
    "Sunflower": ("向日葵", "菊科", "大型头状花序，中心管状花密集"),
    "Pelargonium": ("天竺葵", "牻牛儿苗科", "伞形花序，花两侧微对称"),
    "Bishop of Llandaff": ("兰达夫主教大丽花", "菊科", "头状花序，园艺栽培品种"),
    "Gaura": ("山桃草", "柳叶菜科", "花四瓣，子房下位，花瓣蝶形展开"),
    "Geranium": ("老鹳草", "牻牛儿苗科", "花五瓣，果实具长喙"),
    "Orange Dahlia": ("橙色大丽花", "菊科", "头状花序，园艺栽培品种"),
    "Pink-yellow Dahlia?": ("粉黄花大丽花", "菊科", "头状花序，园艺栽培品种"),
    "Cautleya Spicata": ("红苞距药姜", "姜科", "花序具覆瓦状苞片，苞片红色"),
    "Japanese Anemone": ("秋牡丹（日本银莲花）", "毛茛科", "花多具多数雄蕊与雌蕊，花被片多枚"),
    "Black-eyed Susan": ("黑心金光菊", "菊科", "头状花序，中心管状花深色"),
    "Silverbush": ("银旋花", "旋花科", "花冠漏斗状，叶被银色绒毛"),
    "Californian Poppy": ("花菱草", "罂粟科", "花瓣四枚，雄蕊多数"),
    "Osteospermum": ("蓝目菊", "菊科", "头状花序，舌状花蓝紫色"),
    "Spring Crocus": ("春番红花", "鸢尾科", "花被六枚，具橙色柱头"),
    "Bearded Iris": ("德国鸢尾", "鸢尾科", "花被六枚，外轮下垂具须毛"),
    "Windflower": ("银莲花", "毛茛科", "花被片多枚，雄蕊多数"),
    "Tree Poppy": ("树罂粟", "罂粟科", "花瓣多枚，雄蕊多数，亚灌木"),
    "Gazania": ("勋章菊", "菊科", "头状花序，舌状花具深色环纹"),
    "Azalea": ("杜鹃花", "杜鹃花科", "花冠合瓣，雄蕊常为花冠裂片两倍"),
    "Water Lily": ("睡莲", "睡莲科", "叶浮于水面，花漂浮或挺出水面"),
    "Rose": ("月季（玫瑰）", "蔷薇科", "多为五瓣花，雄蕊多数，茎常具皮刺"),
    "Thorn Apple": ("曼陀罗", "茄科", "花冠合瓣漏斗状，果实具刺"),
    "Morning Glory": ("牵牛花", "旋花科", "花冠漏斗状，攀援草本"),
    "Passion Flower": ("西番莲", "西番莲科", "具丝状副花冠，雌雄蕊柄显著"),
    "Lotus": ("荷花", "莲科", "叶盾形挺出水面，具莲蓬"),
    "Toad Lily": ("油点草", "百合科", "花被六枚，具斑点"),
    "Anthurium": ("花烛（红掌）", "天南星科", "具佛焰苞与肉穗花序，革质叶"),
    "Frangipani": ("鸡蛋花", "夹竹桃科", "花冠五裂，旋转排列"),
    "Clematis": ("铁线莲", "毛茛科", "花多具多数雄蕊与雌蕊，攀援藤本"),
    "Hibiscus": ("木槿", "锦葵科", "花五瓣，具副萼，雄蕊单体"),
    "Columbine": ("耧斗菜", "毛茛科", "花具五枚长距，雄蕊多数"),
    "Desert-rose": ("沙漠玫瑰", "夹竹桃科", "花冠五裂，具膨大肉质茎"),
    "Tree Mallow": ("花葵（木锦葵）", "锦葵科", "花五瓣，具副萼，雄蕊单体"),
    "Magnolia": ("玉兰", "木兰科", "花大型，花被片多枚"),
    "Cyclamen": ("仙客来", "报春花科", "花冠合瓣五裂，花瓣反卷"),
    "Watercress": ("豆瓣菜（西洋菜）", "十字花科", "花瓣四枚排成十字形"),
    "Canna Lily": ("美人蕉", "美人蕉科", "花不对称，雄蕊花瓣状"),
    "Hippeastrum": ("朱顶红", "石蒜科", "花被六枚，花葶中空粗壮"),
    "Bee Balm": ("美国薄荷", "唇形科", "花冠二唇形，茎四棱"),
    "Ball Moss": ("球松萝铁兰", "凤梨科", "叶莲座状，植株附生"),
    "Foxglove": ("毛地黄", "车前科", "总状花序，花冠筒状钟形"),
    "Bougainvillea": ("叶子花（三角梅）", "紫茉莉科", "苞片色彩艳丽，聚伞花序"),
    "Camellia": ("山茶花", "山茶科", "花五瓣，雄蕊多数，革质叶"),
    "Mallow": ("锦葵", "锦葵科", "花五瓣，具副萼，雄蕊单体"),
    "Mexican Petunia": ("翠芦莉（墨西哥矮牵牛）", "爵床科", "花冠合瓣五裂，漏斗状"),
    "Bromelia": ("观赏凤梨", "凤梨科", "叶莲座状排列，花序自中心抽出"),
    "Blanket Flower": ("天人菊", "菊科", "头状花序，舌状花黄红相间"),
    "Trumpet Creeper": ("凌霄花", "紫葳科", "花冠漏斗状，橙红色"),
    "Blackberry Lily": ("射干", "鸢尾科", "花被六枚，具橙色斑点"),
}

assert len(CANONICAL_NAMES) == 102, f"类名表应为 102 项, 实际 {len(CANONICAL_NAMES)}"
assert set(NAME_INFO) == set(CANONICAL_NAMES), (
    f"NAME_INFO 与类名表不一致: 缺 {set(CANONICAL_NAMES) - set(NAME_INFO)}, "
    f"多 {set(NAME_INFO) - set(CANONICAL_NAMES)}"
)

# flower_XXX (imagelabels.mat 标签号) -> (中文名, 英文名, 科, 特征)
SPECIES = {
    f"flower_{i:03d}": (NAME_INFO[name][0], name, NAME_INFO[name][1], NAME_INFO[name][2])
    for i, name in enumerate(CANONICAL_NAMES, start=1)
}

# 个别类名无法可靠定科的说明 (写入工作记录用)
SRC_NOTES = {
    "flower_045": "Bolero Deep Blue 为园艺栽培品种名, 无法确定科属, 按'园艺栽培品种'标注",
    "flower_037": "Cape Flower 在官方类名中无学名对应, 无法可靠定科, 按'园艺栽培品种'标注",
    "flower_002": "Hard-leaved Pocket Orchid 判断为兜兰属(Paphiopedilum), 定兰科; 属级归属置信度中等",
    "flower_027": "Prince of Wales Feathers 常见对应 Amaranthus hypochondriacus, 定苋科; 置信度中等",
}

HABITAT = "常见园艺栽培或野生观赏植物"


def main() -> None:
    parser = argparse.ArgumentParser(description="Flowers102 -> labels.csv")
    parser.add_argument("--all", action="store_true", help="使用全部 8189 张图片 (默认只用官方 train 划分)")
    parser.add_argument("--src", default="data/raw/flowers102", help="Flowers102 整理目录")
    parser.add_argument("--out", default="data/raw/labels.csv")
    args = parser.parse_args()

    src = Path(args.src)
    labels_csv = src / "labels.csv"
    splits_json = src / "splits.json"
    if not labels_csv.is_file() or not splits_json.is_file():
        raise SystemExit(f"[错误] 缺少 {labels_csv} 或 {splits_json}, 请先运行 download_flowers102.py")

    splits = json.loads(splits_json.read_text(encoding="utf-8"))
    train_set = set(splits["train"]) if not args.all else None

    with labels_csv.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out_rows, skipped_unknown = [], set()
    for r in rows:
        name = Path(r["image"]).name  # e.g. image_00001.jpg
        if train_set is not None and name not in train_set:
            continue
        cls = r["species_en"].strip()
        if cls not in SPECIES:
            skipped_unknown.add(cls)
            continue
        zh, en, family, features = SPECIES[cls]
        out_rows.append(
            {
                "image": f"flowers102/images/{name}",
                "species_zh": zh,
                "species_en": en,
                "family": family,
                "features": features,
                "habitat": HABITAT,
            }
        )

    if skipped_unknown:
        print(f"[警告] 未知类别 (已跳过): {sorted(skipped_unknown)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "species_zh", "species_en", "family", "features", "habitat"])
        w.writeheader()
        w.writerows(out_rows)

    mode = "官方 train 划分" if train_set is not None else "全部图片"
    n_cls = len({r["species_zh"] for r in out_rows})
    print(f"已生成 {len(out_rows)} 行 ({mode}, {n_cls} 类) -> {out}")
    print(f"定科置信度说明: {len(SRC_NOTES)} 条 (见脚本顶部 SRC_NOTES)")


if __name__ == "__main__":
    main()
