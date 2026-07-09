# -*- coding: utf-8 -*-
"""
Generate a store-SKU-day inventory forecasting dataset.

Default scale:
  140 stores x 5000 SKUs x 730 days = 511,000,000 store-SKU-day points

The full daily data is stored as dense .npy tensors instead of a giant CSV:
  - sales_qty_uint16.npy   shape=(stores, skus, days)
  - stock_qty_uint16.npy   shape=(stores, skus, days)
  - inbound_qty_uint16.npy shape=(stores, skus, days)
  - stockout_uint8.npy     shape=(stores, skus, days)

Master data and sample long-form Parquet files are written beside the tensors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR


DEFAULT_OUT_DIR = Path(BASE_DIR) / "data" / "store_sku_140x5000_2y"

CATEGORY_WEIGHTS = {
    "化学药制剂": 0.36,
    "中成药": 0.20,
    "保健食品": 0.12,
    "中药饮片": 0.10,
    "医疗器械": 0.08,
    "个人护理日化": 0.08,
    "母婴用品": 0.03,
    "食品其他": 0.03,
}

SUBCATEGORY_POOL = {
    "化学药制剂": [
        "抗感染类",
        "呼吸系统类",
        "消化系统类",
        "心脑血管类",
        "解热镇痛类",
        "抗过敏类",
        "糖尿病用药",
        "皮肤外用类",
        "眼耳鼻喉类",
    ],
    "中成药": [
        "感冒清热类",
        "清热解毒类",
        "止咳化痰类",
        "胃肠用药类",
        "妇科用药类",
        "儿科用药类",
        "风湿骨伤类",
    ],
    "保健食品": ["维生素矿物质类", "钙铁锌类", "益生菌类", "蛋白营养类", "鱼油卵磷脂类"],
    "中药饮片": ["单味饮片", "滋补饮片", "花茶饮片"],
    "医疗器械": ["防护用品类", "检测监测类", "护理耗材类", "康复理疗类"],
    "个人护理日化": ["口腔护理类", "皮肤护理类", "洗护清洁类", "消毒杀菌类"],
    "母婴用品": ["婴幼儿营养类", "婴童护理类", "孕产妇用品类"],
    "食品其他": ["功能饮品类", "健康零食类", "冲调食品类"],
}

PRODUCT_TEMPLATES = {
    "抗感染类": [
        ("阿莫西林胶囊", ["0.25g*24粒", "0.5g*20粒"], ["阿莫仙"], "tablet", "stable"),
        ("头孢克洛胶囊", ["0.25g*12粒", "0.25g*24粒"], ["希刻劳"], "tablet", "stable"),
        ("罗红霉素分散片", ["75mg*12片", "150mg*10片"], [], "tablet", "stable"),
        ("左氧氟沙星片", ["0.1g*12片", "0.2g*6片"], [], "tablet", "stable"),
        ("甲硝唑片", ["0.2g*100片", "0.2g*24片"], [], "tablet", "stable"),
    ],
    "呼吸系统类": [
        ("复方氨酚烷胺胶囊", ["10粒", "12粒"], ["感康", "快克"], "capsule", "winter"),
        ("布洛芬缓释胶囊", ["0.3g*20粒", "0.3g*24粒"], ["芬必得"], "capsule", "winter"),
        ("氨溴索口服溶液", ["100ml", "120ml"], [], "liquid", "winter"),
        ("右美沙芬愈创甘油醚糖浆", ["100ml", "120ml"], [], "liquid", "winter"),
        ("小儿氨酚黄那敏颗粒", ["6g*10袋", "4g*12袋"], ["护彤"], "granule", "winter"),
    ],
    "消化系统类": [
        ("蒙脱石散", ["3g*10袋", "3g*15袋"], ["思密达"], "powder", "holiday"),
        ("多潘立酮片", ["10mg*30片", "10mg*42片"], ["吗丁啉"], "tablet", "holiday"),
        ("奥美拉唑肠溶胶囊", ["20mg*14粒", "20mg*28粒"], [], "capsule", "stable"),
        ("乳酸菌素片", ["0.4g*36片", "0.4g*48片"], [], "tablet", "holiday"),
        ("健胃消食片", ["0.5g*36片", "0.5g*48片"], ["江中"], "tablet", "holiday"),
    ],
    "心脑血管类": [
        ("硝苯地平控释片", ["30mg*7片", "30mg*14片"], ["拜新同"], "tablet", "chronic"),
        ("苯磺酸氨氯地平片", ["5mg*7片", "5mg*14片"], ["络活喜"], "tablet", "chronic"),
        ("阿托伐他汀钙片", ["20mg*7片", "20mg*14片"], ["立普妥"], "tablet", "chronic"),
        ("厄贝沙坦片", ["150mg*7片", "150mg*14片"], [], "tablet", "chronic"),
        ("复方丹参滴丸", ["27mg*180丸", "27mg*270丸"], [], "pill", "chronic"),
    ],
    "解热镇痛类": [
        ("对乙酰氨基酚片", ["0.5g*12片", "0.5g*24片"], ["泰诺林"], "tablet", "winter"),
        ("双氯芬酸钠缓释片", ["75mg*10片", "75mg*20片"], ["扶他林"], "tablet", "stable"),
        ("洛索洛芬钠片", ["60mg*12片", "60mg*20片"], [], "tablet", "stable"),
        ("布洛芬混悬液", ["100ml", "60ml"], ["美林"], "liquid", "winter"),
    ],
    "抗过敏类": [
        ("氯雷他定片", ["10mg*6片", "10mg*12片"], ["开瑞坦"], "tablet", "spring"),
        ("盐酸西替利嗪片", ["10mg*12片", "10mg*24片"], [], "tablet", "spring"),
        ("依巴斯汀片", ["10mg*10片", "10mg*14片"], [], "tablet", "spring"),
        ("糠酸莫米松鼻喷雾剂", ["50ug*60揿", "50ug*120揿"], ["内舒拿"], "spray", "spring"),
    ],
    "糖尿病用药": [
        ("盐酸二甲双胍片", ["0.5g*48片", "0.5g*60片"], [], "tablet", "chronic"),
        ("阿卡波糖片", ["50mg*30片", "50mg*45片"], ["拜唐苹"], "tablet", "chronic"),
        ("格列美脲片", ["2mg*30片", "2mg*60片"], [], "tablet", "chronic"),
        ("瑞格列奈片", ["1mg*30片", "1mg*60片"], [], "tablet", "chronic"),
    ],
    "皮肤外用类": [
        ("莫匹罗星软膏", ["5g", "10g"], ["百多邦"], "ointment", "summer"),
        ("炉甘石洗剂", ["100ml", "150ml"], [], "liquid", "summer"),
        ("复方酮康唑软膏", ["7g", "15g"], [], "ointment", "summer"),
        ("红霉素软膏", ["10g", "20g"], [], "ointment", "stable"),
    ],
    "眼耳鼻喉类": [
        ("玻璃酸钠滴眼液", ["5ml", "10ml"], [], "drop", "stable"),
        ("左氧氟沙星滴眼液", ["5ml", "8ml"], [], "drop", "stable"),
        ("西瓜霜润喉片", ["24片", "36片"], [], "tablet", "winter"),
        ("开喉剑喷雾剂", ["20ml", "30ml"], [], "spray", "winter"),
    ],
    "感冒清热类": [
        ("感冒灵颗粒", ["10g*9袋", "10g*12袋"], ["999"], "granule", "winter"),
        ("连花清瘟胶囊", ["0.35g*24粒", "0.35g*36粒"], [], "capsule", "winter"),
        ("板蓝根颗粒", ["10g*20袋", "10g*30袋"], [], "granule", "winter"),
        ("藿香正气口服液", ["10ml*10支", "10ml*12支"], [], "liquid", "summer"),
    ],
    "清热解毒类": [
        ("蒲地蓝消炎口服液", ["10ml*10支", "10ml*12支"], [], "liquid", "winter"),
        ("双黄连口服液", ["10ml*10支", "10ml*12支"], [], "liquid", "winter"),
        ("牛黄解毒片", ["24片", "36片"], [], "tablet", "stable"),
        ("清火栀麦片", ["24片", "36片"], [], "tablet", "stable"),
    ],
    "止咳化痰类": [
        ("川贝枇杷膏", ["150ml", "300ml"], [], "paste", "winter"),
        ("急支糖浆", ["100ml", "200ml"], [], "liquid", "winter"),
        ("强力枇杷露", ["100ml", "120ml"], [], "liquid", "winter"),
        ("蛇胆川贝液", ["10ml*6支", "10ml*10支"], [], "liquid", "winter"),
    ],
    "胃肠用药类": [
        ("保和丸", ["9g*10丸", "6g*10袋"], [], "pill", "holiday"),
        ("香砂养胃丸", ["200丸", "360丸"], [], "pill", "stable"),
        ("肠炎宁片", ["0.42g*24片", "0.42g*36片"], [], "tablet", "summer"),
        ("四磨汤口服液", ["10ml*10支", "10ml*12支"], [], "liquid", "holiday"),
    ],
    "妇科用药类": [
        ("益母草颗粒", ["15g*10袋", "15g*12袋"], [], "granule", "stable"),
        ("乌鸡白凤丸", ["9g*10丸", "6g*12袋"], [], "pill", "stable"),
        ("妇科千金片", ["0.32g*36片", "0.32g*48片"], [], "tablet", "stable"),
        ("克霉唑阴道片", ["0.5g*1片", "0.5g*2片"], [], "tablet", "stable"),
    ],
    "儿科用药类": [
        ("小儿豉翘清热颗粒", ["2g*6袋", "2g*9袋"], [], "granule", "winter"),
        ("小儿肺热咳喘口服液", ["10ml*6支", "10ml*10支"], [], "liquid", "winter"),
        ("醒脾养儿颗粒", ["2g*12袋", "2g*18袋"], [], "granule", "stable"),
        ("小儿七星茶颗粒", ["7g*10袋", "7g*12袋"], [], "granule", "holiday"),
    ],
    "风湿骨伤类": [
        ("云南白药气雾剂", ["85g+30g", "50g+60g"], ["云南白药"], "spray", "stable"),
        ("活血止痛膏", ["7cm*10cm*5贴", "7cm*10cm*10贴"], [], "patch", "stable"),
        ("麝香壮骨膏", ["7cm*10cm*10贴", "8cm*13cm*8贴"], [], "patch", "stable"),
        ("跌打丸", ["3g*10丸", "3g*12丸"], [], "pill", "stable"),
    ],
    "维生素矿物质类": [
        ("维生素C片", ["100mg*100片", "100mg*200片"], [], "tablet", "stable"),
        ("复合维生素B片", ["100片", "200片"], [], "tablet", "stable"),
        ("多种维生素矿物质片", ["60片", "100片"], ["善存"], "tablet", "stable"),
        ("叶酸片", ["0.4mg*31片", "0.4mg*93片"], ["斯利安"], "tablet", "stable"),
    ],
    "钙铁锌类": [
        ("碳酸钙D3片", ["60片", "100片"], ["钙尔奇"], "tablet", "stable"),
        ("葡萄糖酸锌口服液", ["10ml*12支", "10ml*24支"], [], "liquid", "stable"),
        ("乳酸钙颗粒", ["5g*12袋", "5g*24袋"], [], "granule", "stable"),
        ("右旋糖酐铁口服液", ["10ml*10支", "10ml*20支"], [], "liquid", "stable"),
    ],
    "益生菌类": [
        ("益生菌冻干粉", ["2g*10袋", "2g*20袋"], [], "powder", "holiday"),
        ("双歧杆菌三联活菌胶囊", ["24粒", "36粒"], [], "capsule", "holiday"),
        ("乳酸菌素颗粒", ["1g*12袋", "1g*20袋"], [], "granule", "holiday"),
    ],
    "蛋白营养类": [
        ("蛋白粉", ["400g", "900g"], [], "powder", "stable"),
        ("氨糖软骨素钙片", ["60片", "120片"], [], "tablet", "stable"),
        ("胶原蛋白肽饮品", ["50ml*6瓶", "50ml*10瓶"], [], "liquid", "stable"),
    ],
    "鱼油卵磷脂类": [
        ("深海鱼油软胶囊", ["100粒", "200粒"], [], "capsule", "stable"),
        ("大豆卵磷脂软胶囊", ["100粒", "200粒"], [], "capsule", "stable"),
        ("辅酶Q10软胶囊", ["30粒", "60粒"], [], "capsule", "stable"),
    ],
    "单味饮片": [
        ("黄芪饮片", ["100g", "250g"], [], "herb", "winter"),
        ("党参饮片", ["100g", "250g"], [], "herb", "winter"),
        ("枸杞子", ["150g", "250g"], [], "herb", "stable"),
        ("金银花", ["50g", "100g"], [], "herb", "summer"),
        ("菊花", ["50g", "100g"], [], "herb", "summer"),
    ],
    "滋补饮片": [
        ("西洋参片", ["50g", "100g"], [], "herb", "holiday"),
        ("当归片", ["100g", "250g"], [], "herb", "winter"),
        ("三七粉", ["100g", "250g"], [], "herb", "stable"),
        ("铁皮石斛", ["50g", "100g"], [], "herb", "holiday"),
    ],
    "花茶饮片": [
        ("玫瑰花茶", ["50g", "100g"], [], "tea", "spring"),
        ("胖大海", ["50g", "100g"], [], "tea", "winter"),
        ("罗汉果", ["6个", "12个"], [], "tea", "winter"),
        ("决明子", ["100g", "250g"], [], "tea", "summer"),
    ],
    "防护用品类": [
        ("医用外科口罩", ["10只装", "50只装"], [], "device", "winter"),
        ("医用防护口罩", ["10只装", "20只装"], [], "device", "winter"),
        ("一次性医用手套", ["50只装", "100只装"], [], "device", "stable"),
        ("酒精消毒湿巾", ["20片", "80片"], [], "device", "stable"),
    ],
    "检测监测类": [
        ("电子血压计", ["臂式", "腕式"], [], "device", "stable"),
        ("血糖仪", ["套装", "便携款"], [], "device", "chronic"),
        ("血糖试纸", ["25片", "50片"], [], "device", "chronic"),
        ("体温计", ["电子款", "红外款"], [], "device", "winter"),
    ],
    "护理耗材类": [
        ("创可贴", ["20片", "100片"], ["邦迪"], "device", "stable"),
        ("医用棉签", ["100支", "200支"], [], "device", "stable"),
        ("医用纱布块", ["10片", "20片"], [], "device", "stable"),
        ("碘伏消毒液", ["100ml", "500ml"], [], "device", "stable"),
    ],
    "康复理疗类": [
        ("护腰带", ["M码", "L码"], [], "device", "stable"),
        ("颈椎牵引器", ["家用款", "升级款"], [], "device", "stable"),
        ("热敷贴", ["5贴", "10贴"], [], "device", "winter"),
        ("艾灸贴", ["10贴", "20贴"], [], "device", "winter"),
    ],
    "口腔护理类": [
        ("牙线棒", ["50支", "100支"], [], "daily", "stable"),
        ("漱口水", ["250ml", "500ml"], [], "daily", "stable"),
        ("抗敏感牙膏", ["100g", "120g"], ["舒适达"], "daily", "stable"),
        ("儿童含氟牙膏", ["60g", "90g"], [], "daily", "stable"),
    ],
    "皮肤护理类": [
        ("医用冷敷贴", ["5片", "10片"], [], "daily", "summer"),
        ("维生素E乳", ["100ml", "200ml"], [], "daily", "winter"),
        ("尿素维E乳膏", ["50g", "100g"], [], "daily", "winter"),
        ("防晒乳", ["50ml", "100ml"], [], "daily", "summer"),
    ],
    "洗护清洁类": [
        ("除菌洗手液", ["300ml", "500ml"], [], "daily", "stable"),
        ("头皮护理洗发水", ["200ml", "400ml"], [], "daily", "stable"),
        ("沐浴露", ["300ml", "500ml"], [], "daily", "stable"),
        ("女性护理液", ["200ml", "300ml"], [], "daily", "stable"),
    ],
    "消毒杀菌类": [
        ("75%酒精消毒液", ["100ml", "500ml"], [], "daily", "stable"),
        ("免洗手消毒凝胶", ["100ml", "500ml"], [], "daily", "stable"),
        ("含氯消毒片", ["100片", "200片"], [], "daily", "stable"),
        ("碘伏棉签", ["20支", "50支"], [], "daily", "stable"),
    ],
    "婴幼儿营养类": [
        ("婴幼儿益生菌粉", ["1.5g*10袋", "1.5g*20袋"], [], "baby", "stable"),
        ("DHA藻油软胶囊", ["30粒", "60粒"], [], "baby", "stable"),
        ("维生素AD滴剂", ["30粒", "60粒"], ["伊可新"], "baby", "stable"),
        ("婴幼儿钙颗粒", ["2g*15袋", "2g*30袋"], [], "baby", "stable"),
    ],
    "婴童护理类": [
        ("婴儿湿巾", ["80抽", "120抽"], [], "baby", "stable"),
        ("婴儿护臀膏", ["30g", "50g"], [], "baby", "stable"),
        ("儿童退热贴", ["4贴", "8贴"], [], "baby", "winter"),
        ("儿童口罩", ["10只装", "30只装"], [], "baby", "winter"),
    ],
    "孕产妇用品类": [
        ("孕妇复合维生素片", ["30片", "60片"], [], "baby", "stable"),
        ("产妇护理垫", ["10片", "20片"], [], "baby", "stable"),
        ("防溢乳垫", ["60片", "120片"], [], "baby", "stable"),
        ("孕妇钙片", ["60片", "120片"], [], "baby", "stable"),
    ],
    "功能饮品类": [
        ("葡萄糖电解质饮品", ["500ml", "1L"], [], "food", "summer"),
        ("维生素功能饮料", ["250ml*6瓶", "250ml*12瓶"], [], "food", "summer"),
        ("益生菌饮品", ["100ml*5瓶", "100ml*10瓶"], [], "food", "holiday"),
    ],
    "健康零食类": [
        ("无糖薄荷糖", ["20g", "40g"], [], "food", "stable"),
        ("低糖龟苓膏", ["200g*6杯", "200g*12杯"], [], "food", "summer"),
        ("黑芝麻丸", ["100g", "250g"], [], "food", "stable"),
    ],
    "冲调食品类": [
        ("燕麦片", ["500g", "1kg"], [], "food", "stable"),
        ("无糖藕粉", ["300g", "600g"], [], "food", "stable"),
        ("黑芝麻糊", ["300g", "600g"], [], "food", "winter"),
    ],
}

MANUFACTURERS = [
    "华润三九医药股份有限公司",
    "云南白药集团股份有限公司",
    "哈药集团制药六厂",
    "扬子江药业集团有限公司",
    "太极集团重庆涪陵制药厂有限公司",
    "广州白云山医药集团股份有限公司",
    "修正药业集团股份有限公司",
    "北京同仁堂科技发展股份有限公司",
    "上海上药信谊药厂有限公司",
    "石药集团欧意药业有限公司",
    "国药集团工业有限公司",
    "贵州百灵企业集团制药股份有限公司",
    "仁和药业股份有限公司",
    "葵花药业集团股份有限公司",
    "浙江康恩贝制药股份有限公司",
]

CITY_DISTRICTS = {
    "贵阳": ["云岩区", "南明区", "观山湖区", "花溪区", "白云区", "乌当区"],
    "遵义": ["红花岗区", "汇川区", "播州区", "仁怀市", "桐梓县"],
    "六盘水": ["钟山区", "水城区", "盘州市"],
    "安顺": ["西秀区", "平坝区", "普定县"],
    "毕节": ["七星关区", "大方县", "黔西市"],
    "铜仁": ["碧江区", "万山区", "松桃县"],
    "黔东南": ["凯里市", "黄平县", "镇远县"],
    "黔南": ["都匀市", "福泉市", "龙里县"],
    "黔西南": ["兴义市", "兴仁市", "安龙县"],
    "重庆": ["渝中区", "江北区", "沙坪坝区", "九龙坡区"],
    "成都": ["锦江区", "武侯区", "成华区", "金牛区"],
    "昆明": ["五华区", "盘龙区", "官渡区", "西山区"],
}

BUSINESS_DISTRICTS = ["社区店", "医院周边", "商圈店", "学校周边", "交通枢纽", "乡镇中心"]
STORE_GRADES = ["A", "B", "C", "D"]
GRADE_PROBS = [0.20, 0.35, 0.32, 0.13]
GRADE_SCALE = {"A": (1.45, 2.20), "B": (1.05, 1.55), "C": (0.70, 1.15), "D": (0.40, 0.80)}
GRADE_COVERAGE = {"A": 0.96, "B": 0.84, "C": 0.68, "D": 0.52}

SEASON_ID = {"stable": 0, "winter": 1, "summer": 2, "holiday": 3, "spring": 4, "chronic": 5}
RX_TYPES = ["OTC甲类", "OTC乙类", "处方药", "双跨", "非药品"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate store-SKU-day simulated inventory data.")
    parser.add_argument("--stores", type=int, default=140)
    parser.add_argument("--skus", type=int, default=5000)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--store-chunk", type=int, default=5)
    parser.add_argument("--sample-stores", type=int, default=8)
    parser.add_argument("--sample-skus", type=int, default=300)
    parser.add_argument("--sample-days", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scaled_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {k: total * v for k, v in weights.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def short_manufacturer(name: str) -> str:
    return (
        name.replace("股份有限公司", "")
        .replace("有限责任公司", "")
        .replace("有限公司", "")
        .replace("集团", "")
    )


def ean13_from_index(idx: int) -> str:
    base = f"690{idx:09d}"[:12]
    digits = [int(x) for x in base]
    odd = sum(digits[0::2])
    even = sum(digits[1::2])
    check = (10 - ((odd + even * 3) % 10)) % 10
    return base + str(check)


def approval_no(rng: np.random.Generator, category: str) -> str:
    if category in {"医疗器械", "个人护理日化", "母婴用品", "食品其他", "保健食品"}:
        prefix = rng.choice(["械注准", "食健备", "消证字", "妆备字"])
    else:
        prefix = rng.choice(["国药准字H", "国药准字Z", "国药准字S"])
    return f"{prefix}{rng.integers(2000, 2026)}{rng.integers(1000, 9999)}"


def dosage_to_rx(category: str, sub_category: str, dosage_form: str, rng: np.random.Generator) -> str:
    if category in {"医疗器械", "个人护理日化", "母婴用品", "食品其他"}:
        return "非药品"
    if category == "保健食品":
        return "OTC乙类"
    if sub_category in {"心脑血管类", "糖尿病用药", "抗感染类"}:
        return rng.choice(["处方药", "双跨"], p=[0.78, 0.22])
    return rng.choice(["OTC甲类", "OTC乙类", "双跨"], p=[0.48, 0.42, 0.10])


def price_range(category: str) -> tuple[float, float]:
    return {
        "化学药制剂": (8.0, 85.0),
        "中成药": (10.0, 98.0),
        "保健食品": (35.0, 298.0),
        "中药饮片": (12.0, 260.0),
        "医疗器械": (5.0, 420.0),
        "个人护理日化": (8.0, 168.0),
        "母婴用品": (18.0, 260.0),
        "食品其他": (6.0, 138.0),
    }.get(category, (8.0, 80.0))


def base_demand(category: str, sub_category: str, season: str, rng: np.random.Generator) -> float:
    base_by_category = {
        "化学药制剂": (5.0, 24.0),
        "中成药": (4.0, 20.0),
        "保健食品": (2.0, 10.0),
        "中药饮片": (1.0, 8.0),
        "医疗器械": (2.0, 14.0),
        "个人护理日化": (2.0, 12.0),
        "母婴用品": (1.0, 7.0),
        "食品其他": (1.0, 8.0),
    }
    lo, hi = base_by_category.get(category, (2.0, 12.0))
    value = float(rng.lognormal(mean=np.log((lo + hi) / 2), sigma=0.45))
    value = float(np.clip(value, lo, hi * 1.6))
    if season == "chronic":
        value *= 0.85
    if sub_category in {"呼吸系统类", "感冒清热类", "止咳化痰类"}:
        value *= 1.15
    return value


def generate_product_master(n_skus: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    category_counts = scaled_counts(n_skus, CATEGORY_WEIGHTS)
    rows = []
    sku_idx = 1

    for category, cat_count in category_counts.items():
        subcats = SUBCATEGORY_POOL[category]
        sub_counts = scaled_counts(cat_count, {s: 1 / len(subcats) for s in subcats})
        for sub_category, sub_count in sub_counts.items():
            templates = PRODUCT_TEMPLATES[sub_category]
            for local_idx in range(sub_count):
                generic_name, specs, brands, dosage_form, season = templates[local_idx % len(templates)]
                spec = rng.choice(specs)
                brand = rng.choice(brands) if brands and rng.random() < 0.7 else ""
                manufacturer = rng.choice(MANUFACTURERS)
                sku_id = f"SKU{sku_idx:05d}"
                normalized_sku_id = sku_id
                display_core = f"{brand} {generic_name}".strip()
                display_name = f"{display_core} {spec}"
                erp_name = f"{display_core}{spec} {short_manufacturer(manufacturer)}"
                p_min, p_max = price_range(category)
                base_price = float(np.round(rng.uniform(p_min, p_max), 2))
                demand = base_demand(category, sub_category, season, rng)
                trend = float(rng.uniform(-0.0015, 0.0045))
                if season in {"winter", "summer", "spring"}:
                    trend += float(rng.uniform(0.0, 0.0015))
                popularity = float(rng.beta(1.4, 3.2))
                if sku_idx <= max(100, n_skus // 20):
                    popularity = float(rng.uniform(0.75, 1.0))
                    demand *= float(rng.uniform(1.25, 2.2))

                rows.append(
                    {
                        "sku_id": sku_id,
                        "normalized_sku_id": normalized_sku_id,
                        "barcode": ean13_from_index(sku_idx),
                        "display_name": display_name,
                        "erp_name": erp_name,
                        "generic_name": generic_name,
                        "brand_name": brand,
                        "manufacturer": manufacturer,
                        "spec": spec,
                        "dosage_form": dosage_form,
                        "approval_no": approval_no(rng, category),
                        "category": category,
                        "sub_category": sub_category,
                        "rx_type": dosage_to_rx(category, sub_category, dosage_form, rng),
                        "is_medical_insurance": int(
                            category in {"化学药制剂", "中成药", "中药饮片"} and rng.random() < 0.72
                        ),
                        "base_price": base_price,
                        "base_daily_demand": round(demand, 3),
                        "trend_per_day": round(trend, 6),
                        "season_type": season,
                        "season_id": SEASON_ID[season],
                        "popularity": round(popularity, 4),
                    }
                )
                sku_idx += 1

    df = pd.DataFrame(rows)
    return df.iloc[:n_skus].reset_index(drop=True)


def generate_product_aliases(products: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 10)
    rows = []
    for _, p in products.iterrows():
        if rng.random() > 0.28:
            continue
        aliases = [
            ("包装名", p["display_name"]),
            ("通用名规格", f"{p['generic_name']} {p['spec']}"),
            ("ERP简写", str(p["erp_name"]).replace(" ", "")),
        ]
        if p["brand_name"]:
            aliases.append(("品牌简写", f"{p['brand_name']} {p['spec']}"))
        for alias_type, alias_name in aliases[: int(rng.integers(1, min(4, len(aliases)) + 1))]:
            rows.append(
                {
                    "sku_id": p["sku_id"],
                    "normalized_sku_id": p["normalized_sku_id"],
                    "alias_type": alias_type,
                    "alias_name": alias_name,
                }
            )
    return pd.DataFrame(rows)


def generate_store_master(n_stores: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    cities = list(CITY_DISTRICTS)
    city_weights = np.array([1.8 if c in {"贵阳", "遵义"} else 1.0 for c in cities], dtype=float)
    city_weights = city_weights / city_weights.sum()
    rows = []
    for idx in range(1, n_stores + 1):
        city = rng.choice(cities, p=city_weights)
        district = rng.choice(CITY_DISTRICTS[city])
        grade = rng.choice(STORE_GRADES, p=GRADE_PROBS)
        scale_lo, scale_hi = GRADE_SCALE[grade]
        traffic = float(np.round(rng.uniform(scale_lo, scale_hi), 3))
        business = rng.choice(BUSINESS_DISTRICTS, p=[0.34, 0.19, 0.18, 0.08, 0.07, 0.14])
        area = int(np.round(rng.uniform(80, 260) * traffic))
        store_id = f"STORE{idx:03d}"
        rows.append(
            {
                "store_id": store_id,
                "store_name": f"{city}{district}{idx:03d}店",
                "province_region": "西南区域",
                "city": city,
                "district": district,
                "store_grade": grade,
                "business_district": business,
                "area_sqm": area,
                "traffic_index": traffic,
                "is_medical_insurance_store": int(rng.random() < (0.92 if grade in {"A", "B"} else 0.72)),
                "warehouse_id": f"WH{int(rng.integers(1, 7)):02d}",
                "open_days_before_start": int(rng.integers(120, 3000)),
            }
        )
    return pd.DataFrame(rows)


def build_calendar(start_date: str, days: int) -> pd.DataFrame:
    start = date.fromisoformat(start_date)
    dates = [start + timedelta(days=i) for i in range(days)]
    df = pd.DataFrame({"date": [d.isoformat() for d in dates]})
    dt = pd.to_datetime(df["date"])
    df["day_index"] = np.arange(days)
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_new_year"] = ((dt.dt.month == 1) & (dt.dt.day <= 3)).astype(int)
    df["is_spring_festival_window"] = (
        ((dt.dt.month == 1) & (dt.dt.day >= 24)) | ((dt.dt.month == 2) & (dt.dt.day <= 12))
    ).astype(int)
    df["is_national_day_window"] = ((dt.dt.month == 10) & (dt.dt.day <= 7)).astype(int)
    df["is_year_end_window"] = ((dt.dt.month == 12) & (dt.dt.day >= 20)).astype(int)
    return df


def season_matrix(products: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    months = calendar["month"].to_numpy()
    weekend = calendar["is_weekend"].to_numpy(dtype=np.float32)
    holiday = (
        calendar["is_new_year"].to_numpy()
        | calendar["is_spring_festival_window"].to_numpy()
        | calendar["is_national_day_window"].to_numpy()
        | calendar["is_year_end_window"].to_numpy()
    ).astype(bool)
    days = calendar["day_index"].to_numpy(dtype=np.float32)
    p = len(products)
    d = len(calendar)
    mat = np.ones((p, d), dtype=np.float32)

    season_types = products["season_type"].to_numpy()
    weekend_factor = np.where(products["category"].isin(["个人护理日化", "食品其他", "母婴用品"]), 1.12, 1.06)
    for sid, season in enumerate(season_types):
        if season == "winter":
            mat[sid, np.isin(months, [1, 2, 3, 11, 12])] *= 1.45
            mat[sid, np.isin(months, [7, 8])] *= 0.78
        elif season == "summer":
            mat[sid, np.isin(months, [6, 7, 8])] *= 1.38
            mat[sid, np.isin(months, [1, 2, 12])] *= 0.88
        elif season == "holiday":
            mat[sid, holiday] *= 1.55
            mat[sid, np.isin(months, [1, 2, 10, 12])] *= 1.10
        elif season == "spring":
            mat[sid, np.isin(months, [3, 4, 5])] *= 1.42
            mat[sid, np.isin(months, [10, 11])] *= 1.12
        elif season == "chronic":
            mat[sid, :] *= 0.96
        mat[sid, :] *= 1.0 + weekend * float(weekend_factor[sid] - 1.0)

    trend = products["trend_per_day"].to_numpy(dtype=np.float32)[:, None]
    mat *= np.clip(1.0 + trend * days[None, :], 0.50, 2.25)
    return mat


def generate_assortment(products: pd.DataFrame, stores: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 2)
    n_stores = len(stores)
    n_skus = len(products)
    assortment = np.zeros((n_stores, n_skus), dtype=np.uint8)
    popularity = products["popularity"].to_numpy(dtype=np.float32)
    category = products["category"].to_numpy()
    for s_idx, store in stores.iterrows():
        coverage = GRADE_COVERAGE[str(store["store_grade"])]
        prob = coverage * (0.42 + 0.78 * popularity)
        if store["business_district"] == "医院周边":
            prob = np.where(np.isin(category, ["化学药制剂", "中成药", "医疗器械"]), prob * 1.12, prob * 0.92)
        elif store["business_district"] == "学校周边":
            prob = np.where(np.isin(category, ["母婴用品", "个人护理日化", "食品其他"]), prob * 1.10, prob)
        elif store["business_district"] == "乡镇中心":
            prob = np.where(np.isin(category, ["中药饮片", "中成药"]), prob * 1.10, prob * 0.95)
        prob[: max(50, n_skus // 25)] = 0.995
        assortment[s_idx] = (rng.random(n_skus) < np.clip(prob, 0.05, 0.995)).astype(np.uint8)
    return assortment


def open_tensor(path: Path, shape: tuple[int, int, int], dtype: str):
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(dtype), shape=shape)


def generate_tensors(
    out_dir: Path,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    assortment: np.ndarray,
    seed: int,
    store_chunk: int,
) -> dict:
    rng = np.random.default_rng(seed + 3)
    n_stores, n_skus = assortment.shape
    n_days = len(calendar)
    shape = (n_stores, n_skus, n_days)

    sales_mm = open_tensor(out_dir / "sales_qty_uint16.npy", shape, "uint16")
    stock_mm = open_tensor(out_dir / "stock_qty_uint16.npy", shape, "uint16")
    inbound_mm = open_tensor(out_dir / "inbound_qty_uint16.npy", shape, "uint16")
    stockout_mm = open_tensor(out_dir / "stockout_uint8.npy", shape, "uint8")

    product_base = products["base_daily_demand"].to_numpy(dtype=np.float32)
    product_popularity = products["popularity"].to_numpy(dtype=np.float32)
    season = season_matrix(products, calendar)
    base_product_time = product_base[:, None] * season
    traffic = stores["traffic_index"].to_numpy(dtype=np.float32)
    grade = stores["store_grade"].to_numpy()
    business = stores["business_district"].to_numpy()

    category = products["category"].to_numpy()
    n_chunks = math.ceil(n_stores / store_chunk)
    started = time.time()
    total_sales = 0
    total_stockouts = 0
    active_pairs = int(assortment.sum())

    for chunk_idx, s0 in enumerate(range(0, n_stores, store_chunk), 1):
        s1 = min(s0 + store_chunk, n_stores)
        size = s1 - s0
        active = assortment[s0:s1].astype(np.float32)
        store_scale = traffic[s0:s1].astype(np.float32)
        store_noise = rng.normal(1.0, 0.045, (size, 1, n_days)).astype(np.float32)

        expected = base_product_time[None, :, :] * store_scale[:, None, None] * store_noise
        for local_s, store_type in enumerate(business[s0:s1]):
            if store_type == "医院周边":
                expected[local_s, np.isin(category, ["化学药制剂", "中成药", "医疗器械"]), :] *= 1.16
            elif store_type == "学校周边":
                expected[local_s, np.isin(category, ["母婴用品", "个人护理日化", "食品其他"]), :] *= 1.12
            elif store_type == "乡镇中心":
                expected[local_s, np.isin(category, ["中药饮片", "中成药"]), :] *= 1.12
        expected *= active[:, :, None]
        expected = np.clip(expected, 0, 350)

        demand = rng.poisson(expected).astype(np.uint16)
        avg_daily = np.maximum(expected.mean(axis=2), 0.05) * active
        if np.any(grade[s0:s1] == "A"):
            pass
        target_days = np.array([24 if g == "A" else 21 if g == "B" else 18 if g == "C" else 15 for g in grade[s0:s1]])
        reorder_days = np.array([9 if g in {"A", "B"} else 7 for g in grade[s0:s1]])
        stock = np.ceil(avg_daily * target_days[:, None] * rng.uniform(0.82, 1.18, (size, n_skus))).astype(np.float32)
        stock *= active
        lead_time = np.array([2 if g == "A" else 3 if g == "B" else 4 if g == "C" else 5 for g in grade[s0:s1]])
        max_lead = int(lead_time.max()) + 1
        pipeline = [np.zeros((size, n_skus), dtype=np.float32) for _ in range(max_lead)]
        fill_rate = rng.uniform(0.82, 1.0, (size, n_skus)).astype(np.float32)
        fill_rate *= np.where(product_popularity[None, :] > 0.75, 1.0, rng.uniform(0.90, 1.0, (size, n_skus)))
        fill_rate = np.clip(fill_rate, 0.70, 1.0) * active

        chunk_sales = 0
        chunk_stockouts = 0
        for day_idx in range(n_days):
            arriving = pipeline.pop(0)
            stock += arriving
            pipeline.append(np.zeros((size, n_skus), dtype=np.float32))
            daily_demand = demand[:, :, day_idx].astype(np.float32)
            sold = np.minimum(daily_demand, stock)
            stock -= sold
            stockout = (daily_demand > sold + 0.01) & (active > 0)

            reorder_point = avg_daily * reorder_days[:, None]
            target_stock = avg_daily * target_days[:, None]
            pipeline_qty = np.zeros_like(stock)
            for pending in pipeline:
                pipeline_qty += pending
            need_order = (stock + pipeline_qty) < reorder_point
            order = np.maximum(target_stock - stock - pipeline_qty, 0) * need_order * fill_rate
            # Small stores do not replenish every SKU every day.
            weekday = int(calendar.iloc[day_idx]["day_of_week"])
            order_allowed = np.ones((size, 1), dtype=bool)
            for local_s, g in enumerate(grade[s0:s1]):
                if g in {"C", "D"} and weekday not in {1, 4}:
                    order_allowed[local_s, 0] = False
            order *= order_allowed
            for local_s, lt in enumerate(lead_time):
                pipeline[int(lt)][local_s] += order[local_s]

            sales_u16 = np.clip(np.rint(sold), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            stock_u16 = np.clip(np.rint(stock), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            inbound_u16 = np.clip(np.rint(arriving), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            stockout_u8 = stockout.astype(np.uint8)

            sales_mm[s0:s1, :, day_idx] = sales_u16
            stock_mm[s0:s1, :, day_idx] = stock_u16
            inbound_mm[s0:s1, :, day_idx] = inbound_u16
            stockout_mm[s0:s1, :, day_idx] = stockout_u8

            chunk_sales += int(sales_u16.sum())
            chunk_stockouts += int(stockout_u8.sum())

        total_sales += chunk_sales
        total_stockouts += chunk_stockouts
        elapsed = time.time() - started
        print(
            f"Chunk {chunk_idx:>2}/{n_chunks}: stores {s0 + 1}-{s1}, "
            f"sales={chunk_sales:,}, stockouts={chunk_stockouts:,}, elapsed={elapsed:.1f}s",
            flush=True,
        )

    sales_mm.flush()
    stock_mm.flush()
    inbound_mm.flush()
    stockout_mm.flush()

    return {
        "shape": list(shape),
        "active_store_sku_pairs": active_pairs,
        "total_sales_qty": total_sales,
        "stockout_points": total_stockouts,
        "stockout_rate_on_active_points": round(total_stockouts / max(active_pairs * n_days, 1), 6),
    }


def write_sample_parquet(
    out_dir: Path,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    sample_stores: int,
    sample_skus: int,
    sample_days: int,
) -> dict:
    sales = np.load(out_dir / "sales_qty_uint16.npy", mmap_mode="r")
    stock = np.load(out_dir / "stock_qty_uint16.npy", mmap_mode="r")
    inbound = np.load(out_dir / "inbound_qty_uint16.npy", mmap_mode="r")
    stockout = np.load(out_dir / "stockout_uint8.npy", mmap_mode="r")
    s_n = min(sample_stores, len(stores))
    p_n = min(sample_skus, len(products))
    d_n = min(sample_days, len(calendar))

    rows = []
    date_vals = calendar["date"].iloc[-d_n:].to_numpy()
    for s_idx in range(s_n):
        store_id = stores.iloc[s_idx]["store_id"]
        for p_idx in range(p_n):
            sku_id = products.iloc[p_idx]["sku_id"]
            price = float(products.iloc[p_idx]["base_price"])
            rows.append(
                pd.DataFrame(
                    {
                        "store_id": store_id,
                        "sku_id": sku_id,
                        "date": date_vals,
                        "sales_qty": sales[s_idx, p_idx, -d_n:].astype(np.int32),
                        "stock_qty": stock[s_idx, p_idx, -d_n:].astype(np.int32),
                        "inbound_qty": inbound[s_idx, p_idx, -d_n:].astype(np.int32),
                        "is_stockout": stockout[s_idx, p_idx, -d_n:].astype(np.int8),
                        "price": price,
                    }
                )
            )
    sample = pd.concat(rows, ignore_index=True)
    sample_path = out_dir / "sample_sales_long.parquet"
    sample.to_parquet(sample_path, index=False)
    return {"sample_rows": int(len(sample)), "sample_parquet": str(sample_path)}


def summarize_outputs(
    out_dir: Path,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    tensor_stats: dict,
    sample_stats: dict,
    elapsed: float,
) -> dict:
    sales = np.load(out_dir / "sales_qty_uint16.npy", mmap_mode="r")
    stockout = np.load(out_dir / "stockout_uint8.npy", mmap_mode="r")
    n_stores, n_skus, n_days = sales.shape

    store_sales = sales.sum(axis=(1, 2)).astype(np.int64)
    sku_sales = sales.sum(axis=(0, 2)).astype(np.int64)
    day_sales = sales.sum(axis=(0, 1)).astype(np.int64)
    store_stockouts = stockout.sum(axis=(1, 2)).astype(np.int64)
    sku_stockouts = stockout.sum(axis=(0, 2)).astype(np.int64)

    store_summary = stores[["store_id", "store_name", "city", "store_grade", "business_district"]].copy()
    store_summary["total_sales_qty"] = store_sales
    store_summary["stockout_points"] = store_stockouts
    store_summary.sort_values("total_sales_qty", ascending=False).to_csv(
        out_dir / "store_sales_summary.csv", index=False, encoding="utf-8-sig"
    )

    sku_summary = products[["sku_id", "display_name", "category", "sub_category", "season_type"]].copy()
    sku_summary["total_sales_qty"] = sku_sales
    sku_summary["stockout_points"] = sku_stockouts
    sku_summary.sort_values("total_sales_qty", ascending=False).head(1000).to_csv(
        out_dir / "top_sku_sales_summary.csv", index=False, encoding="utf-8-sig"
    )

    day_summary = pd.DataFrame({"date": calendar["date"], "total_sales_qty": day_sales})
    day_summary.to_csv(out_dir / "daily_sales_summary.csv", index=False, encoding="utf-8-sig")

    category_summary = (
        sku_summary.merge(products[["sku_id", "base_price"]], on="sku_id")
        .groupby("category")
        .agg(skus=("sku_id", "count"), total_sales_qty=("total_sales_qty", "sum"), avg_price=("base_price", "mean"))
        .reset_index()
        .sort_values("total_sales_qty", ascending=False)
    )
    category_summary["avg_price"] = category_summary["avg_price"].round(2)
    category_summary.to_csv(out_dir / "category_sales_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": "store_sku_inventory_simulation",
        "axis_order": ["store", "sku", "day"],
        "stores": int(n_stores),
        "skus": int(n_skus),
        "days": int(n_days),
        "equivalent_long_rows": int(n_stores * n_skus * n_days),
        "date_range": [str(calendar["date"].iloc[0]), str(calendar["date"].iloc[-1])],
        "files": {
            "product_master": "product_master.csv",
            "product_aliases": "product_aliases.csv",
            "store_master": "store_master.csv",
            "calendar": "calendar.csv",
            "assortment": "assortment_uint8.npy",
            "sales_qty": "sales_qty_uint16.npy",
            "stock_qty": "stock_qty_uint16.npy",
            "inbound_qty": "inbound_qty_uint16.npy",
            "stockout": "stockout_uint8.npy",
            "sample_sales_long": "sample_sales_long.parquet",
        },
        "tensor_dtypes": {
            "sales_qty": "uint16",
            "stock_qty": "uint16",
            "inbound_qty": "uint16",
            "stockout": "uint8",
        },
        "tensor_stats": tensor_stats,
        "sample_stats": sample_stats,
        "top_stores": store_summary.sort_values("total_sales_qty", ascending=False)
        .head(5)[["store_id", "store_name", "total_sales_qty"]]
        .to_dict(orient="records"),
        "top_skus": sku_summary.sort_values("total_sales_qty", ascending=False)
        .head(5)[["sku_id", "display_name", "total_sales_qty"]]
        .to_dict(orient="records"),
        "category_sales": category_summary.to_dict(orient="records"),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory already exists and is not empty: {out_dir}. Use --force to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Store-SKU-Day Inventory Dataset Generator")
    print("=" * 72)
    print(f"Output: {out_dir}")
    print(f"Scale: {args.stores} stores x {args.skus} SKUs x {args.days} days")
    print(f"Equivalent long rows: {args.stores * args.skus * args.days:,}")
    print(f"Seed: {args.seed}")

    products = generate_product_master(args.skus, args.seed)
    aliases = generate_product_aliases(products, args.seed)
    stores = generate_store_master(args.stores, args.seed)
    calendar = build_calendar(args.start_date, args.days)
    assortment = generate_assortment(products, stores, args.seed)

    products.to_csv(out_dir / "product_master.csv", index=False, encoding="utf-8-sig")
    aliases.to_csv(out_dir / "product_aliases.csv", index=False, encoding="utf-8-sig")
    stores.to_csv(out_dir / "store_master.csv", index=False, encoding="utf-8-sig")
    calendar.to_csv(out_dir / "calendar.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "assortment_uint8.npy", assortment)

    print(f"Product master: {len(products):,} SKUs")
    print(f"Product aliases: {len(aliases):,} aliases")
    print(f"Store master: {len(stores):,} stores")
    print(f"Active store-SKU pairs: {int(assortment.sum()):,}/{assortment.size:,}")

    tensor_stats = generate_tensors(
        out_dir=out_dir,
        products=products,
        stores=stores,
        calendar=calendar,
        assortment=assortment,
        seed=args.seed,
        store_chunk=max(1, args.store_chunk),
    )
    sample_stats = write_sample_parquet(
        out_dir=out_dir,
        products=products,
        stores=stores,
        calendar=calendar,
        sample_stores=args.sample_stores,
        sample_skus=args.sample_skus,
        sample_days=args.sample_days,
    )
    summary = summarize_outputs(out_dir, products, stores, calendar, tensor_stats, sample_stats, time.time() - started)

    print("\n" + "=" * 72)
    print("Dataset generation complete")
    print("=" * 72)
    print(f"Rows equivalent: {summary['equivalent_long_rows']:,}")
    print(f"Total sales qty: {summary['tensor_stats']['total_sales_qty']:,}")
    print(f"Stockout rate: {summary['tensor_stats']['stockout_rate_on_active_points']:.4%}")
    print(f"Summary: {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
