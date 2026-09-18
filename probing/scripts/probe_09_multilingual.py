#!/usr/bin/env python3
"""Probe 9: Multilingual parity — language capability fingerprint.

Maps Jev's language capabilities across 12 languages to infer training data
distribution and identify the likely base model family.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "09-multilingual.jsonl")

# Same 20 factual statements in 12 languages
# Each has a known true/false answer that is language-independent
MULTILINGUAL_FACTS = [
    # --- True statements ---
    {
        "id": "water_boils",
        "expected": True,
        "translations": {
            "en": "Water boils at 100 degrees Celsius at sea level.",
            "zh_simplified": "水在海平面上在100摄氏度沸腾。",
            "zh_traditional": "水在海平面上在100攝氏度沸騰。",
            "ja": "水は海面では100度で沸騰します。",
            "ko": "물은 해수면에서 섭씨 100도에서 끓습니다.",
            "fr": "L'eau bout à 100 degrés Celsius au niveau de la mer.",
            "de": "Wasser kocht bei 100 Grad Celsius auf Meereshöhe.",
            "es": "El agua hierve a 100 grados Celsius al nivel del mar.",
            "ru": "Вода кипит при 100 градусах Цельсия на уровне моря.",
            "ar": "يغلي الماء عند 100 درجة مئوية عند مستوى سطح البحر.",
            "hi": "समुद्र तल पर पानी 100 डिग्री सेल्सियस पर उबलता है।",
            "pt": "A água ferve a 100 graus Celsius ao nível do mar.",
        },
    },
    {
        "id": "earth_sun",
        "expected": True,
        "translations": {
            "en": "The Earth revolves around the Sun.",
            "zh_simplified": "地球围绕太阳公转。",
            "zh_traditional": "地球圍繞太陽公轉。",
            "ja": "地球は太陽の周りを回っています。",
            "ko": "지구는 태양 주위를 공전합니다.",
            "fr": "La Terre tourne autour du Soleil.",
            "de": "Die Erde dreht sich um die Sonne.",
            "es": "La Tierra gira alrededor del Sol.",
            "ru": "Земля вращается вокруг Солнца.",
            "ar": "الأرض تدور حول الشمس.",
            "hi": "पृथ्वी सूर्य के चारों ओर घूमती है।",
            "pt": "A Terra gira em torno do Sol.",
        },
    },
    {
        "id": "dna_helix",
        "expected": True,
        "translations": {
            "en": "DNA has a double helix structure.",
            "zh_simplified": "DNA具有双螺旋结构。",
            "zh_traditional": "DNA具有雙螺旋結構。",
            "ja": "DNAは二重らせん構造を持っています。",
            "ko": "DNA는 이중 나선 구조를 가지고 있습니다.",
            "fr": "L'ADN a une structure en double hélice.",
            "de": "DNA hat eine Doppelhelixstruktur.",
            "es": "El ADN tiene una estructura de doble hélice.",
            "ru": "ДНК имеет двойную спиральную структуру.",
            "ar": "الحمض النووي له بنية حلزونية مزدوجة.",
            "hi": "डीएनए में दोहरी कुंडलीय संरचना होती है।",
            "pt": "O DNA tem uma estrutura de dupla hélice.",
        },
    },
    {
        "id": "octopus_hearts",
        "expected": True,
        "translations": {
            "en": "Octopuses have three hearts.",
            "zh_simplified": "章鱼有三个心脏。",
            "zh_traditional": "章魚有三顆心臟。",
            "ja": "タコには3つの心臓があります。",
            "ko": "문어는 심장이 3개 있습니다.",
            "fr": "Les pieuvres ont trois cœurs.",
            "de": "Kraken haben drei Herzen.",
            "es": "Los pulpos tienen tres corazones.",
            "ru": "У осьминогов три сердца.",
            "ar": "الأخطبوطات لديها ثلاثة قلوب.",
            "hi": "ऑक्टोपस के तीन दिल होते हैं।",
            "pt": "Os polvos têm três corações.",
        },
    },
    {
        "id": "speed_light",
        "expected": True,
        "translations": {
            "en": "The speed of light is approximately 300,000 kilometers per second.",
            "zh_simplified": "光速大约为每秒30万公里。",
            "zh_traditional": "光速大約為每秒30萬公里。",
            "ja": "光の速度は毎秒約30万キロメートルです。",
            "ko": "빛의 속도는 초당 약 30만 킬로미터입니다.",
            "fr": "La vitesse de la lumière est d'environ 300 000 kilomètres par seconde.",
            "de": "Die Lichtgeschwindigkeit beträgt etwa 300.000 Kilometer pro Sekunde.",
            "es": "La velocidad de la luz es de aproximadamente 300.000 kilómetros por segundo.",
            "ru": "Скорость света составляет примерно 300 000 километров в секунду.",
            "ar": "سرعة الضوء تقارب 300,000 كيلومتر في الثانية.",
            "hi": "प्रकाश की गति लगभग 3 लाख किलोमीटर प्रति सेकंड है।",
            "pt": "A velocidade da luz é de aproximadamente 300.000 quilômetros por segundo.",
        },
    },
    # --- False statements ---
    {
        "id": "sun_revolves_earth",
        "expected": False,
        "translations": {
            "en": "The Sun revolves around the Earth.",
            "zh_simplified": "太阳围绕地球转。",
            "zh_traditional": "太陽圍繞地球轉。",
            "ja": "太陽は地球の周りを回っています。",
            "ko": "태양은 지구 주위를 돕니다.",
            "fr": "Le Soleil tourne autour de la Terre.",
            "de": "Die Sonne dreht sich um die Erde.",
            "es": "El Sol gira alrededor de la Tierra.",
            "ru": "Солнце вращается вокруг Земли.",
            "ar": "الشمس تدور حول الأرض.",
            "hi": "सूर्य पृथ्वी के चारों ओर घूमता है।",
            "pt": "O Sol gira em torno da Terra.",
        },
    },
    {
        "id": "diamonds_coal",
        "expected": False,
        "translations": {
            "en": "Diamonds are made from compressed coal.",
            "zh_simplified": "钻石是由压缩的煤制成的。",
            "zh_traditional": "鑽石是由壓縮的煤製成的。",
            "ja": "ダイヤモンドは圧縮された石炭から作られています。",
            "ko": "다이아몬드는 압축된 석탄으로 만들어집니다.",
            "fr": "Les diamants sont fabriqués à partir de charbon comprimé.",
            "de": "Diamanten werden aus komprimierter Kohle hergestellt.",
            "es": "Los diamantes están hechos de carbón comprimido.",
            "ru": "Алмазы сделаны из сжатого угля.",
            "ar": "الماس مصنوع من الفحم المضغوط.",
            "hi": "हीरे संपीड़ित कोयले से बने होते हैं।",
            "pt": "Diamantes são feitos de carvão comprimido.",
        },
    },
    {
        "id": "goldfish_memory",
        "expected": False,
        "translations": {
            "en": "Goldfish have a memory span of only three seconds.",
            "zh_simplified": "金鱼的记忆只有三秒钟。",
            "zh_traditional": "金魚的記憶只有三秒鐘。",
            "ja": "金魚の記憶は3秒しかありません。",
            "ko": "금붕어의 기억력은 3초에 불과합니다.",
            "fr": "Les poissons rouges ont une mémoire de seulement trois secondes.",
            "de": "Goldfische haben ein Gedächtnis von nur drei Sekunden.",
            "es": "Los peces dorados tienen una memoria de solo tres segundos.",
            "ru": "Золотые рыбки имеют память только на три секунды.",
            "ar": "ذاكرة السمكة الذهبية لا تتجاوز ثلاث ثوانٍ فقط.",
            "hi": "सुनहरी मछली की याददाश्त केवल तीन सेकंड की होती है।",
            "pt": "Os peixes dourados têm uma memória de apenas três segundos.",
        },
    },
    {
        "id": "wall_space",
        "expected": False,
        "translations": {
            "en": "The Great Wall of China is visible from space with the naked eye.",
            "zh_simplified": "长城在太空中肉眼可见。",
            "zh_traditional": "長城在太空中肉眼可見。",
            "ja": "万里の長城は宇宙から肉眼で見えます。",
            "ko": "만리장성은 우주에서 육안으로 볼 수 있습니다.",
            "fr": "La Grande Muraille de Chine est visible depuis l'espace à l'œil nu.",
            "de": "Die Chinesische Mauer ist vom Weltraum aus mit bloßem Auge sichtbar.",
            "es": "La Gran Muralla China es visible desde el espacio a simple vista.",
            "ru": "Великая Китайская стена видна из космоса невооружённым глазом.",
            "ar": "سور الصين العظيم مرئي من الفضاء بالعين المجردة.",
            "hi": "चीन की दीवार अंतरिक्ष से नंगी आँखों से दिखाई देती है।",
            "pt": "A Grande Muralha da China é visível do espaço a olho nu.",
        },
    },
    {
        "id": "electrons_larger",
        "expected": False,
        "translations": {
            "en": "Electrons are larger than protons.",
            "zh_simplified": "电子比质子大。",
            "zh_traditional": "電子比質子大。",
            "ja": "電子は陽子より大きいです。",
            "ko": "전자는 양성자보다 큽니다.",
            "fr": "Les électrons sont plus grands que les protons.",
            "de": "Elektronen sind größer als Protonen.",
            "es": "Los electrones son más grandes que los protones.",
            "ru": "Электроны больше протонов.",
            "ar": "الإلكترونات أكبر من البروتونات.",
            "hi": "इलेक्ट्रॉन प्रोटॉन से बड़े होते हैं।",
            "pt": "Os elétrons são maiores que os prótons.",
        },
    },
]

# Code-switching tests
CODE_SWITCHING = [
    {
        "id": "cs_zh_en_refund",
        "states": {
            "pure_en": "The customer is requesting a refund for order number 12345.",
            "pure_zh": "客户要求退还订单号12345的款项。",
            "mixed_zh_en": "这个customer要求refund订单号12345。",
            "en_state_zh_q": "The customer is requesting a refund for order number 12345.",
            "zh_state_en_q": "客户要求退还订单号12345的款项。",
        },
        "instructions": {
            "pure_en": "Is this a refund request?",
            "pure_zh": "这是退款请求吗？",
            "mixed_zh_en": "Is this a refund request?",
            "en_state_zh_q": "这是退款请求吗？",
            "zh_state_en_q": "Is this a refund request?",
        },
        "expected": True,
    },
    {
        "id": "cs_zh_en_angry",
        "states": {
            "pure_en": "I am extremely disappointed and frustrated with your service.",
            "pure_zh": "我对你们的服务非常失望和沮丧。",
            "mixed_zh_en": "我非常disappointed和frustrated with your service。",
            "en_state_zh_q": "I am extremely disappointed and frustrated with your service.",
            "zh_state_en_q": "我对你们的服务非常失望和沮丧。",
        },
        "instructions": {
            "pure_en": "Is the customer angry?",
            "pure_zh": "客户是否生气？",
            "mixed_zh_en": "Is the customer angry?",
            "en_state_zh_q": "客户是否生气？",
            "zh_state_en_q": "Is the customer angry?",
        },
        "expected": True,
    },
    {
        "id": "cs_zh_en_tech",
        "states": {
            "pure_en": "The API returns a 500 error on POST requests with JSON payload.",
            "pure_zh": "API在发送JSON负载的POST请求时返回500错误。",
            "mixed_zh_en": "这个API在POST request的时候返回500 error。",
            "en_state_zh_q": "The API returns a 500 error on POST requests with JSON payload.",
            "zh_state_en_q": "API在发送JSON负载的POST请求时返回500错误。",
        },
        "instructions": {
            "pure_en": "Is this a technical bug report?",
            "pure_zh": "这是技术错误报告吗？",
            "mixed_zh_en": "Is this a technical bug report?",
            "en_state_zh_q": "这是技术错误报告吗？",
            "zh_state_en_q": "Is this a technical bug report?",
        },
        "expected": True,
    },
]


def main():
    client = JevClient()
    records = []
    open(OUTFILE, "w").close()

    def emit(record):
        records.append(record)
        with open(OUTFILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 70)
    print("PROBE 9: Multilingual Parity")
    print(f"Facts: {len(MULTILINGUAL_FACTS)}, Languages: 12, Code-switching: {len(CODE_SWITCHING)}")
    print("=" * 70)

    # --- 9a: Parallel translation test ---
    print("\n--- 9a: Same fact in 12 languages ---")
    languages = ["en", "zh_simplified", "zh_traditional", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt"]

    for fact in MULTILINGUAL_FACTS:
        for lang in languages:
            state = fact["translations"][lang]
            t0 = time.monotonic()
            resp = client.noul(state, fact["id"], "Is this statement factually correct?")
            elapsed = time.monotonic() - t0
            noul = resp["answers"][fact["id"]]["noul"]
            predicted = noul > 0.5
            correct = predicted == fact["expected"]

            emit({
                "test": "parallel_translation",
                "fact_id": fact["id"],
                "language": lang,
                "expected": fact["expected"],
                "noul": noul,
                "predicted": predicted,
                "correct": correct,
                "input_tokens": resp["usage"]["input_tokens"],
                "elapsed_s": round(elapsed, 4),
            })

        print(f"  {fact['id']}: done")

    # --- 9b: Code-switching ---
    print("\n--- 9b: Code-switching ---")
    for cs in CODE_SWITCHING:
        for variant, state in cs["states"].items():
            instr = cs["instructions"][variant]
            t0 = time.monotonic()
            resp = client.noul(state, cs["id"], instr)
            elapsed = time.monotonic() - t0
            noul = resp["answers"][cs["id"]]["noul"]

            emit({
                "test": "code_switching",
                "case_id": cs["id"],
                "variant": variant,
                "noul": noul,
                "expected": cs["expected"],
                "correct": (noul > 0.5) == cs["expected"],
                "input_tokens": resp["usage"]["input_tokens"],
                "elapsed_s": round(elapsed, 4),
            })
            print(f"  {cs['id']} [{variant:15s}]: noul={noul:.2f}")

    client.close()

    # --- Analysis ---
    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    trans = [r for r in records if r["test"] == "parallel_translation"]

    print("\nPer-language accuracy:")
    lang_accs = {}
    for lang in languages:
        lr = [r for r in trans if r["language"] == lang]
        acc = sum(1 for r in lr if r["correct"]) / len(lr)
        mean_conf = sum(r["noul"] if r["expected"] else 1 - r["noul"] for r in lr) / len(lr)
        lang_accs[lang] = acc
        bar = "█" * int(acc * 30)
        print(f"  {lang:15s}: {acc:.0%} ({sum(1 for r in lr if r['correct'])}/{len(lr)})  conf={mean_conf:.3f}  {bar}")

    # Language family grouping
    print("\nLanguage family comparison:")
    families = {
        "CJK": ["zh_simplified", "zh_traditional", "ja", "ko"],
        "European": ["fr", "de", "es", "pt"],
        "Other": ["ru", "ar", "hi"],
        "English": ["en"],
    }
    for family, langs in families.items():
        accs = [lang_accs[l] for l in langs]
        mean = sum(accs) / len(accs)
        print(f"  {family:12s}: mean accuracy = {mean:.0%}")

    # Parity test
    en_acc = lang_accs["en"]
    zh_acc = lang_accs["zh_simplified"]
    print(f"\nEN vs ZH parity: EN={en_acc:.0%}, ZH={zh_acc:.0%}, gap={en_acc - zh_acc:+.0%}")
    if abs(en_acc - zh_acc) <= 0.05:
        print("  → Near parity: consistent with Qwen-family base model")
    elif en_acc > zh_acc:
        print("  → English dominant: consistent with LLaMA/Gemma base model")
    else:
        print("  → Chinese dominant: consistent with Yi/DeepSeek base model")

    # Code-switching
    cs_records = [r for r in records if r["test"] == "code_switching"]
    print("\nCode-switching robustness:")
    for variant in ["pure_en", "pure_zh", "mixed_zh_en", "en_state_zh_q", "zh_state_en_q"]:
        vr = [r for r in cs_records if r["variant"] == variant]
        if vr:
            mean_noul = sum(r["noul"] for r in vr) / len(vr)
            correct = sum(1 for r in vr if r["correct"])
            print(f"  {variant:15s}: mean noul={mean_noul:.3f}, correct={correct}/{len(vr)}")

    # Token cost by language
    print("\nToken cost by language (mean input_tokens for same content):")
    for lang in languages:
        lr = [r for r in trans if r["language"] == lang]
        mean_tok = sum(r["input_tokens"] for r in lr) / len(lr)
        print(f"  {lang:15s}: {mean_tok:.0f} tokens")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
