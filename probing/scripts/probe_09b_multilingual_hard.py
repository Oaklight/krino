#!/usr/bin/env python3
"""Probe 9b: Hard multilingual — questions near capability boundary.

Probe 9 flaw: factual questions were too easy (100% in almost all languages).
This probe uses harder questions where language-specific training quality matters:
- Cultural nuance that requires deep language understanding
- Domain jargon in each language (not translations of English)
- Pragmatic inference that depends on language-specific conventions
- Rare/literary vocabulary
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTFILE = os.path.join(RESULTS_DIR, "09b-multilingual-hard.jsonl")

# Hard language-specific tests — NOT translations of the same question.
# Each tests understanding that requires real depth in that language.
LANG_TESTS = {
    "en": [
        ("The phrase 'break a leg' means to wish someone good luck.", "idiom1", "Is this the correct meaning?", True),
        ("In English, 'I could care less' and 'I couldn't care less' mean the same thing in common usage.", "idiom2", "Is this true in practice?", True),
        ("The word 'literally' is never used to mean 'figuratively' in modern English.", "usage1", "Is this correct?", False),
        ("A 'red herring' in an argument is a piece of strong supporting evidence.", "idiom3", "Is this the correct meaning?", False),
        ("'Whom' is the object form of 'who'.", "grammar1", "Is this grammatically correct?", True),
        ("The sentence 'Time flies like an arrow; fruit flies like a banana' is a pun.", "pun1", "Is this correct?", True),
        ("'Inflammable' means 'not flammable' in English.", "prefix1", "Is this correct?", False),
        ("In British English, 'pants' refers to underwear, not trousers.", "dialect1", "Is this correct?", True),
    ],
    "zh": [
        ("成语'画蛇添足'的意思是做多余的事情。", "idiom1", "这个解释正确吗？", True),
        ("'知足常乐'表达的意思是知道满足的人经常快乐。", "idiom2", "这个理解正确吗？", True),
        ("在中文里，'马马虎虎'形容做事非常认真。", "idiom3", "这个解释正确吗？", False),
        ("成语'杯弓蛇影'形容因为疑心而害怕。", "idiom4", "这个解释正确吗？", True),
        ("'不以为然'的意思是非常赞同。", "usage1", "这个用法正确吗？", False),
        ("'差强人意'表示勉强令人满意。", "idiom5", "这个是正确的解释吗？", True),
        ("量词'条'可以用来形容鱼。", "grammar1", "这个量词使用正确吗？", True),
        ("'七月流火'形容七月天气非常炎热。", "idiom6", "这个理解正确吗？", False),
    ],
    "ja": [
        ("日本語で「猫の手も借りたい」は非常に忙しいという意味です。", "idiom1", "この説明は正しいですか？", True),
        ("「空気を読む」とは天気予報をすることです。", "idiom2", "この説明は正しいですか？", False),
        ("敬語の「いらっしゃる」は「いる」の謙譲語です。", "grammar1", "この説明は正しいですか？", False),
        ("「木を見て森を見ず」は細部にとらわれて全体を見失うことです。", "idiom3", "この説明は正しいですか？", True),
        ("日本語で「先輩」は年下の人を指します。", "usage1", "この説明は正しいですか？", False),
        ("「一期一会」は一生に一度の出会いを大切にするという意味です。", "idiom4", "この説明は正しいですか？", True),
        ("「お疲れ様です」は朝の挨拶として使います。", "usage2", "この使い方は一般的ですか？", False),
        ("「三日坊主」は何事も長続きしない人のことです。", "idiom5", "この説明は正しいですか？", True),
    ],
    "ko": [
        ("한국어에서 '눈치'는 상대방의 기분이나 상황을 파악하는 능력을 의미합니다.", "idiom1", "이 설명이 맞습니까?", True),
        ("'식은 죽 먹기'는 매우 어려운 일을 뜻합니다.", "idiom2", "이 설명이 맞습니까?", False),
        ("한국어의 존댓말에서 '-습니다'는 반말 어미입니다.", "grammar1", "이 설명이 맞습니까?", False),
        ("'우물 안 개구리'는 견문이 좁은 사람을 의미합니다.", "idiom3", "이 설명이 맞습니까?", True),
        ("한국어에서 '형'은 여자가 오빠를 부르는 호칭입니다.", "usage1", "이 설명이 맞습니까?", False),
        ("'빈수레가 요란하다'는 실속 없는 사람이 더 떠든다는 뜻입니다.", "idiom4", "이 설명이 맞습니까?", True),
        ("'다다익선'은 많으면 많을수록 좋다는 의미입니다.", "idiom5", "이 설명이 맞습니까?", True),
        ("한국어에서 '선배'는 나이가 어린 사람을 가리킵니다.", "usage2", "이 설명이 맞습니까?", False),
    ],
    "fr": [
        ("L'expression 'avoir le cafard' signifie être triste ou déprimé.", "idiom1", "Est-ce le bon sens?", True),
        ("En français, 'je m'en fiche' est une expression très polie.", "usage1", "Est-ce correct?", False),
        ("Le subjonctif est utilisé pour exprimer le doute ou le souhait.", "grammar1", "Est-ce correct?", True),
        ("'Poser un lapin' signifie offrir un cadeau.", "idiom2", "Est-ce le bon sens?", False),
        ("En français, le passé composé se forme toujours avec l'auxiliaire 'avoir'.", "grammar2", "Est-ce correct?", False),
        ("L'expression 'il pleut des cordes' signifie qu'il pleut très fort.", "idiom3", "Est-ce correct?", True),
        ("'Métro, boulot, dodo' décrit une routine quotidienne monotone.", "idiom4", "Est-ce correct?", True),
        ("Le mot 'formidable' en français signifie toujours 'effrayant'.", "usage2", "Est-ce correct en français moderne?", False),
    ],
    "de": [
        ("Das Wort 'Schadenfreude' beschreibt die Freude am Unglück anderer.", "idiom1", "Ist diese Definition korrekt?", True),
        ("'Ich verstehe nur Bahnhof' bedeutet, dass man etwas gut versteht.", "idiom2", "Ist das korrekt?", False),
        ("Im Deutschen steht das konjugierte Verb im Hauptsatz immer an letzter Stelle.", "grammar1", "Ist das korrekt?", False),
        ("'Torschlusspanik' beschreibt die Angst, eine Gelegenheit zu verpassen.", "idiom3", "Ist diese Bedeutung korrekt?", True),
        ("'Gemütlichkeit' lässt sich einfach mit einem Wort ins Englische übersetzen.", "usage1", "Ist das korrekt?", False),
        ("Der Konjunktiv II wird im Deutschen für irreale Bedingungen verwendet.", "grammar2", "Ist das korrekt?", True),
        ("'Wanderlust' bedeutet im Deutschen die Lust am Wandern oder Reisen.", "idiom4", "Ist das korrekt?", True),
        ("Im Deutschen gibt es genau zwei grammatische Geschlechter.", "grammar3", "Ist das korrekt?", False),
    ],
    "es": [
        ("La expresión 'tener mala leche' significa tener mala suerte.", "idiom1", "¿Es correcto este significado?", False),
        ("El subjuntivo en español se usa para expresar deseos e incertidumbre.", "grammar1", "¿Es correcto?", True),
        ("'Dar en el clavo' significa cometer un error grave.", "idiom2", "¿Es correcto?", False),
        ("En español, 'estar' se usa para condiciones permanentes y 'ser' para temporales.", "grammar2", "¿Es correcto?", False),
        ("'Ponerse las pilas' significa empezar a trabajar con más energía.", "idiom3", "¿Es correcto?", True),
        ("La Real Academia Española fue fundada en el siglo XVIII.", "fact1", "¿Es correcto?", True),
        ("'Merienda' se refiere a la comida principal del mediodía.", "usage1", "¿Es correcto?", False),
        ("El voseo se usa en varios países latinoamericanos, especialmente Argentina.", "grammar3", "¿Es correcto?", True),
    ],
    "ru": [
        ("Выражение 'вешать лапшу на уши' означает обманывать.", "idiom1", "Это правильное значение?", True),
        ("В русском языке есть шесть падежей.", "grammar1", "Это правильно?", True),
        ("'Авось' выражает уверенность в результате.", "usage1", "Это правильно?", False),
        ("Выражение 'медвежья услуга' означает помощь, которая приносит вред.", "idiom2", "Это правильно?", True),
        ("В русском языке ударение всегда падает на первый слог.", "grammar2", "Это правильно?", False),
        ("Слово 'тоска' точно переводится на английский как 'sadness'.", "usage2", "Это полностью правильно?", False),
        ("'Белая ворона' означает человека, который сильно отличается от других.", "idiom3", "Это правильно?", True),
        ("Совершенный вид глагола указывает на завершённое действие.", "grammar3", "Это правильно?", True),
    ],
}


def main():
    client = JevClient()
    records = []
    open(OUTFILE, "w").close()

    print("=" * 70)
    print("PROBE 9b: Hard Multilingual (Idioms, Grammar, Cultural Nuance)")
    print(f"Languages: {len(LANG_TESTS)}, Questions per language: 8")
    print("=" * 70)

    for lang, questions in LANG_TESTS.items():
        correct = 0
        total = 0
        nouls = []
        for state, qid, instructions, expected in questions:
            t0 = time.monotonic()
            resp = client.noul(state, qid, instructions)
            elapsed = time.monotonic() - t0
            noul = resp["answers"][qid]["noul"]
            predicted = noul > 0.5
            is_correct = predicted == expected

            record = {
                "language": lang,
                "state": state,
                "qid": qid,
                "noul": noul,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "input_tokens": resp["usage"]["input_tokens"],
                "elapsed_s": round(elapsed, 4),
            }
            records.append(record)
            with open(OUTFILE, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total += 1
            if is_correct:
                correct += 1
            conf = noul if expected else 1 - noul
            nouls.append(conf)

        acc = correct / total
        mean_conf = sum(nouls) / len(nouls)
        print(f"  {lang:5s}: {correct}/{total} ({acc:.0%})  conf={mean_conf:.3f}")

    client.close()

    # --- Analysis ---
    print(f"\n{'=' * 70}")
    print("ANALYSIS: Hard Multilingual Parity")
    print("=" * 70)

    lang_accs = {}
    lang_confs = {}
    for lang in LANG_TESTS:
        lr = [r for r in records if r["language"] == lang]
        acc = sum(1 for r in lr if r["correct"]) / len(lr)
        conf = sum(r["noul"] if r["expected"] else 1 - r["noul"] for r in lr) / len(lr)
        lang_accs[lang] = acc
        lang_confs[lang] = conf

    print("\nRanked by accuracy:")
    for lang, acc in sorted(lang_accs.items(), key=lambda x: -x[1]):
        conf = lang_confs[lang]
        bar = "█" * int(acc * 30)
        print(f"  {lang:5s}: {acc:.0%}  conf={conf:.3f}  {bar}")

    # EN vs ZH gap
    en_acc = lang_accs.get("en", 0)
    zh_acc = lang_accs.get("zh", 0)
    gap = en_acc - zh_acc
    print(f"\nEN vs ZH: EN={en_acc:.0%}, ZH={zh_acc:.0%}, gap={gap:+.0%}")

    # CJK vs European
    cjk = [lang_accs.get(l, 0) for l in ["zh", "ja", "ko"]]
    eur = [lang_accs.get(l, 0) for l in ["fr", "de", "es", "ru"]]
    cjk_mean = sum(cjk) / len(cjk) if cjk else 0
    eur_mean = sum(eur) / len(eur) if eur else 0
    print(f"CJK mean: {cjk_mean:.0%}, European mean: {eur_mean:.0%}")

    # Confidence gap (more sensitive than accuracy)
    en_conf = lang_confs.get("en", 0)
    zh_conf = lang_confs.get("zh", 0)
    print(f"\nConfidence gap: EN={en_conf:.3f}, ZH={zh_conf:.3f}, delta={en_conf - zh_conf:+.3f}")
    print("(Confidence is more sensitive than accuracy for detecting training data gaps)")

    print(f"\nTotal: {len(records)} records written to {OUTFILE}")


if __name__ == "__main__":
    main()
