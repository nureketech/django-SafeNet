"""``reports/dataset_report.md``: sizes, distributions and examples (all numbers computed)."""

from __future__ import annotations

import pandas as pd

from safenet import paths
from safenet.common import dedup_key
from safenet.generator.split import leakage


def _table(series: pd.Series, title: str) -> str:
    counts = series.value_counts()
    share = (counts / counts.sum() * 100).round(1)
    lines = [f"| {title} | саны | % |", "|---|---:|---:|"]
    lines += [f"| {k} | {counts[k]} | {share[k]} |" for k in counts.index]
    return "\n".join(lines)


def _crosstab(df: pd.DataFrame, rows: str, cols: str) -> str:
    ct = pd.crosstab(df[rows], df[cols])
    header = f"| {rows} \\ {cols} | " + " | ".join(map(str, ct.columns)) + " |"
    sep = "|---|" + "---:|" * len(ct.columns)
    body = [f"| {idx} | " + " | ".join(str(v) for v in row) + " |" for idx, row in ct.iterrows()]
    return "\n".join([header, sep, *body])


def _example(text: str) -> str:
    return text.replace("\n", " ⏎ ").replace("|", "/")


def build_report(df: pd.DataFrame, gold: pd.DataFrame, truth: dict) -> str:
    none = df[df["label"] == "none"]
    schemes = df[df["label"] != "none"]
    obf = df["obfuscations"].map(lambda o: list(o) or ["(жоқ)"]).explode()
    near_dup = df["text"].map(dedup_key).duplicated().mean() * 100
    templates = (
        df[df["lang"].isin(["ru", "kk"])]
        .groupby(["label", "lang"])["template_id"].nunique().unstack()
    )
    parts = [
        "# Синтетикалық датасет туралы есеп",
        "",
        "> Барлық деректер синтетикалық (`synthetic == True`): карталар Луна тексерісінен өтпейді, "
        "телефондар кездейсоқ, домендер `.test` / `.example` аймақтарында, аттар — шартты никнеймдер. "
        "Датасет тек детекторды оқыту мен тексеруге арналған.",
        "",
        "Есеп `python -m safenet.generator` (`make data`) арқылы автоматты түрде жасалады.",
        "",
        "## Өлшемдер",
        "",
        f"- Барлық жазба: **{len(df)}**",
        f"- Шаблондар: **{df['template_id'].nunique()}**; бөліну (60/20/20, `template_id` бойынша): "
        + ", ".join(f"{s} — {int((df['split'] == s).sum())}" for s in ("train", "val", "test")),
        f"- Бөлінулер арасындағы шаблон қиылысуы: {leakage(df)}",
        f"- Қиын негативтер: **{int(none['is_hard_negative'].sum())}** — `none` класының "
        f"**{none['is_hard_negative'].mean() * 100:.1f}%**",
        f"- Бүркемеленген жазбалар: схемалар — **{(schemes['obfuscations'].map(len) > 0).mean() * 100:.1f}%**, "
        f"негативтер — **{(none['obfuscations'].map(len) > 0).mean() * 100:.1f}%**",
        f"- Толық көшірмелер: {int(df['text'].duplicated().sum())}; дерлік көшірмелер: {near_dup:.2f}%",
        f"- Скриншоттар (PNG): {int(df['image_path'].notna().sum())}; рендер параметрлері: {truth.get('images', {})}",
        f"- Gold-жинақ (қолмен жазылған, тек бағалауға): **{len(gold)}** жазба",
        "",
        "## Бөлінулер",
        "",
        _table(df["label"], "label"),
        "",
        _table(df["lang"], "lang"),
        "",
        _table(df["source_type"], "source_type"),
        "",
        _table(df["platform"], "platform"),
        "",
        _table(obf, "obfuscation"),
        "",
        "### Тәуекел деңгейі (risk_gold) × класс",
        "",
        _crosstab(df, "label", "risk_gold"),
        "",
        "### Бөліну × класс",
        "",
        _crosstab(df, "split", "label"),
        "",
        "### Бөліну × тіл",
        "",
        _crosstab(df, "split", "lang"),
        "",
        "### Қиын негативтер мен қалыпты контент түрлері (`none`)",
        "",
        _table(none["subtype"], "subtype"),
        "",
        "### Бірегей шаблондар саны (ru / kk жазбалары бойынша)",
        "",
        "| label | ru | kk |",
        "|---|---:|---:|",
        *[f"| {lab} | {int(templates.loc[lab, 'ru'])} | {int(templates.loc[lab, 'kk'])} |" for lab in templates.index],
        "",
        "## Желілер (граф үшін, `networks.json`)",
        "",
        "| желі | трек | ортақ нысан | аккаунттар | жазбалар |",
        "|---|---|---|---:|---:|",
        *[
            f"| {n['network_id']} | {n['track']} | {n['shared_entity_type']} | {len(n['accounts'])} | {len(n['record_ids'])} |"
            for n in truth["networks"]
        ],
        "",
        "Тұзақ: ортақ сайты бар заңды дүкендер — "
        + "; ".join(f"{t['trap_id']}: {len(t['accounts'])} аккаунт" for t in truth["legit_shared_domains"]),
        "",
        "## Мысалдар (әр класқа 3, train жинағынан)",
        "",
    ]
    train = df[df["split"] == "train"]
    for label in ("lottery", "casino", "pyramid", "none"):
        parts.append(f"### {label}")
        parts.append("")
        sample = train[train["label"] == label].sample(3, random_state=7)
        for _, row in sample.iterrows():
            parts.append(
                f"- `{row['id']}` [{row['lang']}, {row['source_type']}, risk={row['risk_gold']}, "
                f"hn={row['is_hard_negative']}] {_example(row['text'])}"
            )
        parts.append("")
    parts += [
        "## Gold-жинақ",
        "",
        _table(gold["label"], "label"),
        "",
        _table(gold["lang"], "lang"),
        "",
        f"Gold ішіндегі қиын негативтер: {int(gold['is_hard_negative'].sum())}.",
        "",
        "## Белгілі шектеулер",
        "",
        "- Синтетикалық мәтіндер шаблондық: test жинағы фразалар деңгейінде бөлек болса да, нақты әлеуметтік желі "
        "мәтіндерінен оңайырақ. Шынайы сапаны gold-жинақ көрсетеді.",
        "- Қазақ тіліндегі кейбір слот жалғаулары (күн/ай атаулары) әрдайым грамматикалық дұрыс емес — бұл "
        "тірі жазуға ұқсайды, бірақ әдейі жасалған жоқ.",
        "",
    ]
    return "\n".join(parts)


def save_report(text: str) -> None:
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "dataset_report.md").write_text(text, encoding="utf-8")
