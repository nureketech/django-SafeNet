"""``python -m safenet.graph`` — score the dataset, build ``models/graph.joblib`` and the graph report."""

from __future__ import annotations

import argparse

from safenet import paths
from safenet.graph.lookup import load_graph, save_graph
from safenet.graph.report import all_bonuses, build_from_dataset, load_truth, planted_recovery, trap_bonuses
from safenet.graph.viz import cluster_html


def write_report(model, records, recovery, traps, bonuses) -> None:
    s = model.stats
    found = int(recovery["found"].sum())
    lines = [
        "# Байланыстар графы: есеп",
        "",
        "`python -m safenet.graph` (`make train` ішінде) арқылы бүкіл датасетте құрылды.",
        "",
        "## Граф",
        "",
        f"- Түйіндер: {s['nodes']} ({', '.join(f'{k}: {v}' for k, v in sorted(s['nodes_by_kind'].items()))}); "
        f"қабырғалар: {s['edges']}",
        f"- «Шулы» түйіндер алынды (> {model.params['max_entity_accounts']} аккаунтпен байланысқан): "
        f"{s['noisy_removed'] or 'жоқ'}",
        f"- Кластерлер (≥ 2 аккаунт): **{s['clusters']}**; құру уақыты: **{s['build_seconds']} с** (талап < 10 с)",
        "- Карта мен телефон түйіндері — SHA-256 хэш, белгісі — маска; 6-дан аз цифры белгілі маскалар "
        "байланыс ретінде қолданылмайды.",
        "",
        "## Ендірілген желілерді қалпына келтіру (`networks.json`)",
        "",
        f"Табылды: **{found} / {len(recovery)}** (желі «табылды» = қатысушыларының ≥ 70%-ы бір кластерде).",
        "",
        "| желі | трек | ортақ нысан | аккаунттар | кластер | кластер өлшемі | қамту | кластер тәуекелі |",
        "|---|---|---|---:|---|---:|---:|---:|",
        *[f"| {r.network_id} | {r.track} | {r.shared_entity} | {r.accounts} | {r.cluster_id} | "
          f"{r.cluster_accounts} | {r.coverage:.2f} | {r.cluster_risk if r.cluster_risk is not None else '—'} |"
          for r in recovery.itertuples()],
        "",
        "## Тұзақ: ортақ сайты бар заңды дүкендер",
        "",
        "| тұзақ | жазба | өз ұпайы | кластер | network_bonus |",
        "|---|---|---:|---|---:|",
        *[f"| {r.trap_id} | {r.id} | {r.risk_score} | {r.cluster_id} | {r.bonus} |" for r in traps.itertuples()],
        "",
        "## network_bonus",
        "",
        f"- Ереже: жазбаның өз ұпайы ≥ `medium`, кластердің жоғары тәуекелді жазбалар үлесі ≥ "
        f"{model.params['min_cluster_risk']}, аккаунттарда ортақ төлем/жарнама нысаны бар (карта, телефон, "
        "промокод, сілтеме, домен) → бонус = 10 × үлес (≤ 10).",
        f"- Бонус алған жазбалар: {int((bonuses > 0).sum())} / {len(bonuses)}; ауқымы "
        f"[{bonuses.min():.1f}, {bonuses.max():.1f}]",
        "",
        "## Кластерлер",
        "",
        "| кластер | аккаунттар | жазбалар | тәуекел үлесі | орташа ұпай | трек | ортақ нысандар |",
        "|---|---:|---:|---:|---:|---|---|",
        *[f"| {c.cluster_id} | {len(c.accounts)} | {len(c.record_ids)} | {c.risk_share:.2f} | {c.mean_score} | "
          f"{c.top_track} | {'; '.join(f'{e.kind}: {e.label}' for e in c.shared[:3])} |"
          for c in model.clusters.values()],
        "",
        "Интерактивті көрініс: `reports/graph/<кластер>.html` (pyvis; түйін түсі — тәуекел деңгейі).",
        "",
    ]
    (paths.REPORTS_DIR / "graph_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", action="store_true", help="recompute dataset scores (default: when missing)")
    args = parser.parse_args(argv)
    model, records = build_from_dataset(rescore=args.rescore)
    save_graph(model)
    load_graph.cache_clear()
    truth = load_truth()
    recovery = planted_recovery(model, truth)
    traps = trap_bonuses(model, records, truth)
    bonuses = all_bonuses(model, records)
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_report(model, records, recovery, traps, bonuses)
    out_dir = paths.REPORTS_DIR / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid in recovery["cluster_id"].dropna().unique()[:3]:
        (out_dir / f"{cid}.html").write_text(cluster_html(model, cid), encoding="utf-8")
    print(f"graph: {model.stats['nodes']} nodes, {model.stats['clusters']} clusters, "
          f"{int(recovery['found'].sum())}/{len(recovery)} planted networks found, "
          f"build {model.stats['build_seconds']} s")


if __name__ == "__main__":
    main()
