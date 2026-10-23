"""Interactive cluster view (pyvis, vis.js inlined so it works offline)."""

from __future__ import annotations

from pyvis.network import Network

from safenet.graph.build import GraphModel

LEVEL_COLORS = {"high": "#d64545", "medium": "#e8a33d", "low": "#4caf7a"}
ENTITY_COLORS = {
    "card": "#4f6bd8", "phone": "#4f6bd8", "promo": "#b0509f", "domain": "#7d5bc7", "link": "#7d5bc7",
    "group": "#9aa3ad",
}
ENTITY_TITLES = {
    "card": "Карта (маска)", "phone": "Телефон (маска)", "promo": "Промокод", "domain": "Домен",
    "link": "Сілтеме", "group": "Топ / чат",
}


def cluster_html(model: GraphModel, cluster_id: str, height: str = "560px") -> str:
    """HTML page with the accounts of a cluster and the entities linking them."""
    cluster = model.clusters[cluster_id]
    G = model.graph
    accounts = [f"acc:{a}" for a in cluster.accounts]
    shared = {e.node_id for e in cluster.shared}
    net = Network(height=height, width="100%", cdn_resources="in_line", notebook=False, bgcolor="#ffffff")
    net.barnes_hut(gravity=-6000, spring_length=140)
    for acc in accounts:
        d = G.nodes[acc]
        level = d.get("level", "low")
        net.add_node(acc, label=d["label"], color=LEVEL_COLORS.get(level, "#999999"), shape="dot", size=18,
                     title=f"Аккаунт · деңгей: {level} · ең жоғары ұпай: {d.get('max_score', 0):.0f} · "
                           f"жазбалар: {d.get('n_records', 0)}")
    for node_id in sorted(shared):
        d = G.nodes[node_id]
        net.add_node(node_id, label=d["label"], color=ENTITY_COLORS.get(d["kind"], "#999999"), shape="box",
                     title=ENTITY_TITLES.get(d["kind"], d["kind"]))
    for acc in accounts:
        for nb in G.neighbors(acc):
            if nb in shared:
                relation = G[acc][nb].get("relation", "mentions")
                net.add_edge(acc, nb, title="жариялайды" if relation == "publishes" else "атайды")
    return net.generate_html()
