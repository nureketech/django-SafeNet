"""Dashboard pages: case queue, case card, content check, network, metrics."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from collections import Counter
from urllib.parse import urlencode

import markdown
from django.conf import settings
from django.contrib.auth.decorators import login_not_required, permission_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db.models import Count
from django.http import FileResponse, Http404, HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.decorators.http import require_POST
from PIL import Image, UnidentifiedImageError

from app.dashboard import i18n
from app.dashboard.audit import audit
from app.dashboard.models import Case, Feedback
from app.dashboard.network import (
    account_label, case_text, cluster_of_case, links_for, live_graph, live_lookup,
)
from app.dashboard.present import card_context, link_story, network_summary
from app.dashboard.services import absorb_manual_copies, record_feedback, save_case
from app.dashboard.sources import ContextError, build_context
from app.dashboard.telegram import FetchError, fetch_channel, fetch_post, is_telegram_link, parse_link
from safenet import LABELS, paths
from safenet.demo import DEMO_DIR
from safenet.pipeline import analyze, analyze_many

can_view = permission_required("dashboard.view_case", raise_exception=True)
PAGE_SIZE = 50
RECENT_CHECKS = 10
FILTERS = {"track": "primary_track", "lang": "lang", "platform": "platform", "source": "source_type",
           "level": "risk_level", "origin": "origin", "account": "account_id", "group": "group_id"}
REPORT_IMAGES = {"confusion_test.png", "confusion_gold.png", "pr_curve.png", "ablation.png"}


def _t(request) -> tuple[str, dict]:
    lang = i18n.get_lang(request)
    return lang, i18n.UI[lang]


@login_not_required
def set_lang(request, code: str):
    target = request.META.get("HTTP_REFERER") or ""
    if not url_has_allowed_host_and_scheme(target, allowed_hosts={request.get_host()},
                                           require_https=request.is_secure()):
        target = reverse("queue")
    response = redirect(target)
    if code in i18n.LANGS:
        response.set_cookie(i18n.COOKIE, code, max_age=365 * 24 * 3600, samesite="Lax")
    return response


# ------------------------------------------------------------------- queue
def _rows(cases, model=None) -> list[dict]:
    rows = []
    for case in cases:
        snippet = case.text or (case.result.get("ocr_text") or "")
        cluster = cluster_of_case(model, case)
        rows.append({"case": case, "snippet": snippet[:180] + ("…" if len(snippet) > 180 else ""),
                     "network": {"id": cluster.cluster_id, "size": len(cluster.accounts)} if cluster else None})
    return rows


@can_view
def queue(request):
    lang, t = _t(request)
    qs = Case.objects.all()
    selected = {}
    for param, field in FILTERS.items():
        value = request.GET.get(param, "")
        if value:
            qs = qs.filter(**{field: value})
            selected[param] = value
    model = live_graph()
    in_networks = {r for c in model.clusters.values() for r in c.record_ids} if model is not None else set()
    if request.GET.get("linked") == "1":
        qs = qs.filter(pk__in=in_networks)
        selected["linked"] = "1"
    if request.GET.get("sort") == "new":  # default order: risk score (Case.Meta.ordering)
        qs = qs.order_by("-analysed_at", "id")
        selected["sort"] = "new"
    counts = {row["risk_level"]: row["n"] for row in Case.objects.values("risk_level").annotate(n=Count("id"))}
    page = Paginator(qs, PAGE_SIZE).get_page(request.GET.get("page"))
    query = urlencode(selected)
    return render(request, "dashboard/queue.html", {
        "active": "queue",
        "rows": _rows(page.object_list, model),
        "linked_count": len(in_networks),
        "page": page,
        "counts": {lvl: counts.get(lvl, 0) for lvl in ("high", "medium", "low")},
        "total": sum(counts.values()),
        "filtered": qs.count(),
        "demo_count": Case.objects.filter(origin="demo").count(),
        "check_count": Case.objects.filter(origin="check").count(),
        "newest_first": selected.get("sort") == "new",
        "selected": selected,
        "query": query,
        "options": {
            "track": [(k, i18n.TRACKS[lang][k]) for k in LABELS],
            "lang": [(k, i18n.LANG_NAMES[lang][k]) for k in ("ru", "kk", "mixed", "translit")],
            "platform": [(k, i18n.PLATFORM_NAMES[k]) for k in i18n.PLATFORMS],
            "source": list(i18n.SOURCES[lang].items()),
            "level": [(k, t[f"level_{k}"]) for k in ("high", "medium", "low")],
        },
    })


# --------------------------------------------------------------------- case
@can_view
def case_detail(request, case_id: str):
    lang, t = _t(request)
    case = get_object_or_404(Case, pk=case_id)
    audit(request, "case_view", case=case.id)
    result = case.analysis
    ctx = card_context(result, lang, t)
    links = links_for(account_id=case.account_id, text=case_text(case), case_id=case.id, group_id=case.group_id)
    return render(request, "dashboard/case.html", {
        "active": "queue",
        "case": case,
        "account": account_label(case.account_id) if case.account_id else "",
        "account_masked": account_label(case.account_id) != case.account_id,
        "links": links,
        "story": link_story(links, lang, t),
        "image_url": reverse("case_image", args=[case.id]) if case.image_path else "",
        "feedback": Feedback.objects.filter(case_ref=case.id)[:10],
        "verdicts": [(v, t[f"fb_{v}"]) for v, _ in Feedback.VERDICTS],
        "fb_saved": request.GET.get("fb") == "1",
        **ctx,
    })


@require_POST
@permission_required("dashboard.add_feedback", raise_exception=True)
def case_feedback(request, case_id: str):
    case = get_object_or_404(Case, pk=case_id)
    verdict = request.POST.get("verdict", "")
    if verdict not in dict(Feedback.VERDICTS):
        return HttpResponseBadRequest("unknown verdict")
    record_feedback(case, verdict, request.POST.get("comment", ""), user=request.user)
    audit(request, "feedback", case=case.id, verdict=verdict)
    return redirect(reverse("case", args=[case.id]) + "?fb=1")


@can_view
def case_image(request, case_id: str):
    case = get_object_or_404(Case, pk=case_id)
    path = (paths.ROOT / case.image_path).resolve()
    allowed = {paths.IMAGES_DIR.resolve(), DEMO_DIR.resolve()}
    if not case.image_path or not allowed.intersection(path.parents) or not path.exists():
        raise Http404("no image")
    return FileResponse(path.open("rb"), content_type="image/png")


# -------------------------------------------------------------------- check
def _fetch_error(request, ctx: dict, t: dict, exc: FetchError):
    ctx["error"] = t[exc.code]
    return render(request, "dashboard/check.html", ctx, status=503 if exc.code == "fetch_failed" else 422)


def _check_channel(request, ctx: dict, lang: str, t: dict):
    """Check a Telegram channel at once: its latest posts and its description, all saved to the queue."""
    if not request.user.has_perm("dashboard.add_case"):
        raise PermissionDenied
    link = ctx["form"]["source_url"]
    if not is_telegram_link(link):
        ctx["error"] = t["fetch_need_link"]
        return render(request, "dashboard/check.html", ctx, status=422)
    try:
        channel = fetch_channel(parse_link(link)[0])
    except FetchError as exc:
        return _fetch_error(request, ctx, t, exc)
    items = [({"id": p.case_id, "text": p.text, "source_type": "post", "platform": "telegram",
               "account_id": channel.account}, p.url, p.published_at) for p in channel.posts]
    if channel.about_text:
        items.append(({"id": f"tg-{channel.channel}-about"[:64], "text": channel.about_text,
                       "source_type": "account_bio", "platform": "telegram", "account_id": channel.account},
                      channel.url, None))
    results = analyze_many([item for item, _, _ in items], use_ocr=False, network_lookup=live_lookup())
    visible = {p.case_id: p.visible_text for p in channel.posts}
    for (_, url, published), result in zip(items, results):
        case = save_case(result, origin="check", source_url=url, published_at=published)
        absorb_manual_copies(case, visible.get(case.pk, ""))
    audit(request, "check_channel", channel=channel.account, posts=len(channel.posts))
    model = live_graph()
    cluster = model.clusters.get(model.account_cluster.get(channel.account, "")) if model is not None else None
    levels = Counter(r.risk_level for r in results)
    ctx["channel_report"] = {
        "account": channel.account,
        "title": channel.title,
        "url": channel.url,
        "saved_text": t["channel_saved"].format(n=len(channel.posts)),
        "levels": [(lvl, t[f"level_{lvl}"], levels.get(lvl, 0)) for lvl in ("high", "medium", "low")],
        "rows": _rows(Case.objects.filter(pk__in=[r.id for r in results]).order_by("-risk_score"), model),
        "network": network_summary(cluster, lang, t) if cluster is not None else None,
    }
    ctx["recent"] = _rows(Case.objects.filter(origin="check").order_by("-analysed_at", "id")[:RECENT_CHECKS])
    return render(request, "dashboard/check.html", ctx)


@can_view
def check(request):
    lang, t = _t(request)
    demo = settings.SAFENET_DEMO_MODE
    ctx: dict = {
        "active": "check",
        "examples": [(k, i18n.EXAMPLE_NAMES[lang][k]) for k in i18n.EXAMPLES] if demo else [],
        "sources": list(i18n.SOURCES[lang].items()),
        "platforms": [(k, i18n.PLATFORM_NAMES[k]) for k in i18n.PLATFORMS],
        # checked content goes to the queue unless the analyst unticks it, so it can be found again
        "form": {"text": i18n.EXAMPLES.get(request.GET.get("example", ""), "") if demo else "",
                 "source_type": "post", "platform": "", "account": "", "group": "", "source_url": "",
                 "save": request.user.has_perm("dashboard.add_case")},
    }
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        source_type = request.POST.get("source_type") or "post"
        save = request.POST.get("save") == "1"
        if save and not request.user.has_perm("dashboard.add_case"):
            raise PermissionDenied
        ctx["form"] = {"text": text, "source_type": source_type, "platform": request.POST.get("platform", ""),
                       "save": save, **{k: request.POST.get(k, "").strip() for k in ("account", "group", "source_url")}}
        if request.POST.get("action") == "channel":
            return _check_channel(request, ctx, lang, t)
        upload = request.FILES.get("image")
        if upload and upload.size > settings.SAFENET_MAX_IMAGE_BYTES:
            ctx["error"] = t["image_too_large"]
            return render(request, "dashboard/check.html", ctx, status=422)
        image_bytes = upload.read() if upload else None
        if image_bytes:
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError):
                ctx["error"] = t["bad_image"]
                return render(request, "dashboard/check.html", ctx, status=422)
        # a t.me post link: one post is one case; without pasted text the post is read from Telegram
        post = None
        telegram = parse_link(ctx["form"]["source_url"]) if is_telegram_link(ctx["form"]["source_url"]) else None
        if telegram and not text and not image_bytes:
            if telegram[1] is None:
                ctx["error"] = t["fetch_need_post"]
                return render(request, "dashboard/check.html", ctx, status=422)
            try:
                post = fetch_post(*telegram)
            except FetchError as exc:
                return _fetch_error(request, ctx, t, exc)
            text = ctx["form"]["text"] = post.text
            ctx["form"]["source_url"] = post.url
            ctx["fetched"] = True
        try:
            context = build_context(account=ctx["form"]["account"], group=ctx["form"]["group"],
                                    source_url=ctx["form"]["source_url"], platform=ctx["form"]["platform"])
        except ContextError:
            ctx["error"] = t["bad_url"]
            return render(request, "dashboard/check.html", ctx, status=422)
        if not text and not image_bytes:
            ctx["error"] = t["empty_input"]
            return render(request, "dashboard/check.html", ctx, status=422)
        item = {"text": text, "source_type": source_type if text or not image_bytes else "screenshot",
                "platform": context.platform or None, "account_id": context.account_id or None}
        result = analyze(item, image=image_bytes, network_lookup=live_lookup())
        if image_bytes and len(image_bytes) < 4 * 1024 * 1024:
            ctx["image_url"] = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        if save:
            if telegram and telegram[1] is not None:
                result.id = f"tg-{telegram[0]}-{telegram[1]}"[:64]
            else:
                # the same text from another account or post is another case (copies are what links a network)
                key = text + (result.ocr_text or "")
                if context.account_id or context.source_url:
                    key = "\n".join((key, context.account_id, context.source_url))
                result.id = f"check-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]}"
            case = save_case(result, origin="check", group_id=context.group_id, source_url=context.source_url,
                             published_at=post.published_at if post else None)
            if telegram and telegram[1] is not None:
                absorb_manual_copies(case, post.visible_text if post else text)
            ctx["saved_case"] = result.id
        audit(request, "check", chars=len(text), image=bool(image_bytes), level=result.risk_level,
              saved=ctx.get("saved_case", "-"), fetched=bool(post))
        ctx.update(card_context(result, lang, t))
        links = links_for(account_id=context.account_id, text="\n".join(p for p in (text, result.ocr_text or "") if p),
                          case_id=ctx.get("saved_case"), group_id=context.group_id)
        ctx["links"], ctx["story"] = links, link_story(links, lang, t)
    ctx["recent"] = _rows(Case.objects.filter(origin="check").order_by("-analysed_at", "id")[:RECENT_CHECKS])
    return render(request, "dashboard/check.html", ctx)


# ------------------------------------------------------------------ network
@can_view
def network(request):
    """Networks of the live graph (real saved cases only, see ``app.dashboard.network``)."""
    lang, t = _t(request)
    model = live_graph()
    clusters = list(model.clusters.values()) if model is not None else []
    ctx: dict = {"active": "network", "clusters": clusters, "cards": [network_summary(c, lang, t) for c in clusters]}
    selected_id = request.GET.get("cluster") or (clusters[0].cluster_id if clusters else None)
    selected = model.clusters.get(selected_id) if model is not None and selected_id else None
    if selected:
        accounts = []
        for key in selected.accounts:
            post = key.removeprefix("post:") if key.startswith("post:") else None
            label = account_label(key)
            accounts.append({"label": label, "case": post,
                             "query": urlencode({"account": key, "sort": "new"}) if label == key else ""})
        ctx.update({
            "selected": selected,
            "summary": network_summary(selected, lang, t),
            "shared": [e for e in selected.shared if e.kind != "group"],
            "groups": [e for e in selected.shared if e.kind == "group"],
            "accounts": accounts,
            "cases": Case.objects.filter(pk__in=selected.record_ids).order_by("-risk_score")[:30],
            "graph_url": reverse("network_graph", args=[selected.cluster_id]),
        })
    return render(request, "dashboard/network.html", ctx)


@xframe_options_sameorigin
@can_view
def network_graph(request, cluster_id: str):
    from safenet.graph.viz import cluster_html

    model = live_graph()
    if model is None or cluster_id not in model.clusters:
        raise Http404("unknown cluster")
    return HttpResponse(cluster_html(model, cluster_id))


# ------------------------------------------------------------------ metrics
def _markdown(path) -> str:
    html = markdown.markdown(path.read_text(encoding="utf-8"), extensions=["tables"])
    html = re.sub(r'src="([\w.-]+\.png)"', lambda m: f'src="{reverse("report_file", args=[m.group(1)])}"', html)
    return re.sub(r'href="errors\.md"', f'href="{reverse("report_file", args=["errors.md"])}"', html)


@can_view
def metrics(request):
    lang, t = _t(request)
    md_path = paths.REPORTS_DIR / "metrics.md"
    json_path = paths.REPORTS_DIR / "metrics.json"
    ctx: dict = {"active": "metrics", "available": md_path.exists() and json_path.exists()}
    if ctx["available"]:
        summary = json.loads(json_path.read_text(encoding="utf-8"))
        ctx["cards"] = [
            {"split": split, **{k: summary["hybrid"][split][k] for k in
                                ("precision", "recall", "f1", "hard_negative_fpr")}}
            for split in ("test", "gold")
        ]
        ctx["report_html"] = _markdown(md_path)  # embeds confusion matrices, PR curve and ablation
    return render(request, "dashboard/metrics.html", ctx)


@can_view
def report_file(request, name: str):
    if name in REPORT_IMAGES and (paths.REPORTS_DIR / name).exists():
        return FileResponse((paths.REPORTS_DIR / name).open("rb"), content_type="image/png")
    if name == "errors.md" and (paths.REPORTS_DIR / name).exists():
        return render(request, "dashboard/metrics.html", {
            "active": "metrics", "available": True, "report_html": _markdown(paths.REPORTS_DIR / name),
            "only_report": True,
        })
    raise Http404("unknown report")
