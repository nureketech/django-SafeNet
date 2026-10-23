"""Dashboard pages: case queue, case card, content check, network, metrics."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re

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
from app.dashboard.present import card_context
from app.dashboard.services import record_feedback, save_case
from safenet import LABELS, paths
from safenet.demo import DEMO_DIR
from safenet.pipeline import analyze

can_view = permission_required("dashboard.view_case", raise_exception=True)
PAGE_SIZE = 50
FILTERS = {"track": "primary_track", "lang": "lang", "platform": "platform", "source": "source_type",
           "level": "risk_level", "origin": "origin"}
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
    counts = {row["risk_level"]: row["n"] for row in Case.objects.values("risk_level").annotate(n=Count("id"))}
    page = Paginator(qs, PAGE_SIZE).get_page(request.GET.get("page"))
    rows = []
    for case in page.object_list:
        snippet = case.text or (case.result.get("ocr_text") or "")
        rows.append({"case": case, "snippet": snippet[:180] + ("…" if len(snippet) > 180 else "")})
    query = "&".join(f"{k}={v}" for k, v in selected.items())
    return render(request, "dashboard/queue.html", {
        "active": "queue",
        "rows": rows,
        "page": page,
        "counts": {lvl: counts.get(lvl, 0) for lvl in ("high", "medium", "low")},
        "total": sum(counts.values()),
        "filtered": qs.count(),
        "demo_count": Case.objects.filter(origin="demo").count(),
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
    return render(request, "dashboard/case.html", {
        "active": "queue",
        "case": case,
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
@can_view
def check(request):
    lang, t = _t(request)
    demo = settings.SAFENET_DEMO_MODE
    ctx: dict = {
        "active": "check",
        "examples": [(k, i18n.EXAMPLE_NAMES[lang][k]) for k in i18n.EXAMPLES] if demo else [],
        "sources": list(i18n.SOURCES[lang].items()),
        "platforms": [(k, i18n.PLATFORM_NAMES[k]) for k in i18n.PLATFORMS],
        "form": {"text": i18n.EXAMPLES.get(request.GET.get("example", ""), "") if demo else "",
                 "source_type": "post", "platform": "", "save": False},
    }
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        source_type = request.POST.get("source_type") or "post"
        platform = request.POST.get("platform") or None
        save = request.POST.get("save") == "1"
        if save and not request.user.has_perm("dashboard.add_case"):
            raise PermissionDenied
        ctx["form"] = {"text": text, "source_type": source_type, "platform": platform or "", "save": save}
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
        if not text and not image_bytes:
            ctx["error"] = t["empty_input"]
            return render(request, "dashboard/check.html", ctx, status=422)
        item = {"text": text, "source_type": source_type if text or not image_bytes else "screenshot",
                "platform": platform}
        result = analyze(item, image=image_bytes)
        if image_bytes and len(image_bytes) < 4 * 1024 * 1024:
            ctx["image_url"] = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        if save:
            digest = hashlib.sha1((text + (result.ocr_text or "")).encode("utf-8")).hexdigest()[:10]
            result.id = f"check-{digest}"
            save_case(result, origin="check")
            ctx["saved_case"] = result.id
        audit(request, "check", chars=len(text), image=bool(image_bytes), level=result.risk_level,
              saved=ctx.get("saved_case", "-"))
        ctx.update(card_context(result, lang, t))
    return render(request, "dashboard/check.html", ctx)


# ------------------------------------------------------------------ network
@can_view
def network(request):
    from safenet.graph.lookup import load_graph

    lang, t = _t(request)
    model = load_graph()
    ctx: dict = {"active": "network", "graph": model is not None}
    if model is None:
        return render(request, "dashboard/network.html", ctx)
    clusters = list(model.clusters.values())
    selected_id = request.GET.get("cluster") or (clusters[0].cluster_id if clusters else None)
    selected = model.clusters.get(selected_id) if selected_id else None
    ctx["clusters"] = clusters
    if selected:
        cases = Case.objects.filter(account_id__in=selected.accounts).order_by("-risk_score")[:30]
        ctx.update({
            "selected": selected,
            "shared": [e for e in selected.shared if e.kind != "group"],
            "groups": [e for e in selected.shared if e.kind == "group"],
            "cases": cases,
            "graph_url": reverse("network_graph", args=[selected.cluster_id]),
        })
    return render(request, "dashboard/network.html", ctx)


@xframe_options_sameorigin
@can_view
def network_graph(request, cluster_id: str):
    from safenet.graph.lookup import load_graph
    from safenet.graph.viz import cluster_html

    model = load_graph()
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
