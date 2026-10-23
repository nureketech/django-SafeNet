"""``python manage.py purge_cases --origin dataset demo [--yes]`` — remove cases from the queue.

Before going live, the synthetic evaluation cases (``dataset``) and showcase cases (``demo``)
must be removed. Verdicts stay in the history (their case link is cleared).
"""

from django.core.management.base import BaseCommand, CommandError

from app.dashboard.models import Case


class Command(BaseCommand):
    help = "Delete queue cases of the given origins (dataset, demo, check, api)"

    def add_arguments(self, parser):
        parser.add_argument("--origin", nargs="+", required=True, choices=[o for o, _ in Case.ORIGINS])
        parser.add_argument("--yes", action="store_true", help="really delete (otherwise only count)")

    def handle(self, *args, origin, yes, **options):
        qs = Case.objects.filter(origin__in=origin)
        n = qs.count()
        if not yes:
            self.stdout.write(f"{n} cases of origin {', '.join(origin)} would be deleted; add --yes to delete")
            return
        if n and not qs.delete()[0]:
            raise CommandError("nothing was deleted")
        self.stdout.write(self.style.SUCCESS(f"deleted {n} cases of origin {', '.join(origin)}"))
