"""``python manage.py load_cases [--split test|val|train|all|gold] [--limit N] [--if-empty] [--replace]``."""

from django.core.management.base import BaseCommand

from app.dashboard.models import Case
from app.dashboard.services import load_dataset_cases


class Command(BaseCommand):
    help = "Analyse dataset records and put them into the analyst queue"

    def add_arguments(self, parser):
        parser.add_argument("--split", default="test", choices=["train", "val", "test", "all", "gold"])
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--if-empty", action="store_true", help="do nothing when the queue already has cases")
        parser.add_argument("--replace", action="store_true", help="remove previously loaded dataset cases first")

    def handle(self, *args, split, limit, if_empty, replace, **options):
        if if_empty and Case.objects.exists():
            self.stdout.write(f"queue already has {Case.objects.count()} cases")
            return
        n = load_dataset_cases(split, limit, replace)
        self.stdout.write(self.style.SUCCESS(f"loaded {n} cases from '{split}'"))
