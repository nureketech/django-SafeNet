"""``python manage.py load_demo`` — the five showcase cases of ``config/demo_cases.yaml``."""

from django.core.management.base import BaseCommand
from django.db import transaction

from app.dashboard.models import Case
from safenet import paths
from safenet.demo import analyse, load_cases
from safenet.multimodal import ocr as ocr_module


class Command(BaseCommand):
    help = "Analyse the prepared demo cases and put them into the analyst queue (origin=demo)"

    def handle(self, *args, **options):
        cases = load_cases()
        ocr_ok = ocr_module.available_engine()[0] is not None
        analysed = analyse(cases)
        with transaction.atomic():
            Case.objects.filter(origin="demo").delete()
            Case.objects.filter(pk__in=[c.id for c in cases]).delete()
            for case, result in analysed:
                image = str(case.image_path.relative_to(paths.ROOT).as_posix()) if case.image_path else ""
                Case.from_result(result, image_path=image, origin="demo").save()
        for case, result in analysed:
            expected = case.expected(ocr_ok)
            line = (f"{case.id:18} {result.risk_score:3d}  {result.risk_level:6}  {result.primary_track:8}  "
                    f"{case.title_kk}")
            if result.risk_level == expected:
                self.stdout.write(line)
            else:
                self.stdout.write(self.style.WARNING(f"{line}  (expected {expected})"))
        self.stdout.write(self.style.SUCCESS(f"loaded {len(analysed)} demo cases"))
