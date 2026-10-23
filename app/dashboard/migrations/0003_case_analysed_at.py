import django.utils.timezone
from django.db import migrations, models


def backfill(apps, schema_editor):
    Case = apps.get_model("dashboard", "Case")
    Case.objects.update(analysed_at=models.F("created_at"))


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0002_production_auth'),
    ]

    operations = [
        migrations.AddField(
            model_name='case',
            name='analysed_at',
            field=models.DateTimeField(auto_now=True, db_index=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]
