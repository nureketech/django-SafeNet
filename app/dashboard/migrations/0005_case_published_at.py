from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0004_case_context'),
    ]

    operations = [
        migrations.AddField(
            model_name='case',
            name='published_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
