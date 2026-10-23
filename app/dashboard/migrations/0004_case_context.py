from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0003_case_analysed_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='case',
            name='group_id',
            field=models.CharField(blank=True, db_index=True, max_length=100),
        ),
        migrations.AddField(
            model_name='case',
            name='source_url',
            field=models.URLField(blank=True, max_length=500),
        ),
    ]
