# Generated by Django 3.2.12 on 2022-02-18 21:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0007_report_classification'),
    ]

    operations = [
        migrations.AlterField(
            model_name='report',
            name='image',
            field=models.ImageField(upload_to='static/images/% Y/% m/% d/'),
        ),
    ]