# Generated by Django 2.1.4 on 2018-12-17 01:56

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Plant_Info',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('plant_no', models.IntegerField()),
                ('plant_type', models.CharField(max_length=50)),
                ('condition', models.CharField(max_length=50)),
                ('disease', models.TextField()),
                ('diagnosis', models.TextField()),
            ],
            options={
                'ordering': ('scan_no',),
            },
        ),
        migrations.CreateModel(
            name='Scan',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.BooleanField(default=True)),
                ('date', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'ordering': ('date',),
            },
        ),
        migrations.AddField(
            model_name='plant_info',
            name='scan_no',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='serializer.Scan'),
        ),
    ]
