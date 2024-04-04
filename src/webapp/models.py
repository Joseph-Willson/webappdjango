import csv
from django.db import models

class Pizza(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    price = models.DecimalField(max_digits=6, decimal_places=2)

    @classmethod
    def import_from_csv(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                _, created = cls.objects.get_or_create(
                    name=row[0],
                    description=row[1],
                    price=row[2]
                )
