from django.db import models


class Email(models.Model):
    ttype = models.IntegerField()
    mail = models.TextField()
    converted_text = models.BinaryField()
    file_name = models.CharField(max_length=100, default="")

    def description(self):
        return str(self.mail)[0:30]
