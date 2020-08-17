from django.shortcuts import render, HttpResponse
from .models import Email
import pickle
import base64
import numpy as np
import random

from django.views.generic import DetailView


emails = Email.objects.all()

from .predictor import Predictor

newP = Predictor()
newP.make_data()


def home(request):
    random_emails = random.sample(list(emails), 10)

    context = {"emails": random_emails}
    return render(request, "filter/home.html", context)


class EmailDetailView(DetailView):
    model = Email


def verify_email(request, pk):
    email = emails.get(pk=pk)
    result = newP.result(email.file_name)

    if int(result) == 1:
        result = "SPAM"
    else:
        result = "HAM"

    return render(request, "filter/robo.html", {"type": result},)


def spam(request):
    spam_emails = emails.filter(ttype=1)
    random_spam_emails = random.sample(list(spam_emails), 5)

    context = {"emails": random_spam_emails}
    return render(request, "filter/home.html", context)


def ham(request):
    ham_emails = emails.filter(ttype=0)
    random_ham_emails = random.sample(list(ham_emails), 5)

    context = {"emails": random_ham_emails}
    return render(request, "filter/home.html", context)


def about(request):
    # new_types = list(newP.sample_target)

    # for i in range(len(newP.list_of_email)):
    #     email, filename = newP.list_of_email[i]
    #     ttype = new_types[i]

    #     np_bytes = pickle.dumps(np.array(newP.sample_texts)[i])
    #     np_base64 = base64.b64encode(np_bytes)

    #     newEmail = Email(
    #         ttype=ttype, mail=email, converted_text=np_base64, file_name=filename
    #     )
    #     newEmail.save()

    return render(request, "filter/about.html", {"title": "About"})
