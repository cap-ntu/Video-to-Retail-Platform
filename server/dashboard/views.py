"""
Hysia dashboard view
Author: Jiang Wenbo
Email: jian0085@e.ntu.edu.sg
"""
from django.contrib.auth.decorators import user_passes_test
# Create your views here.
from django.shortcuts import render, redirect
from django.urls import reverse_lazy


@user_passes_test(lambda u: u.is_authenticated and u.is_staff or not u.is_authenticated)
def index(request):
    if request.user.is_authenticated:
        return render(request, template_name="index.html")
    else:
        return redirect(reverse_lazy("login"))
