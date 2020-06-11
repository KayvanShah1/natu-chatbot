from django.shortcuts import render, redirect, HttpResponse

# Create your views here.
def homepage(request):
    return render(request, 'homepage.html')