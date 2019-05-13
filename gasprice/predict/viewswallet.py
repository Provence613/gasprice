from django.shortcuts import render
def wallet(request):
    return render(request,"clara/wallet.html")
def pwallet(request):
    return render(request,"wallet.html")