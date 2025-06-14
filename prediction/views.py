from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
def predict_view(request):
    # load model, predict, return result
    return JsonResponse({'result': 'ok'})
