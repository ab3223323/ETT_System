from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from .forms import *
from django.views.decorators.csrf import csrf_exempt
from .predict_model import *
  
# Create your views here.
@csrf_exempt
def index(request):
    if request.method == 'POST':
        form = UploadModelForm(request.POST, request.FILES)
  
        if form.is_valid():
            form.save()
            fileName = request.POST.get("fileName")
            # ETT預測
            ETT_result_1,ETT_result_2,ETT_result_3,ETT_result_4,ettImg =define_dataset(fileName)
            resultImg = ensemble_ett(ETT_result_1,ETT_result_2,ETT_result_3,ETT_result_4)
            ETT_PrintPoint(resultImg,ettImg,fileName)

            #return redirect('success')
            return render(request, 'index.html',{'form' : form} )  
    else:
        form = UploadModelForm()
    return render(request, 'index.html',{'form' : form} )  
  
  
def success(request):
    return HttpResponse('successfully uploaded')