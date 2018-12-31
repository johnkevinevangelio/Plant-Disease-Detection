from django.shortcuts import render

from rest_framework import status, viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Scan, Plant_Info
from .serializers import Scan_Serializer, Plant_Info_Serializer


class ItemViewSet(viewsets.ModelViewSet):
    queryset = Scan.objects.all()
    serializer_class = Scan_Serializer


class Plant_Infos(viewsets.ModelViewSet):
    queryset = Plant_Info.objects.all()
    serializer_class = Plant_Info_Serializer

    
