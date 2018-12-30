from rest_framework import serializers
from .models import Scan, Plant_Info


##class Scan_Serializer(serializers.ModelSerializer):
##	class Meta:
##		model = Scan
##		fields = ('id', 'status', 'date')
##


class Plant_Info_Serializer(serializers.ModelSerializer):
        class Meta:
                model = Plant_Info
                fields = ('id', 'plant_no', 'condition', 'disease', 'diagnosis','model_pic')


class Scan_Serializer(serializers.ModelSerializer):
        scan_details = Plant_Info_Serializer(many=True)
        class Meta:
                model = Scan
                fields = ('id','status','date','scan_details',)
