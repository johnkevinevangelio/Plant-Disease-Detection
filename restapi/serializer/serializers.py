from rest_framework import serializers
from .models import Scan, Plant_Info


class Scan_Serializer(serializers.ModelSerializer):
	class Meta:
		model = Scan
		fields = ('id', 'status', 'date')

class Plant_Info_Serializer(serializers.ModelSerializer):
	class Meta:
		model = Plant_Info
		fields = ('id', 'scan_no', 'plant_no', 'plant_type', 'condition', 'disease', 'diagnosis')



