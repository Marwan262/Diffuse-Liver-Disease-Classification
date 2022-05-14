from django.contrib import admin

# Register your models here.
from .models import Patient
from .models import Doctor
from .models import Admin
from .models import Report
from .models import Scan

admin.site.register(Patient)
admin.site.register(Doctor)
admin.site.register(Admin)
admin.site.register(Report)
admin.site.register(Scan)