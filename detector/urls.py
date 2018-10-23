from django.urls import path, include
from .views import FrameReceiver, index_view
urlpatterns = [
    path('frame-receiver',FrameReceiver.as_view()),
    path('',index_view),

]
