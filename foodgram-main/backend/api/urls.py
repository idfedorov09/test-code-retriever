from django.urls import include, path
from rest_framework.routers import DefaultRouter

from api.views import IngredientViewSet, RecipeViewSet, TagViewSet
from users.views import UserViewSet

api_v1 = DefaultRouter()
api_v1.register('tags', TagViewSet)
api_v1.register('ingredients', IngredientViewSet)
api_v1.register('recipes', RecipeViewSet)
api_v1.register(r'users', UserViewSet, basename='users')

urlpatterns = [
    path('', include(api_v1.urls)),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.authtoken'))
]
