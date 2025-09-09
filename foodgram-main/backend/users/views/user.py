from djoser import views as djoser_views
from rest_framework import response, status
from rest_framework.decorators import action
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request

from users.models import User
from users.serializers import AvatarSerializer, UserSerializer
from users.views.subscription import SubscriptionMixin


class UserViewSet(djoser_views.UserViewSet, SubscriptionMixin):
    """Вьюсет пользователей."""

    queryset = User.objects.all()
    serializer_class = UserSerializer
    pagination_class = PageNumberPagination
    pagination_class.page_size_query_param = 'limit'

    @action(
        ["GET", "PUT", "PATCH", "DELETE"],
        detail=False,
        permission_classes=[IsAuthenticated]
    )
    def me(self, request, *args, **kwargs):
        return super().me(request, *args, **kwargs)

    @action(
        ['PUT'],
        detail=False,
        url_path='me/avatar',
        name='set_avatar',
        permission_classes=[IsAuthenticated]
    )
    def avatar(self, request: Request, *args, **kwargs):
        if 'avatar' not in request.data:
            return response.Response(
                {'avatar': 'Отсутствует изображение'},
                status=status.HTTP_400_BAD_REQUEST
            )

        serializer = AvatarSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        avatar_data = serializer.validated_data.get('avatar')
        request.user.avatar = avatar_data
        request.user.save()

        image_url = request.build_absolute_uri(
            f'/media/users/{avatar_data.name}'
        )
        return response.Response(
            {'avatar': str(image_url)}, status=status.HTTP_200_OK
        )

    @avatar.mapping.delete
    def delete_avatar(self, request: Request, *args, **kwargs):
        self.request.user.avatar = None
        self.request.user.save()
        return response.Response(status=status.HTTP_204_NO_CONTENT)
