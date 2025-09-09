from typing import Optional

from django.contrib import admin
from django.forms import ModelForm, PasswordInput
from django.http.response import HttpResponse
from rest_framework.request import Request

from api.models import RecipeFavorite, ShoppingCart
from users.models.user import User


class UserAdminForm(ModelForm):
    """Форма пользователя со скрытым паролем."""

    class Meta:
        model = User
        fields = '__all__'

    # Переопределяем метод, чтобы скрыть пароль
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.instance.pk:  # Только для создания
            self.fields['password'].widget = PasswordInput(
                attrs={'class': 'vTextField'}  # Длина поля для админки
            )


class RecipeFavoriteInline(admin.TabularInline):
    """Инлайн-блок управления избранными рецептами из пользователя."""

    model = RecipeFavorite
    autocomplete_fields = ['recipe']
    extra = 1


class ShoppingCartInline(admin.TabularInline):
    """Инлайн-блок управления корзиной из пользователя."""

    model = ShoppingCart
    autocomplete_fields = ['recipe']
    extra = 1


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    """
    Страничка управления пользователями в админке.

    Включает в себя управление связаннами с пользователями объектами, а именно:
    - Управление избранными рецептами;
    - Управление корзиной;
    """

    form = UserAdminForm
    list_display = (
        'username', 'email', 'get_full_name', 'last_login',
        'is_active', 'is_staff'
    )
    search_fields = ('username', 'email')
    list_filter = ('is_active', 'is_staff')
    ordering = ('id', )

    readonly_fields = ('date_joined', 'last_login')
    exclude = ('groups', 'user_permissions')

    def set_fieldsets(
        self, enabled_password: bool = True, fields: list = []
    ) -> None:
        """
        Функция устанавливает форму в зависимости от переданных данных.

        Параметры:
        * enabled_password - Требуется ли в форме пароль. По умолчанию True
        * fields - Отображаемые необязательные поля. По умолчанию пустой
        список. В отображаемых полях всегда есть date_joined.
        """
        self.fieldsets = [
            (
                'Персональные данные', {
                    'fields': (
                        'username', 'email', 'first_name', 'last_name',
                        *(['password'] if enabled_password else [])
                    )
                }
            ),
            (
                'Признаки', {
                    'fields': ('is_active', 'is_staff', 'is_superuser')
                }
            ),
            (
                'Служебная информация', {
                    'fields': ('date_joined', *fields)
                }
            )
        ]

    def add_view(
        self, request: Request, extra_content: Optional[dict] = None
    ) -> HttpResponse:
        self.set_fieldsets()
        return super(UserAdmin, self).add_view(request)

    def change_view(
        self,
        request: Request,
        object_id: int,
        extra_content: Optional[dict] = None
    ) -> HttpResponse:
        self.set_fieldsets(enabled_password=False, fields=['last_login'])
        self.inlines = [RecipeFavoriteInline, ShoppingCartInline]
        return super(UserAdmin, self).change_view(request, object_id)
