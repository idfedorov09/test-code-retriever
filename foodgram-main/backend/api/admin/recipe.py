from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from api.models.recipe import Recipe
from api.models.recipe_favorite import RecipeFavorite
from api.models.recipe_ingredients import RecipeIngredients
from api.models.recipe_tags import RecipeTags


class RecipeIngredientInline(admin.TabularInline):
    """Инлайн-блок управления ингредиентами из рецептов."""

    model = RecipeIngredients
    autocomplete_fields = ['ingredient']
    extra = 1


class RecipeTagInline(admin.TabularInline):
    """Инлайн-блок управления тегами из рецептов."""

    model = RecipeTags
    autocomplete_fields = ['tag']
    extra = 1


@admin.register(Recipe)
class RecipeAdmin(admin.ModelAdmin):
    """
    Страничка управления рецептами в админке.

    Включает в себя управление связаннами с рецептами объектами, а именно:
    - Управление тегами рецептов;
    - Управление ингредиентами рецептов;
    """

    list_display = (
        'name', 'get_author_recipe', 'pub_date', 'get_favorite_count',
    )
    search_fields = (
        'name', 'author__username', 'author__first_name', 'author__last_name'
    )
    list_filter = ('tags',)
    inlines = [RecipeIngredientInline, RecipeTagInline]
    autocomplete_fields = ('author', )
    readonly_fields = ('short_link', 'pub_date')
    ordering = ('id',)

    def get_author_recipe(self, obj: Recipe):
        # Создаем ссылку на редактирование автора рецепта
        url = reverse('admin:users_user_change', args=[obj.author.id])
        return format_html(
            '<a href="{}">{}</a>', url, obj.author.__str__()
        )

    def get_favorite_count(self, obj: Recipe):
        return RecipeFavorite.objects.filter(recipe=obj).count()

    get_author_recipe.short_description = 'Автор рецепта'
    get_favorite_count.short_description = 'В избранном у ...'
