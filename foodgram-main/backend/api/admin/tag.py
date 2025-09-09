from django.contrib import admin

from api.models.tag import Tag


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    """Страничка управления тегами в админке."""

    list_display = ('name', 'slug')
    search_fields = ('name', 'slug')
    ordering = ('name',)
