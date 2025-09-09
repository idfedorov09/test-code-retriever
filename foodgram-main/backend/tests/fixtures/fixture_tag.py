import pytest

from tests.utils.models import tag_model
from tests.utils.tag import TAG_DATA

Tag = tag_model()


@pytest.fixture
def tags():
    tags = [Tag(**item) for item in TAG_DATA]
    Tag.objects.bulk_create(tags)
    return list(Tag.objects.all())
