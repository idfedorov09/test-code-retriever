import pytest

from users.models.subscription import Subscription


@pytest.fixture
def third_user_subscribed_to_first(
    first_user, third_user
):
    return Subscription.objects.create(
        author_recipe=first_user,
        user=third_user
    )


@pytest.fixture
def third_user_subscribed_to_second(
    second_user, third_user
):
    return Subscription.objects.create(
        author_recipe=second_user,
        user=third_user
    )


@pytest.fixture
def third_user_subscriptions(
    third_user_subscribed_to_first, third_user_subscribed_to_second
) -> list:
    return [third_user_subscribed_to_first, third_user_subscribed_to_second]
