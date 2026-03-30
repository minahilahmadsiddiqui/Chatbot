from __future__ import annotations

from django.test.runner import DiscoverRunner


class NoDurationsDiscoverRunner(DiscoverRunner):
    """
    Django 6 passes `durations` into unittest.TextTestRunner.
    In some Python/unittest builds, TextTestRunner does not accept that
    keyword argument, causing `manage.py test` to crash.
    """

    def get_test_runner_kwargs(self):
        kwargs = super().get_test_runner_kwargs()
        # Remove incompatible kwarg for older unittest.TextTestRunner signatures.
        kwargs.pop("durations", None)
        return kwargs

