"""``python manage.py create_api_token <name> [--user <username>]`` — issue an API token.

The token is printed once; only its SHA-256 hash is stored. Revoke it in ``/admin/`` (API tokens).
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from app.dashboard.models import ApiToken


class Command(BaseCommand):
    help = "Issue a bearer token for an integration (shown once)"

    def add_arguments(self, parser):
        parser.add_argument("name", help="who uses the token, e.g. telegram-collector")
        parser.add_argument("--user", default=None, help="account responsible for the integration")

    def handle(self, *args, name, user, **options):
        owner = None
        if user:
            try:
                owner = get_user_model().objects.get_by_natural_key(user)
            except get_user_model().DoesNotExist as exc:
                raise CommandError(f"no such user: {user}") from exc
        token, key = ApiToken.issue(name.strip(), owner)
        self.stdout.write(self.style.SUCCESS(f"token for «{token.name}» (prefix {token.key_prefix}):"))
        self.stdout.write(key)
        self.stdout.write("Store it now: it is not saved in plain text and cannot be shown again.")
