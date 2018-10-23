from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from detector import trainer

class Command(BaseCommand):
    help = 'Trains MOG'

    # def add_arguments(self, parser):
    #     parser.add_argument('poll_id', nargs='+', type=int)

    def handle(self, *args, **options):
            settings.MOG = trainer.fun().copy()
            self.stdout.write(self.style.SUCCESS('Successfully trained and attached to settings'))