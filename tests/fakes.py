"""Helper di test condivisi."""


class FakeTime:
    """Orologio finto: `sleep` fa avanzare il tempo invece di bloccare."""

    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds

    def advance(self, seconds):
        self.value += seconds
