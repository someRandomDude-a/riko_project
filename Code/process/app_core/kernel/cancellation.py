"""One vocabulary for work that stopped on purpose, whichever provider or package stopped it."""


class Deferred(Exception):
    """Stopped on purpose (a Stop, a live turn, shutdown), not failed: drop or retry it, never report it as an error."""


class TurnCancelled(Deferred):
    """Expected user interruption, not a provider failure."""


class BackgroundPreempted(Deferred, RuntimeError):
    """Background inference (initiative, reflection) gave way to a live turn, a cancel or shutdown; retry it later."""


class TurnBusy(RuntimeError):
    """Another foreground turn holds the session: a conflict to retry (every route answers 409), not a failure. The
    message is the one SessionManager always raised, so a client that still compares the text keeps working."""
    MESSAGE = 'Riko is already handling another turn'
    def __init__(self, message=MESSAGE): super().__init__(message)
