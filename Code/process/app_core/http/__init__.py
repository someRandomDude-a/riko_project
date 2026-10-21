"""The backend's HTTP layer: app.create_app assembles one router module per URL domain over a Backend (backend.py), which
routes read through the `Services` dependency from app.state. Nothing else in app_core imports this package; the entry point
Code/desktop_server.py supplies the composition root's factories."""
