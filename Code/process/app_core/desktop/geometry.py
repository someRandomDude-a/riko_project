"""Shared validation for saved settings, API requests and model window tools."""
from __future__ import annotations


def validate_geometry(target, geometry, displays=()):
    if target not in {'avatar', 'whiteboard'}:
        raise ValueError('Unknown surface')
    if not isinstance(geometry, dict) or set(geometry) - {'x', 'y', 'width', 'height', 'screen'}:
        raise ValueError('Invalid geometry fields')
    if any(type(value) is not int for value in geometry.values()):
        raise ValueError('Geometry must use integers')
    minimum = 100 if target == 'avatar' else 200
    if any(not minimum <= geometry[key] <= 4096 for key in ('width', 'height') if key in geometry):
        raise ValueError(f'Surface size must be {minimum}–4096')
    if geometry.get('screen', 0) < 0:
        raise ValueError('Screen index must be nonnegative')
    if 'screen' in geometry and displays and geometry['screen'] not in {display['index'] for display in displays}:
        raise ValueError('Unknown display index; inspect runtime displays')
