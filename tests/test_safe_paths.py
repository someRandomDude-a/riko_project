"""kernel/safe_paths.py: paths from the renderer, the model and settings forms are checked as text before any filesystem
call (on Windows, resolving a UNC path contacts the host and authenticates). ntpath and posixpath are passed explicitly,
so every Windows case runs on every CI OS."""
import ntpath
import os
from pathlib import Path
import posixpath

import pytest

from process.app_core.kernel.safe_paths import NETWORK, Outside, lexical, local_path, resolve_within, within

ROOTS, BASE = [r'C:\data\character_files', r'C:\data\persistent_memories\generated_assets'], r'C:\data'


@pytest.mark.parametrize('value', [
    r'\\attacker\share\a.png', '//attacker/share/a.png', r'/\attacker\share',  # UNC
    r'\\?\C:\data\character_files\a.png', r'\\.\pipe\x', r'\??\UNC\h\s\a.png',  # device paths
    'C:relative.png', r'C:\data\character_files\a.png:stream',  # drive-relative, alternate data stream
    r'character_files\CON.png', r'character_files\nul', r'character_files\com1.tar.gz', r'character_files\COM1 .png', r'character_files\nul .png',
    r'character_files\CON .mp4', 'character_files\\lpt\u00b9.png',  # reserved names, also with spaces before the extension
    'character_files\\a.png.', 'character_files\\a.png ', 'character_files\\a\x00.png', 'character_files\\a\n.png',
    r'..\outside.png', r'C:\data\character_files\..\secret.png', r'D:\elsewhere\a.png', '/character_files/a.png',  # outside
    'character_files\\a?.png', r'1:\x.png', r'CC:\x.png', '', 'x' * 4097])
def test_windows_values_refused_as_text(value):
    with pytest.raises(ValueError): within(value, ROOTS, BASE, flavour=ntpath)


@pytest.mark.parametrize('value', [r'character_files\a.png', 'character_files/sub/a.png', r'C:\data\character_files\a.png', r'character_files\console.png',
    r'character_files\com10.png', r'character_files\nul_x.png',
    r'c:\DATA\Character_Files\a.png', r'character_files\.\a.png', r'character_files\sub\..\a.png'])
def test_windows_values_inside_the_roots(value):
    assert ntpath.normcase(within(value, ROOTS, BASE, flavour=ntpath)).startswith(ntpath.normcase(ROOTS[0]))


def test_a_data_root_on_a_share_keeps_working_but_other_shares_never_match():
    roots, base = [r'\\nas\riko\character_files'], r'\\nas\riko'
    assert within(r'\\nas\riko\character_files\a.png', roots, base, flavour=ntpath) == r'\\nas\riko\character_files\a.png'
    assert within('character_files/a.png', roots, base, flavour=ntpath) == r'\\nas\riko\character_files\a.png'
    for value in (r'\\nas\other\character_files\a.png', r'\\nas\riko\secret.png', r'\\attacker\riko\character_files\a.png'):
        with pytest.raises(Outside): within(value, roots, base, flavour=ntpath)


def test_imports_refuse_network_paths_and_accept_any_local_drive():
    for value in (r'\\h\s\a.vrm', '//h/s/a.vrm', r'\\?\D:\a.vrm'):
        with pytest.raises(ValueError, match='Network and device paths'): lexical(value, BASE, flavour=ntpath)
    assert lexical(r'D:\Downloads\a.vrm', BASE, flavour=ntpath) == r'D:\Downloads\a.vrm'
    assert lexical(r'models\a.vrm', BASE, flavour=ntpath) == r'C:\data\models\a.vrm'
    with pytest.raises(ValueError, match='Network and device paths'): local_path('//attacker/share/a.vrm', '/tmp')


def test_posix_rules():
    roots = ['/data/character_files']
    for value in ('//h/s/a.png', '/data/character_files/../x.png', '../x.png'):
        with pytest.raises(ValueError): within(value, roots, '/data', flavour=posixpath)
    assert within('character_files/a:b.png', roots, '/data', flavour=posixpath) == '/data/character_files/a:b.png'
    assert lexical('a\\b?.png', '/data', flavour=posixpath) == '/data/a\\b?.png'  # legal POSIX names
    with pytest.raises(ValueError, match=NETWORK): lexical('//h/s/a.png', '/data', flavour=posixpath)


def test_a_symlink_inside_a_root_cannot_point_out(tmp_path):
    if os.name == 'nt': pytest.skip('symlinks need privileges on Windows')
    root = tmp_path / 'character_files'; root.mkdir()
    (tmp_path / 'secret.png').write_bytes(b'x'); (root / 'real.png').write_bytes(b'x')
    (root / 'link.png').symlink_to(tmp_path / 'secret.png')
    assert resolve_within('character_files/real.png', [root], tmp_path) == (root / 'real.png').resolve()
    with pytest.raises(Outside): resolve_within('character_files/link.png', [root], tmp_path)


def test_local_path_expands_home_and_accepts_path_objects(tmp_path):
    assert local_path(tmp_path / 'a.vrm', '/elsewhere') == Path(os.path.normpath(tmp_path / 'a.vrm'))
    assert local_path('~/a.vrm', tmp_path) == Path(os.path.normpath(os.path.expanduser('~/a.vrm')))


def test_resolve_media_refuses_a_network_path_before_any_filesystem_call(tmp_path, monkeypatch):
    from process.app_core.desktop.media import resolve_media
    touched = []
    def guard(original):
        def call(path, *args, **kwargs):
            if 'attacker' in str(path): touched.append(str(path)); raise AssertionError('a network path reached the filesystem')
            return original(path, *args, **kwargs)
        return call
    for name in ('resolve', 'stat', 'exists', 'is_file'): monkeypatch.setattr(Path, name, guard(getattr(Path, name)))
    for module, name in ((os, 'stat'), (os, 'lstat'), (os.path, 'realpath')): monkeypatch.setattr(module, name, guard(getattr(module, name)))
    for value in ('//attacker/share/x.png', r'\\attacker\share\x.png', '//attacker/share/character_files/x.png'):
        with pytest.raises(ValueError): resolve_media(tmp_path, value)
    assert not touched


def test_resolve_media_accepts_a_data_root_behind_a_symlink_by_either_spelling(tmp_path):
    if os.name == 'nt': pytest.skip('symlinks need privileges on Windows')
    from process.app_core.desktop.media import resolve_media
    real = tmp_path / 'real'; (real / 'character_files').mkdir(parents=True); (real / 'character_files' / 'a.png').write_bytes(b'x')
    link = tmp_path / 'link'; link.symlink_to(real, target_is_directory=True)
    expected = (real / 'character_files' / 'a.png').resolve()
    for value in ('character_files/a.png', str(link / 'character_files' / 'a.png'), str(expected)):  # relative, as given, as stored (resolved)
        assert resolve_media(link, value) == expected, value
    with pytest.raises(ValueError, match='outside approved'): resolve_media(link, str(tmp_path / 'elsewhere.png'))
    effects = tmp_path / 'clips'; effects.mkdir(); (effects / 'rain.mp4').write_bytes(b'x')
    (real / 'fx').symlink_to(effects, target_is_directory=True)  # desktop.effects_directory naming a link, as the effect tool stores it
    assert resolve_media(link, str((effects / 'rain.mp4').resolve()), 'fx') == (effects / 'rain.mp4').resolve()
    assert resolve_media(link, 'fx/rain.mp4', 'fx') == (effects / 'rain.mp4').resolve()
    with pytest.raises(ValueError, match='Unsupported media type'): resolve_media(link, 'character_files/a.txt')
