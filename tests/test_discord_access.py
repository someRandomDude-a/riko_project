from uuid import uuid4
import pytest
from process.app_core.integrations.discord.access import DiscordAccess
from process.app_core.integrations.discord.launcher import DiscordLauncher
from process.app_core.desktop.state import DesktopState
from process.app_core.persistence.conversation_store import ConversationStore
from process.app_core.events.bus import RuntimeEvent


def test_access_revision_and_live_gates(tmp_path, monkeypatch):
    monkeypatch.setattr('dotenv.dotenv_values', lambda path: {'Discord_admins':'1'})
    service = DiscordAccess(tmp_path)
    original = service.read()
    values = {'admins':['1'], 'users':['2'], 'channels':['3'], 'allow_dms':False, 'admin_actions':False}
    saved = service.save(values, original['revision'])
    assert saved['revision'] != original['revision']
    assert service.settings().allows(2,3,guild=True)
    assert not service.settings().allows(2,4,guild=True)
    assert not service.settings().allows(1,3,guild=True,admin=True)
    assert not service.settings().allows(2,3)
    with pytest.raises(RuntimeError): service.save(values, original['revision'])
    with pytest.raises(ValueError): service.save({**values,'users':[2]},saved['revision'])


def test_reports_ignore_blocked_content_and_track_real_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr('dotenv.dotenv_values',lambda path:{'Discord_admins':'1'})
    state = DesktopState()
    service = DiscordLauncher(tmp_path,state)
    client = str(uuid4())
    service.attach(client)
    assert not service.status()['ready']
    with pytest.raises(ValueError): service.attach(str(uuid4()))
    service.report(client,{'kind':'ready','bot_name':'Test bot'})
    assert state.snapshot()['discord']['ready']
    message = {'kind':'message','message_id':'10','user_id':'2','channel_id':'3','guild_id':None,'text':'blocked'}
    service.report(client,message)
    assert not state.incoming
    service.report(client,{**message,'message_id':'11','user_id':'1','text':'hello'})
    assert state.incoming[0]['source']=='discord'
    service.report(client,{**message,'message_id':'12','user_id':'1','author_bot':True})
    assert len(state.incoming)==1
    assert len(service.inbox_snapshot()['messages'])==3
    service.detach(client)
    assert not service.status()['running']


def test_discord_history_uses_separate_session(tmp_path):
    store = ConversationStore(tmp_path/'history.sqlite3')
    try:
        sid = 'discord:test:dm:3'
        store.observe(RuntimeEvent('chat.input',{'text':'hello','source':'discord','conversation_id':sid},turn_id='turn'))
        store.observe(RuntimeEvent('chat.completed',{'text':'hi','source':'discord','conversation_id':sid},turn_id='turn'))
        page = store.page()
        assert all(m['session_id']==sid and m['source']=='discord' for m in page['messages'])
        assert page['sessions'][sid]['source']=='discord'
    finally: store.close()
