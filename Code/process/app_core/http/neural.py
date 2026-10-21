"""/api/neural: the emotion probe's status, training, samples and retained corpora (emotion/probe.py), when the provider
captures hidden states."""
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .backend import Services

router = APIRouter()


def current_probe(backend):
    host=backend.probe_host()
    probe=host.probe if host else None
    if probe is None: raise HTTPException(409,host.probe_error.rstrip('.')+'; fix the cause and restart Python' if host and host.probe_error
        else 'Enable the expression probe and a compatible in-process native library, then restart Python')
    return probe

@router.get('/api/neural/status')
def neural_status(backend: Services):
    host=backend.probe_host()
    probe=host.probe if host else None
    error=host.probe_error if host else ''  # the probe failed to start; chat runs without it
    return {'available':probe is not None, **(probe.status() if probe else {'mode':'unavailable','samples':0}), 'probe_error':error,
        'note':error.rstrip('.')+'. Chat works without it; fix the cause and restart Python.' if error else 'Requires hidden-state capture from the compatible in-process riko-native library.'}

@router.post('/api/neural/train')
def neural_train(backend: Services):
    try: return current_probe(backend).request_training()
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc

@router.get('/api/neural/data')
def neural_data(backend: Services, offset:int=0,limit:int=40,group:str|None=None):
    if offset<0 or not 1<=limit<=100: raise HTTPException(400,'Invalid page')
    return current_probe(backend).data_page(offset,limit,group)

@router.get('/api/neural/groups')
def neural_groups(backend: Services, offset:int=0,limit:int=20):
    if offset<0 or not 1<=limit<=100: raise HTTPException(400,'Invalid page')
    return current_probe(backend).data_groups(offset,limit)

class NeuralEdit(BaseModel):
    values:dict
    revision:str

@router.patch('/api/neural/data/{sample_id}')
def neural_edit(sample_id:str,request:NeuralEdit, backend: Services):
    try: return current_probe(backend).edit_sample(sample_id,request.values,request.revision)
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(409,str(exc)) from exc

@router.get('/api/neural/corpora')
def neural_corpora(backend: Services):
    from ..emotion.probe_storage import corpus_files
    results=[]
    for file in corpus_files(backend.config.paths):
        try:
            values=json.loads(file.read_text(encoding='utf-8'))
            results.append({'key':file.parent.name,'examples':len(values.get('examples',[])),
                'model':file.parents[1].name if file.parents[2].name=='expression' else file.parents[2].name if file.parent.parent.name=='expression probe' else 'Legacy dataset'})
        except (OSError,ValueError): continue
    return {'corpora':results}

@router.post('/api/neural/replay/{key}')
def neural_replay(key:str, backend: Services):
    if len(key)!=64 or any(c not in '0123456789abcdef' for c in key): raise HTTPException(400,'Invalid model key')
    probe=current_probe(backend)
    from ..emotion.probe_storage import corpus_files
    path=next((file for file in corpus_files(backend.config.paths) if file.parent.name==key),None)
    if path is None: raise HTTPException(404,'Retained examples not found')
    try: examples=json.loads(path.read_text(encoding='utf-8'))['examples']
    except (OSError,ValueError,KeyError,TypeError): raise HTTPException(404,'Retained examples not found')
    if not isinstance(examples,list) or not examples: raise HTTPException(400,'No retained text is available')
    try: probe.replay(examples,backend.probe_host().replay_lane)  # at background priority on the capture slot: a live turn preempts it
    except RuntimeError as exc: raise HTTPException(409,str(exc)) from exc
    return {'queued':True,'examples':len(examples),'note':'Old text is replayed; new activations and labels are collected. Old weights are not reused.'}
