"""Two-rank beat gradient equivalence includes both explicit negative branches."""
import os
from pathlib import Path
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from improvnet.model.encoders.objective import beat_loss


def worker(rank,port,directory):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    try:
        torch.manual_seed(1)
        network=DDP(torch.nn.Linear(5,3,bias=False))
        inputs=torch.randn(4,4,5)
        ids=torch.tensor([1,1,2,3])
        times=torch.tensor([[0,100],[3000,3100],[0,100],[0,100]])
        sl=slice(rank*2,(rank+1)*2)
        projected=network(inputs[:,sl].reshape(-1,5)).reshape(4,2,3)
        loss,_=beat_loss(*projected,ids[sl],times[sl],ids[sl],times[sl]+10000,distributed=True)
        loss.backward()
        torch.save(network.module.weight.grad,Path(directory)/f'{rank}.pt')
        if rank==0:
            torch.manual_seed(1)
            single=torch.nn.Linear(5,3,bias=False)
            expected=single(inputs.reshape(-1,5)).reshape(4,4,3)
            full_loss,_=beat_loss(*expected,ids,times,ids,times+10000)
            full_loss.backward()
            torch.save(single.weight.grad,Path(directory)/'expected.pt')
    finally:dist.destroy_process_group()


def test_distributed_matches_full_global_gradient(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.start_processes(worker,args=(port,str(tmp_path)),nprocs=2,join=True,start_method='forkserver')
    expected=torch.load(tmp_path/'expected.pt',weights_only=True)
    for rank in range(2):
        torch.testing.assert_close(torch.load(tmp_path/f'{rank}.pt',weights_only=True),expected,rtol=2e-5,atol=2e-6)


def train_worker(rank,port,config):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    from improvnet.model.encoders.beat.train import train
    train(config)


def test_two_rank_full_beat_trainer_checkpoint_and_resume(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from improvnet.model.encoders.beat import config as beat_config
    from improvnet.metrics.continuity.continuity_train import settings_dict
    from improvnet.scripts.melody_manifest import build_manifest
    # Reuse existing synthetic MIDI fixtures; no production files/GPUs involved.
    monkeypatch.syspath_prepend(str(Path(__file__).parent/'melody'))
    from melody_test_helpers import corpus,mix
    config,_=corpus(tmp_path,ratio=1,raw_count=0)
    data=Path(config.PREPARED_DATA_FILE)
    index=data.with_suffix('.index.json')
    rows=json.loads(index.read_text())['rows']
    with data.open('ab') as handle:
        for number,genre in enumerate(('jazz','classical')):
            rows.append(dict(offset=handle.tell(),genre=genre,source='fixture',split='train',
                source_group=f'additional:{genre}',sha256=str(number+100),instruments=['Acoustic Piano','Acoustic Bass','drum']))
            handle.write((json.dumps(mix(20,genre))+'\n').encode())
    index.write_text(json.dumps(dict(rows=rows)))
    Path(config.DATA_FILE).unlink()
    build_manifest(config)
    settings=SimpleNamespace(**settings_dict(beat_config))
    settings.MANIFEST=str(config.DATA_FILE);settings.SAVE_DIR=str(tmp_path/'distributed-beat')
    settings.EMBED_DIM=16;settings.N_HEADS=2;settings.N_KV_HEADS=1;settings.N_LAYERS=1
    settings.BATCH_SIZE=2;settings.VALIDATION_BATCH_SIZE=2;settings.NUM_WORKERS=0
    settings.N_STEPS=2;settings.DECAY_STEPS=1;settings.WARMUP_STEPS=0;settings.SAMPLES_PER_EPOCH=8
    settings.TENSORBOARD=False;settings.DEVICE='cpu';settings.VALIDATE_EVERY=1;settings.SAVE_EVERY=1;settings.LOG_EVERY=1
    settings.DATASET_OPTIONS=dict(settings.DATASET_OPTIONS,piano_drum_draw_prob=0)
    def run():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        mp.start_processes(train_worker,args=(port,settings),nprocs=2,join=True,start_method='forkserver')
    run()
    path=Path(settings.SAVE_DIR)/'latest_checkpoint.pt'
    state=torch.load(path,weights_only=True)
    assert state['trained_steps']==2 and state['world_size']==2 and state['format']=='beat_v1'
    assert len(state['rng_states'])==2
    assert all(s['step']==2 for s in state['optimizer_state_dict']['state'].values())
    run()
    resumed=torch.load(path,weights_only=True)
    for name,value in state['model_state_dict'].items():
        torch.testing.assert_close(resumed['model_state_dict'][name],value,rtol=0,atol=0)
