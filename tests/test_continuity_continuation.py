"""Controlled continuation preserves the completed schedule and optimizer."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_continuity import tiny_config, make_manifest
from improvnet.metrics.continuity.continuity_train import lr_factor, train, validate_config, validate_resume_settings, settings_dict
from improvnet.model.encoders.style import config as production


def legacy_continuation_config():
    config = SimpleNamespace(**settings_dict(production))
    config.CONTINUATION_START_STEP = config.DECAY_STEPS
    config.CONTINUATION_END_LR = config.TAIL_END_LR
    del config.DECAY_STEPS, config.TAIL_END_LR
    return config


def test_production_tail_is_continuous_without_rewarming():
    config=SimpleNamespace(**settings_dict(production))
    original=deepcopy(config)
    del original.DECAY_STEPS, original.TAIL_END_LR
    original.N_STEPS=20000
    validate_config(config)
    for step in (0,100,499,500,10000,19999):
        assert lr_factor(step,config)==lr_factor(step,original)
    assert config.LR*lr_factor(20000,config)==pytest.approx(1e-5)
    assert config.LR*lr_factor(25000,config)==pytest.approx(7.5e-6)
    assert config.LR*lr_factor(30000,config)==pytest.approx(5e-6)
    assert all(lr_factor(i+1,config)<=lr_factor(i,config) for i in range(20000,30000))
    legacy = legacy_continuation_config()
    for step in (0, 499, 500, 10000, 19999, 20000, 25000, 30000):
        assert lr_factor(step, config) == lr_factor(step, legacy)
    validate_resume_settings(settings_dict(legacy), config, 30000)
    assert not hasattr(config, 'CONTINUATION_START_STEP')


def test_continuation_guards_reject_partial_runs_and_unrelated_changes():
    config=legacy_continuation_config()
    saved=settings_dict(config)
    saved.pop('CONTINUATION_START_STEP');saved.pop('CONTINUATION_END_LR');saved['N_STEPS']=20000
    with pytest.raises(ValueError,match='completed'):
        validate_resume_settings(saved,config,19500)
    bad=deepcopy(config);bad.BATCH_SIZE+=1
    with pytest.raises(ValueError,match='BATCH_SIZE'):
        validate_resume_settings(saved,bad,20000)
    bad=deepcopy(config);bad.LR*=2
    with pytest.raises(ValueError,match='LR'):
        validate_resume_settings(saved,bad,20000)
    resumed=settings_dict(config)
    bad=deepcopy(config);bad.CONTINUATION_END_LR=1e-6
    with pytest.raises(ValueError,match='CONTINUATION_END_LR'):
        validate_resume_settings(resumed,bad,23000)
    validate_resume_settings(resumed,config,23000)


def test_cpu_train_extend_and_resume_retains_optimizer_and_best(tmp_path):
    make_manifest(tmp_path)
    config=tiny_config(tmp_path)
    train(config)
    path=Path(config.SAVE_DIR)/'latest_checkpoint.pt'
    saved=torch.load(path,weights_only=True)
    old_best=saved['best_validation_loss']
    best_path=Path(config.SAVE_DIR)/'best_checkpoint.pt'
    original_best=best_path.read_bytes()
    config.N_STEPS=4
    config.CONTINUATION_START_STEP=2
    config.CONTINUATION_END_LR=config.MIN_LR/2
    train(config)
    extended=torch.load(path,weights_only=True)
    assert extended['trained_steps']==4
    assert extended['best_validation_loss']<=old_best
    assert extended['scheduler_state_dict']['last_epoch']==4
    assert extended['optimizer_state_dict']['param_groups'][0]['lr']==pytest.approx(config.MIN_LR/2)
    assert all(state['step']==4 for state in extended['optimizer_state_dict']['state'].values())
    if extended['best_validation_loss']==old_best:
        assert best_path.read_bytes()==original_best
    # A completed continuation must resume without configuration migration again.
    train(config)
    resumed=torch.load(path,weights_only=True)
    assert resumed['trained_steps']==4
    for key in extended['model_state_dict']:
        torch.testing.assert_close(extended['model_state_dict'][key],resumed['model_state_dict'][key],rtol=0,atol=0)


def test_continuation_cannot_start_from_scratch(tmp_path):
    config=tiny_config(tmp_path)
    config.N_STEPS=4;config.CONTINUATION_START_STEP=2;config.CONTINUATION_END_LR=config.MIN_LR/2
    with pytest.raises(FileNotFoundError,match='existing latest_checkpoint'):
        train(config)


def test_clean_schedule_can_train_from_scratch_and_resume(tmp_path):
    from improvnet.model.encoders.style.train import train as style_train
    make_manifest(tmp_path)
    config = tiny_config(tmp_path)
    config.N_STEPS = 4
    config.DECAY_STEPS = 2
    config.TAIL_END_LR = config.MIN_LR / 2
    style_train(config)
    path = Path(config.SAVE_DIR) / 'latest_checkpoint.pt'
    saved = torch.load(path, weights_only=True)
    assert saved['trained_steps'] == 4
    assert saved['optimizer_state_dict']['param_groups'][0]['lr'] == pytest.approx(config.TAIL_END_LR)
    style_train(config)
    resumed = torch.load(path, weights_only=True)
    assert resumed['trained_steps'] == 4
    for key in saved['model_state_dict']:
        torch.testing.assert_close(saved['model_state_dict'][key], resumed['model_state_dict'][key], rtol=0, atol=0)


def test_legacy_schedule_migration_keeps_strict_guards():
    config = SimpleNamespace(**settings_dict(production))
    saved = settings_dict(legacy_continuation_config())
    changed = deepcopy(config)
    changed.TAIL_END_LR /= 2
    with pytest.raises(ValueError, match='TAIL_END_LR'):
        validate_resume_settings(saved, changed, 25000)
    changed = deepcopy(config)
    changed.BATCH_SIZE += 1
    with pytest.raises(ValueError, match='BATCH_SIZE'):
        validate_resume_settings(saved, changed, 25000)
