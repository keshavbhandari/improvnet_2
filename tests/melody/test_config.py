"""Current standalone configs, corpus sampling and artifact paths."""
from pathlib import Path
import ast
import runpy
import tokenize
from types import SimpleNamespace
import pytest
from improvnet.model import melody_config as cfg
from improvnet.model.encoders.style import config as encoder_config
from improvnet.model.encoders.beat import config as beat_config
from improvnet.utils.melody_data import MelodyDataset
from melody_test_helpers import corpus


def test_configs_are_standalone_and_settings_are_explained():
    for config in (encoder_config, beat_config, cfg):
        path = Path(config.__file__)
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert 'config' not in (node.module or '')
                assert all('config' not in alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                assert all('config' not in alias.name for alias in node.names)
        with path.open() as stream:
            comment_lines = {token.start[0] for token in tokenize.generate_tokens(stream.readline)
                             if token.type == tokenize.COMMENT}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                assert all(line in comment_lines for line in range(node.lineno, node.end_lineno + 1))
    for config in (encoder_config, beat_config):
        values = runpy.run_path(config.__file__)
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom))
                       for node in ast.walk(ast.parse(Path(config.__file__).read_text())))
        assert values['N_STEPS'] == 30000
        assert 'CONTINUATION_START_STEP' not in values


def test_current_settings_use_the_single_config():
    assert cfg.CLASSICAL_RATIO==2
    assert cfg.SAVE_DIR.endswith('/melody_refinement_v15')
    assert cfg.DATA_FILE.name=='melody_manifest_v7.json'
    assert cfg.RUN_NAME=='melody_refinement_v15'
    assert cfg.PROTOCOL_VERSION==20 and not hasattr(cfg,'BEAT_TOKENS')
    assert cfg.MODEL_KWARGS['memory_tokens']==80
    assert cfg.REFERENCE_WINDOWS==2 and cfg.REFERENCE_PROB==1.
    assert cfg.REFINE_MIN_MASK_RATIO==.9 and cfg.REFINE_FULL_MASK_PROB==.25
    assert cfg.BATCH_SIZE==24 and cfg.ACCUM_STEPS==1
    assert 'ar_context_split_instrument' in cfg.INIT_CHECKPOINT
    assert '/style/' in cfg.STYLE_ENCODER
    assert not Path(cfg.__file__).with_name('melody_config_v7.py').exists()
    launcher=Path(__file__).resolve().parents[2]/'run_scripts/melody_launch.sh'
    assert '--config' not in launcher.read_text()


def test_encoder_cohort_stays_independent_of_finetuning_ratio():
    assert Path(encoder_config.MANIFEST).name=='melody_manifest_v5.json'
    assert str(cfg.DATA_FILE)!=encoder_config.MANIFEST


def test_lower_ratio_preserves_jazz_passes_and_held_out_files(tmp_path):
    config,_=corpus(tmp_path,ratio=2,raw_count=3)
    training=MelodyDataset(config.DATA_FILE,'train',config)
    validation=MelodyDataset(config.DATA_FILE,'validation',config)
    assert training.main_slots()==len(training.jazz_blues)*(2+config.JAZZ_BLUES_REPEATS)
    assert config.JAZZ_BLUES_REPEATS==2 and len(validation)==2


def test_training_rejects_legacy_memory_layout():
    from improvnet.utils.melody_chunks import validate_lengths
    config=SimpleNamespace(**{name:getattr(cfg,name) for name in dir(cfg) if name.isupper()})
    validate_lengths(config)
    config.MODEL_KWARGS=dict(cfg.MODEL_KWARGS,memory_tokens=64,drum_memory_tokens=0)
    with pytest.raises(ValueError,match='80 memory tokens'):
        validate_lengths(config)
