"""Recover ignored MIDI metadata without changing notes, controllers or timing."""
from collections import Counter
from pathlib import Path

import mido
from mido.midifiles import midifiles as smf
from mido.midifiles.meta import KeySignatureError, meta_charset, build_meta_message

from improvnet.tokenizer.midi import MidiDict, midi_to_dict


class RecoverableMidiFile(mido.MidiFile):
    """Strict musical events; tolerate invalid key metadata and SysEx payloads."""
    def __init__(self, *args, **kwargs):
        self.recovery=Counter()
        super().__init__(*args, **kwargs)

    def _load(self, infile):
        with meta_charset(self.charset):
            self.type, count, self.ticks_per_beat=smf.read_file_header(infile)
            for _ in range(count):
                name,size=smf.read_chunk_header(infile)
                if name!=b'MTrk':
                    raise OSError('no MTrk header at start of track')
                end=infile.tell()+size
                track=mido.MidiTrack()
                last_status=None
                while infile.tell()<end:
                    delta=smf.read_variable_int(infile)
                    status=smf.read_byte(infile)
                    if status<0x80:
                        if last_status is None:
                            raise OSError('running status without last_status')
                        peek,status=[status],last_status
                    else:
                        peek=[]
                        if status!=0xff:
                            last_status=status
                    if status==0xff:
                        meta_type=smf.read_byte(infile)
                        data=smf.read_bytes(infile,smf.read_variable_int(infile))
                        try:
                            msg=build_meta_message(meta_type,data,delta)
                        except KeySignatureError:
                            self.recovery['invalid_key_signature']+=1
                            msg=mido.MetaMessage('sequencer_specific',time=delta)
                        except IndexError:
                            expected={0x58:4,0x59:2}.get(meta_type)
                            if expected is None or len(data)>=expected:
                                raise
                            name='truncated_time_signature' if meta_type==0x58 else 'truncated_key_signature'
                            self.recovery[name]+=1
                            # Keep the delta time. Broken descriptive metadata
                            # must not alter musical events or tempo messages.
                            msg=mido.MetaMessage('sequencer_specific',time=delta)
                    elif status in (0xf0,0xf7):
                        data=smf.read_bytes(infile,smf.read_variable_int(infile))
                        if data and data[0]==0xf0:
                            data=data[1:]
                        if data and data[-1]==0xf7:
                            data=data[:-1]
                        if any(byte>127 for byte in data):
                            self.recovery['invalid_sysex']+=1
                            msg=mido.MetaMessage('sequencer_specific',time=delta)
                        else:
                            msg=mido.Message('sysex',data=data,time=delta)
                    else:
                        # Never clip an invalid pitch, velocity, controller or
                        # delta time to manufacture a plausible musical event.
                        msg=smf.read_message(infile,status,peek,delta,clip=False)
                    track.append(msg)
                if infile.tell()!=end:
                    raise OSError('MIDI event exceeds its track boundary')
                self.tracks.append(track)


def read_midi_dict(path, recovery=None):
    try:
        midi=mido.MidiFile(path)
    except (KeySignatureError,ValueError,OSError,IndexError) as original:
        try:
            midi=RecoverableMidiFile(path)
        except Exception:
            raise original
        if not midi.recovery:
            raise original
        if recovery is not None:
            recovery.update(midi.recovery)
    result=MidiDict(**midi_to_dict(midi))
    result.metadata['abs_load_path']=str(Path(path).absolute())
    return result
