import numpy as np

from ravens.ccda_video import CCDAVideoRecorder


class FakeEnvironment:
    def render(self, config):
        shape = tuple(config['image_size']) + (3,)
        return np.full(shape, 127, dtype=np.uint8), None, None


def test_video_recorder_stride_and_close_schema(tmp_path, monkeypatch):
    class UnavailableWriter:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(
        'ravens.ccda_video.cv2.VideoWriter', lambda *args: UnavailableWriter())
    recorder = CCDAVideoRecorder(
        {'image_size': [12, 16]}, tmp_path / 'video.mp4', fps=30, stride=2)
    recorder.bind(FakeEnvironment())
    for _ in range(5):
        recorder.record('probe')
    result = recorder.close()
    assert result['record_calls'] == 5
    assert result['frames_written'] == 3
    assert result['fps'] == 30.
    assert result['stride'] == 2
    assert not result['video_available']
    assert result['error']
    assert len(list((tmp_path / 'frames').glob('*.png'))) == 3
