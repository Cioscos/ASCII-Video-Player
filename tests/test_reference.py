import cv2


def test_tiny_video_fixture(tiny_video):
    cap = cv2.VideoCapture(tiny_video)
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 48
    cap.release()
