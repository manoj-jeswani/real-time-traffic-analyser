import skvideo.io
import cv2
def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap


def fun():
    VIDEO_SOURCE = "input.mp4"

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=1000, detectShadows=True)
    # Set up image source
    # You can use also CV2, for some reason it not working for me
    cap = skvideo.io.vreader(VIDEO_SOURCE)

    # skipping 500 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=500)
    return bg_subtractor