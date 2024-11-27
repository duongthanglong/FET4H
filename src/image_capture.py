import cv2, time
import threading

_fixed_img = None
class _image_capture(threading.Thread):
    queue_max_size = 500
    queue = []

    def __init__(self, source=0):
        threading.Thread.__init__(self)
        # Open the source
        self.source = source
        self.vid = cv2.VideoCapture(source)
        if not self.vid.isOpened():
            raise ValueError("Can NOT OPEN the source", source)

        # Get source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('Opened video source, width x heigh = (', self.width, self.height, '), frame per second =',self.vid.get(cv2.CAP_PROP_FPS))
        self.is_running = False

    def get_frame(self, mili_second_to_read = None):
        if self.source==_fixed_img:  return (True,_fixed_img)
        elif self.vid.isOpened():
            if mili_second_to_read is not None:
                self.vid.set(cv2.CAP_PROP_POS_MSEC, mili_second_to_read)
            ret, frame = self.vid.read()
            if self.source==0:
                frame = cv2.flip(frame,1)
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)
            else:
                return (ret, None)
        else:
            return (False, None)

    def get_frame_from_queue(self, get_last):
        k = len(self.queue)
        if k>0:
            return self.queue.pop(0) if get_last==True else self.queue.pop(k-1)
        return None

    def run(self):
        while self.is_running:
            rt,fr = self.get_frame()
            if fr is not None:
                if len(self.queue) > self.queue_max_size:
                    self.queue.pop(0)
                self.queue.append(fr)
            time.sleep(0.1)

    def start(self):
        self.setDaemon(True)
        self.is_running = True
        threading.Thread.start(self)

    def stop(self):
        self.is_running = False

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
########################################################################################
if __name__ == "__main__":
    import os
    from datetime import datetime
    import time
    a = _image_capture() 
    pt = os.getcwd() + '/images'
    count = 1
    while True:
        rt, fr = a.get_frame()
        hour = datetime.now().strftime("%Y-%m-%d-%H")
        if rt: #create subfolders
            if True:
                pt2 = pt + '/' + hour
                if not os.path.exists(pt2):
                    os.mkdir(pt2)
            else:
                pt2 = pt
            fn1 = pt2 + '/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3] + '.jpg'
            cv2.imwrite(fn1, fr)
            print(count,fr.shape, fn1)
            count += 1
        else:
            print('No captured!')
        time.sleep(0.5)
    del(a)
