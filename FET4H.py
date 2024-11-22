# -------------------------------------------------------------------------------#
# ------------------------ GUI FOR FACIAL EMOTION TRACKING ----------------------#
# -------------------------------------------------------------------------------#
import tkinter
from tkinter import filedialog, messagebox
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os, cv2, numpy as np
import PIL.Image, PIL.ImageTk
from datetime import datetime
import mediapipe as mp
from image_capture import _image_capture
import tensorflow as tf
from FET4H_model import *

class FET4H:
    def set_parameters(self):
        self.current_frame = None
        self.canvas_width = 1280*2//3
        self.canvas_height = 720*2//3
        self.saving_mark_time = 0
        self.labels = {0:'negative',1:'neutral',2:'positive'}
        self.LAB_count = {}
        self.EMO_confidence = None
        # Set global font sizes
        rcParams['font.size'] = 8  # General font size
        rcParams['axes.titlesize'] = 10  # Title size
        rcParams['axes.labelsize'] = 8  # Axis label size
        rcParams['xtick.labelsize'] = 7  # X-axis tick size
        rcParams['ytick.labelsize'] = 7  # Y-axis tick size
        rcParams['legend.fontsize'] = 7  # Legend font size

    def __init__(self):
        self.set_parameters()
        self.window = tkinter.Tk()
        self.window_title = 'Facial emotions tracking'
        self.window.title(self.window_title)
        self.btn_face_detection_var = tkinter.IntVar()
        self.btn_face_recognition_var = tkinter.IntVar()
        self.par_capture_delay = tkinter.IntVar(value=50)
        self.par_saving_delay = tkinter.IntVar(value=2000)
        self.par_storage_folder = os.getcwd() + '/saved'
        os.makedirs(self.par_storage_folder, exist_ok = True)
        self.par_image_source = 0  # 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'

        # create objects and initial them all
        self.image_capture = _image_capture(self.par_image_source)
        self.mp_face_detector = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.emotion_recognizer = tf.keras.models.load_model('FET4H_model')

        # Main frame for layout organization
        self.main_frame = tkinter.Frame(self.window)
        self.main_frame.pack(fill=tkinter.BOTH, expand=True)
        # Video canvas (top)
        self.video_frame = tkinter.Frame(self.main_frame, height=self.canvas_height)
        self.video_frame.pack_propagate(False)  # Prevent resizing to fit contents
        self.video_frame.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        # Create a canvas for video inside the video frame
        self.canvas = tkinter.Canvas(self.video_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        self.isShowCanvas = True

        # Middle frame for plots
        self.plot_frame = tkinter.Frame(self.main_frame)
        self.plot_frame.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        # Add pie plot
        self.pie_figure = Figure(figsize=(2, 2), dpi=100)
        self.pie_ax = self.pie_figure.add_subplot(111)
        self.pie_ax.set_title("Emotion Distribution")
        self.pie_canvas = FigureCanvasTkAgg(self.pie_figure, self.plot_frame)
        self.pie_canvas.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        self.update_pie_chart()
        # Add line plot
        self.line_figure = Figure(figsize=(5, 2), dpi=100)
        self.line_ax = self.line_figure.add_subplot(111)
        self.line_ax.set_title("Facial Emotion Confidence Over Time")
        self.line_ax.set_xlabel("Time")
        self.line_ax.set_ylabel("Confidence")
        self.line_ax.grid(True)
        self.line_canvas = FigureCanvasTkAgg(self.line_figure, self.plot_frame)
        self.line_canvas.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        self.update_line_chart()

        # Control buttons (bottom)
        self.controls_frame = tkinter.Frame(self.main_frame)
        self.controls_frame.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        self.add_controls()

    def update_pie_chart(self):
        self.pie_ax.clear()  # Clear the existing pie chart    
        data = self.LAB_count
        if len(data)==0:  # Check if data is empty
            self.pie_ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')
        else:
            labels = data.keys()
            sizes = data.values()
            self.pie_ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=[{'NEG':'red','NEU':'green','POS':'blue'}[lab] for lab in labels])
            self.pie_canvas.draw()

    def update_line_chart(self):
        if self.EMO_confidence is not None:
            self.line_ax.clear()
            emo_colors = {0:'red',1:'green',2:'blue'}
            tms = np.arange(len(self.EMO_confidence))
            for emo in self.labels.keys():
                self.line_ax.plot(tms, self.EMO_confidence[:,emo], label=self.labels[emo], color=emo_colors[emo], marker=".")
            self.line_canvas.draw()

    def add_controls(self):
        # Add some example controls
        f0 = tkinter.Frame(self.controls_frame)
        self.lbl_capture_delay = tkinter.Label(f0, text="Capture delay", anchor='e')
        self.lbl_capture_delay.pack(side=tkinter.LEFT, expand=True)
        self.btn_lbl_capture_delay = tkinter.Spinbox(f0, from_=50, to=1000, width=4, fg='blue',
                                                        textvariable=self.par_capture_delay)
        self.btn_lbl_capture_delay.pack(side=tkinter.LEFT, expand=True)
        self.lbl_saving_delay = tkinter.Label(f0, text="Saving delay", anchor='e')
        self.lbl_saving_delay.pack(side=tkinter.LEFT, expand=True)
        self.btn_saving_delay = tkinter.Spinbox(f0, from_=1000, to=50000, width=5, fg='blue',
                                                        textvariable=self.par_saving_delay)
        self.btn_saving_delay.pack(side=tkinter.LEFT, expand=True)
        f0.pack(side=tkinter.TOP)

        f1 = tkinter.Frame(self.controls_frame)
        self.btn_folder = tkinter.Button(f1, text='Set folder',
                                         command=self.cmd_set_folder)
        self.btn_folder.pack(side=tkinter.LEFT, expand=True)
        self.lbl_folder = tkinter.Label(f1, text=self.par_storage_folder, anchor='e', width=25, fg='blue')
        self.lbl_folder.pack(side=tkinter.LEFT, expand=True)
        self.btn_load_model = tkinter.Button(f1, text="Load model",
                                              command=self.cmd_load_model)
        self.btn_load_model.pack(side=tkinter.LEFT, expand=True)
        f1.pack(side=tkinter.TOP)

        f2 = tkinter.Frame(self.controls_frame)
        self.btn_detect_face = tkinter.Checkbutton(f2, text="Face detection & saving",
                                                   variable=self.btn_face_detection_var,
                                                   command=self.cmd_face_detection)
        self.btn_detect_face.pack(side=tkinter.LEFT, expand=True)
        self.btn_recognize = tkinter.Checkbutton(f2, text="Face recognition & reporting",
                                                 variable=self.btn_face_recognition_var,
                                                 command=self.cmd_emotion_recognition)
        self.btn_recognize.pack(side=tkinter.LEFT, expand=True)
        self.btn_clear = tkinter.Button(f2, text="Reset visualization",
                                                 command=self.cmd_reset_visualization)
        self.btn_clear.pack(side=tkinter.LEFT, expand=True)
        f2.pack(side=tkinter.TOP)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.update()
        #self.window.bind('<Button-1>', self.left_clicked)
        self.topmost()
        self.window.geometry(f'{self.canvas_width}x{self.canvas_height*2}')
        self.window.mainloop()

    def topmost(self, isOn=True):
        # self.window.wm_attributes("-topmost", isOn)
        return

    def cmd_set_folder(self):
        folder = filedialog.askdirectory(title='Choose saving folder',
                                         initialdir=self.par_storage_folder,
                                         mustexist=True, parent=self.window)
        if len(folder) > 0:
            self.par_storage_folder = folder
            self.lbl_folder.config(text=self.par_storage_folder)
            print(self.par_storage_folder)
        return None

    def cmd_face_detection(self):
        if self.btn_face_detection_var.get() == 1:
            if self.mp_face_detector is None:
                messagebox.showinfo('Notification','The FACE detector is NOT LOADED...!')
        return None

    def cmd_load_model(self):
        self.topmost(False)
        '''afile = filedialog.askopenfilename( title='Choose file of model (*.keras/h5)',
                                            initialdir=os.getcwd(),
                                            filetypes=(('keras','*.keras'),('h5','*.h5')), parent=self.window)
        self.topmost()'''
        afolder = filedialog.askdirectory(title='Choose folder to open MODEL',
                                         initialdir=self.par_storage_folder,
                                         mustexist=True, parent=self.window)
        if afolder:
            '''pModel = {  'input_shape': [70,70,3],
                        'num_classes': 3,
                        'downsamples': [(7,2),(2,2),(2,2),(2,2)],
                        'depths': [2,2,2,2],
                        'dims': [40,80,160,320],
                        'drop_path_rate': 0.25 }
            self.emotion_recognizer =  MyModel.create_FET4H(**pModel)
            self.emotion_recognizer.load_weights(afile)'''
            self.emotion_recognizer = tf.keras.models.load_model(afolder)
            messagebox.showinfo('Notification',f'Loaded the MODEL from {afolder}...!')
        return None

    def cmd_emotion_recognition(self):
        if self.btn_face_recognition_var.get() == 1:
            if self.emotion_recognizer is None:
                messagebox.showinfo('Notification','The MODEL for recognition is NOT LOADED...!')
        return None

    def cmd_reset_visualization(self):
        self.LAB_count = {}
        self.EMO_confidence = None
        self.update_pie_chart()
        self.update_line_chart()
        return None

    def update(self): # automatically run by period of time (self.delay)
        # Get a frame from the video source
        ret, self.current_frame = self.image_capture.get_frame()
        if ret:
            image = self.current_frame.copy()

            # face detection & saving
            if self.btn_face_detection_var.get() == 1 and self.mp_face_detector is not None:
                with self.mp_face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                    # To improve performance, optionally mark the image as not writeable to pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    detected_faces,list_emotions = [],[]
                    dt_now = datetime.now()
                    get_fecf = lambda fecf: (self.labels[fecf[0]][:3].upper(),fecf[1])
                    if results.detections:
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            fs = detection.score[0]
                            # Extract normalized coordinates
                            xmin = bbox.xmin*0.95
                            ymin = bbox.ymin*0.9
                            width = bbox.width*1.05
                            height = bbox.height*1.1
                            # Convert to absolute pixel values
                            img_height, img_width, _ = image.shape
                            x = int(xmin * img_width)
                            y = int(ymin * img_height)
                            w = int(width * img_width)
                            h = int(height * img_height)
                            detected_faces.append((x,y,w,h,fs))

                        if len(detected_faces)>0 and self.btn_face_recognition_var.get() == 1 and self.emotion_recognizer is not None:
                            faces = []
                            for x,y,w,h,fs in detected_faces:
                                af = cv2.resize(self.current_frame[y:y+h, x:x+w, :],(70,70))/127.5-1                                
                                faces.append(af)
                            faces = np.array(faces)
                            #print(faces.shape)
                            y_preds = self.emotion_recognizer.predict(faces)
                            i_preds = np.argmax(y_preds, axis=1)
                            list_emotions = [(j_lab,y_preds[i][j_lab]) for i,j_lab in enumerate(i_preds)] #emotion/conf
                            yp = np.array(y_preds)
                            self.EMO_confidence = yp if self.EMO_confidence is None else np.vstack((self.EMO_confidence,yp))

                        # Draw the face detection annotations on the image.
                        image.flags.writeable = True
                        for x,y,w,h,fs in detected_faces:
                            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),0)
                        for fecf in list_emotions:
                            fe,cf = get_fecf(fecf)
                            txt_show = f'{fe}({cf:.2f})'
                            cv2.putText(image, txt_show, (x + 6, y + 15), cv2.FONT_HERSHEY_DUPLEX, .7, (255, 255, 255), 2)
                            if fe in self.LAB_count.keys():  self.LAB_count[fe] += 1
                            else:  self.LAB_count[fe] = 1

                    if True: #FOR SAVING
                        if not os.path.exists(self.par_storage_folder):
                            print(f'Saving FOLDER "{self.par_storage_folder}" DOES NOT exists!')
                        else:
                            if self.saving_mark_time == 0:
                                self.saving_mark_time = dt_now
                            if (dt_now - self.saving_mark_time).total_seconds()*1000 >= self.par_saving_delay.get():
                                dirday = dt_now.strftime("%Y-%m-%d")
                                dirsave = self.par_storage_folder + '/' + dirday
                                if not os.path.exists(dirsave):
                                    os.makedirs(dirsave, exist_ok=True)
                                fe,cf = get_fecf(list_emotions[0]) if len(list_emotions)>0 else ('NO',0)
                                st_now = dt_now.strftime("%Y%b%d-%H%M%S-%f")
                                cv2.imwrite(f'{dirsave}/{fe}-{cf:.2f}-{st_now[:-3]}.jpg', self.current_frame)
                                with open(f'{dirsave}/results.txt','a') as af:
                                    fsw = st_now
                                    for i,(x,y,w,h,fs) in enumerate(detected_faces):
                                        fsw = f'{fsw};{x},{y},{w},{h},{fs:.2f}'
                                        if i<len(list_emotions):
                                            fe,cf = list_emotions[i]
                                            fe = self.labels[fe]
                                            fsw = f'{fsw},{fe},{cf:.2f}'
                                    af.write(f'{fsw}\n')
                                    af.close()
                                self.saving_mark_time = 0
        
            # show image to canvas of window
            if self.isShowCanvas:
                h, w = image.shape[:2]
                th, tw = self.canvas_height/h, self.canvas_width/w
                tm = min(th,tw)
                image = cv2.resize(image, (int(w*tm),int(h*tm)))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                               
                self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(image))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                self.update_pie_chart()
                self.update_line_chart()
            else:
                print('Captured:', ret, image.shape)

        self.window.after(self.par_capture_delay.get(), self.update)
        return None

    def left_clicked(self, event):
        return None

########################################################################################
if __name__ == "__main__":
    a = FET4H()
