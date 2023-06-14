import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import torch
import math
import torch.nn as nn
import itertools
import dlib
from imutils import face_utils
from transformers import ViTConfig, ViTModel

# Optim
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
TRANSFORMER_LEARNING_RATE = 5e-5
TRANSFORMER_WEIGHT_DECAY = 0.0
EPOCHS = 100

DROPOUT = 0.3
BATCH_SIZE = 32

FACE_KP_PATH = "./shape_predictor_68_face_landmarks.dat"
DETECTOR_KP = dlib.get_frontal_face_detector()
PREDICTOR_KP = dlib.shape_predictor(FACE_KP_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
ID_TO_LABEL = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}
LABEL_TO_ID = {"Neutral": 0, "Happy": 1, "Sad": 2, "Surprise": 3, "Fear": 4, "Disgust": 5, "Anger": 6, "Contempt": 7}
MU_TYPE = ["eye_mouth_center", "bounding_box_center", "mean"]

class CustomViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.mu = mu
        self.patch_embeddings = None

    def gauss(self, p, x_y, mu, sigma):
        # return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(- (p - mu[x_y])**2 / (2 * sigma**2))
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.e**(- (p - mu[:, x_y])**2 / (2 * sigma**2))

    def forward(self, pixel_values, bool_masked_pos = None, interpolate_pos_encoding=False):
        # SIGMA = 10
        # kp = pixel_values[:, :-2]
        # mu = pixel_values[:, -2:]
        # positions_len = kp.shape[1]
        # embedding_dim = 197#positions_len//2
        # d_model = embedding_dim//2
        # batch_size = pixel_values.shape[0]

        # PE = torch.Tensor(np.zeros((batch_size, embedding_dim, positions_len))).to(DEVICE)
        # for i in range(positions_len//2):
        #     for j in range(embedding_dim//2):
        #         # posizioni pari
        #         PE[:, 2*j, 2*i] = np.sin((2*i) / 10000**((2*j)/d_model))*self.gauss(kp[:, 2*i], 0, mu, SIGMA)
        #         PE[:, (2*j) + 1, 2*i] = np.cos((2*i) / 10000**((2*j)/d_model))*self.gauss(kp[:, 2*i], 0, mu, SIGMA)
        #         # posizioni dispari
        #         PE[:, 2*j, (2*i) + 1] = np.sin(((2*i) + 1) / 10000**((2*j)/d_model))*self.gauss(kp[:, (2*i) + 1], 1, mu, SIGMA)
        #         PE[:, (2*j) + 1, (2*i) + 1] = np.cos(((2*i) + 1) / 10000**((2*j)/d_model))*self.gauss(kp[:, (2*i) + 1], 1, mu, SIGMA)
        return pixel_values

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initializing a ViT vit-base-patch16-224 style configuration
        configuration = ViTConfig()

        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        configuration.image_size = 224
        configuration.hidden_size = 136
        configuration.num_hidden_layers = 8
        configuration.num_attention_heads = 8
        self.transformer = ViTModel(configuration)

        # pretrained_model = ViTModel.from_pretrained('openai/clip-vit-base-patch16')
        # model.load_state_dict(pretrained_model.state_dict())
        
        self.embedding = CustomViTEmbeddings(configuration)
        self.embedding.patch_embeddings = self.transformer.embeddings.patch_embeddings

        self.transformer.embeddings = self.embedding
        self.classifier = nn.Linear(configuration.hidden_size, len(LABEL))
        # self.dropout = torch.nn.Dropout(DROPOUT)
    
    def forward(self, x):
        # label = input["label"].to(DEVICE)
        po = self.transformer(x)["pooler_output"]
        # o = self.dropout(po)
        o = self.classifier(po)
        # loss = self.compute_loss(o, label)
        return o

    # def compute_loss(
    #     self, 
    #     preds: torch.Tensor, 
    #     labels: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Compute the loss of the model.
    #     Args:
    #         logits (`torch.Tensor`):
    #             The logits of the model.
    #         labels (`torch.Tensor`):
    #             The labels of the model.
    #     Returns:
    #         obj:`torch.Tensor`: The loss of the model.
    #     """
    #     return F.cross_entropy(
    #         preds,
    #         labels
    #     )

def keypoints(image, im_show = False):
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    # Get faces in the image
    rects = DETECTOR_KP(gray, 0)
    # Assume that all image are with one face
    if len(rects) == 0:
        return None
    rect = rects[0]
    '''for (i, rect) in enumerate(rects):  if there are more than one face'''
    # Make the prediction and transfom it to numpy array
    shape = PREDICTOR_KP(gray, rect)
    shape = face_utils.shape_to_np(shape)
    shape = np.array([(x, image.shape[1] - y) for (x,y) in shape])
    keypoints = {"contour": shape[0:17],
                 "left_eyebrow": shape[17:22],
                 "right_eyebrow": shape[22:27],
                 "nose": shape[27:36],
                 "left_eye": shape[36:42],
                 "right_eye": shape[42:48],
                 "mouth": shape[48:],
                 "all": shape}
    # if im_show:
    #     colors = list(itertools.product([255, 0], repeat = 3))[-7:]
    #     # Draw on our image, all the finded cordinate points (x,y) 
    #     for i, (key, values) in enumerate(keypoints.items()):
    #         if key == "all":
    #             continue
    #         for x, y in values:
    #             cv2.circle(image, (x, image.shape[1] - y), 7, colors[i], -1)
    #     # Show the image
    #     cv2_imshow(image)
    # else:
    return keypoints

def get_bounding_box(keypoints):
    bounding_box = {}
    bounding_box["right"] = keypoints[keypoints.argmax(0)[0]]
    bounding_box["top"] = keypoints[keypoints.argmax(0)[1]]
    bounding_box["left"] = keypoints[keypoints.argmin(0)[0]]
    bounding_box["bot"] = keypoints[keypoints.argmin(0)[1]]
    return bounding_box

# def plot_bounding_box(ax, bounding_box, line_size, label = None):
#     ax.plot([
#                 bounding_box["left"][0],
#                 bounding_box["left"][0],
#                 bounding_box["right"][0],
#                 bounding_box["right"][0],
#                 bounding_box["left"][0]
#             ],
#             [  
#                 bounding_box["bot"][1],
#                 bounding_box["top"][1],
#                 bounding_box["top"][1],
#                 bounding_box["bot"][1],
#                 bounding_box["bot"][1]
#             ], color = "red", linewidth = line_size, label = label, zorder = 0)
    
# def plot_first_approach(ax, dot_size, legend_size):
#     for i, kp in enumerate(all_kp):
#         ax.scatter(kp[0], kp[1], c = "Black", s = dot_size, label = "Facial Landmark Keypoints" if i == 0 else None)
#     ax.scatter(mu["mean"][0], mu["mean"][1], c = "Green", s = dot_size, label = "µ as mean")
#     ax.legend(loc=0, prop={'size': legend_size})

# def plot_second_approach(ax, dot_size, legend_size, line_size):
#     for i, kp in enumerate(all_kp):
#         ax.scatter(kp[0], kp[1], c = "Black", s = dot_size, label = "Facial Landmark Keypoints" if i == 0 else None)
#     plot_bounding_box(ax, bounding_box, line_size, label = "Face bounding box")
#     ax.plot([
#                 bounding_box["left"][0],
#                 bounding_box["right"][0]
#             ], 
#             [
#                 mu["bounding_box_center"][1],
#                 mu["bounding_box_center"][1]
#             ], color = "red", linestyle = "dashed", zorder = 0)
#     ax.plot([
#                 mu["bounding_box_center"][0],
#                 mu["bounding_box_center"][0]
#             ], 
#             [
#                 bounding_box["bot"][1],
#                 bounding_box["top"][1]
#             ], color = "red", linestyle = "dashed", zorder = 0)
#     ax.scatter(mu["bounding_box_center"][0], mu["bounding_box_center"][1], c = "Green", s = dot_size, label = "µ as center of the face bounding box")
#     ax.legend(loc=0, prop={'size': legend_size})
    
# def plot_third_approach(ax, dot_size, legend_size, line_size):
#     for i, kp in enumerate(all_kp):
#         ax.scatter(kp[0], kp[1], c = "Black", s = dot_size, label = "Facial Landmark Keypoints" if i == 0 else None)
#     plot_bounding_box(ax, left_eye_bounding_box, line_size, label = "Eyes and mouth bounding boxes")
#     plot_bounding_box(ax, right_eye_bounding_box, line_size)
#     plot_bounding_box(ax, mouth_bounding_box, line_size)

#     ax.plot([
#                 right_eye_bounding_box["left"][0],
#                 mu["eye_mouth_center"][0]
#             ], 
#             [
#                 right_eye_bounding_box["bot"][1],
#                 mu["eye_mouth_center"][1]
#             ], color = "red", linestyle = "dashed", zorder = 0)
#     ax.plot([
#                 left_eye_bounding_box["right"][0],
#                 mu["eye_mouth_center"][0]
#             ], 
#             [
#                 left_eye_bounding_box["bot"][1],
#                 mu["eye_mouth_center"][1]
#             ], color = "red", linestyle = "dashed", zorder = 0)
#     ax.plot([
#                 (mouth_kp.sum(0)/len(mouth_kp))[0],
#                 mu["eye_mouth_center"][0]
#             ], 
#             [
#                 mouth_bounding_box["top"][1],
#                 mu["eye_mouth_center"][1]
#             ], color = "red", linestyle = "dashed", zorder = 0)
#     ax.scatter(mu["eye_mouth_center"][0], mu["eye_mouth_center"][1], c = "Green", s = dot_size, label = "µ as center of the eyes and mouth bounding boxes")
#     ax.legend(loc=0, prop={'size': legend_size})

def get_kp_info(image):
    # Get image
    # image = cv2.imread(path)
    # Get the image size
    height, width, _ = image.shape
    # Calculate the desired size
    desired_size = 224
    ratio = min(desired_size/height, desired_size/width)
    resized_height = int(height * ratio)
    resized_width = int(width * ratio)
    # Resize the image
    image = cv2.resize(image, (resized_width, resized_height))
    # Find facial landmark keypoints
    facial_kp = keypoints(image, im_show = False)
    
    if facial_kp == None:
        return None

    all_kp = facial_kp["all"]
    mu = {"mean": [0, 0],
        "bounding_box_center": [0, 0],
        "eye_mouth_center": [0, 0]}

    # First approach
    mu["mean"] = all_kp.sum(0)/len(all_kp)

    # Second approach
    bounding_box = get_bounding_box(all_kp)
    mu["bounding_box_center"][0] = (bounding_box["left"][0] + bounding_box["right"][0])/2
    mu["bounding_box_center"][1] = (bounding_box["top"][1] + bounding_box["bot"][1])/2

    # Tird approach
    left_eye_kp = facial_kp["left_eye"]
    left_eye_bounding_box = get_bounding_box(left_eye_kp)
    right_eye_kp = facial_kp["right_eye"]
    right_eye_bounding_box = get_bounding_box(right_eye_kp)
    mouth_kp = facial_kp["mouth"]
    mouth_bounding_box = get_bounding_box(mouth_kp)
    mu["eye_mouth_center"][0] = ((mouth_kp.sum(0)/len(mouth_kp))[0] 
                                    + left_eye_bounding_box["right"][0]
                                    + right_eye_bounding_box["left"][0])/3
    mu["eye_mouth_center"][1] = (mouth_bounding_box["top"][1] 
                                    + (left_eye_bounding_box["bot"][1] + right_eye_bounding_box["bot"][1])/2)/2
    out = {"all_kp": all_kp,
           "mu": mu,
           "image": image,
           "bounding_box": bounding_box,
           "left_eye_kp": left_eye_kp,
           "left_eye_bounding_box": left_eye_bounding_box,
           "right_eye_kp": right_eye_kp,
           "right_eye_bounding_box": right_eye_bounding_box,
           "mouth_kp": mouth_kp,
           "mouth_bounding_box": mouth_bounding_box
           }
    return out

def PE(pixel_values):
    SIGMA = 10
    kp = pixel_values[:-2]
    mu = pixel_values[-2:]
    positions_len = kp.shape[0]
    embedding_dim = 197#positions_len//2
    d_model = embedding_dim//2

    PE = torch.Tensor(np.zeros((embedding_dim, positions_len))).to(DEVICE)
    for i in range(positions_len//2):
        for j in range(embedding_dim//2):
            # posizioni pari
            PE[2*j, 2*i] = np.sin((2*i) / 10000**((2*j)/d_model))*gauss(kp[2*i], 0, mu, SIGMA)
            PE[(2*j) + 1, 2*i] = np.cos((2*i) / 10000**((2*j)/d_model))*gauss(kp[2*i], 0, mu, SIGMA)
            # posizioni dispari
            PE[2*j, (2*i) + 1] = np.sin(((2*i) + 1) / 10000**((2*j)/d_model))*gauss(kp[(2*i) + 1], 1, mu, SIGMA)
            PE[(2*j) + 1, (2*i) + 1] = np.cos(((2*i) + 1) / 10000**((2*j)/d_model))*gauss(kp[(2*i) + 1], 1, mu, SIGMA)
    return PE

def gauss(p, x_y, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(- (p - mu[x_y])**2 / (2 * sigma**2))

def calculate_average_rgb(frame):
    # Calcola il valore medio RGB del frame
    info = get_kp_info(frame)
    output = "no face detected"
    if info is not None:            
        all_kp = info["all_kp"]
        mu = info["mu"]
        input = all_kp.flatten().tolist() + mu[MU_TYPE[0]]
        input = torch.Tensor(np.array(input)).to(DEVICE)
        input = PE(input).unsqueeze(0)
        output = model(input).detach().numpy()
        output = np.argmax(output)
        output = ID_TO_LABEL[output]
    return output

def update_text():
    if frame[0] is not None:
        # Calcola il valore medio RGB del frame corrente
        text = calculate_average_rgb(frame[0])
        # Aggiorna la scritta con il valore medio RGB
        # text = f"RGB: {r}, {g}, {b}"
        text_label.config(text=text)
    root.after(1000, update_text)


def show_frame():
    ret, f = cap.read()
    if ret:
        # Ridimensiona il frame per adattarlo alla finestra
        f = cv2.resize(f, (640, 480))
        # Converti l'immagine in formato corretto per Tkinter
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        # Aggiorna l'immagine visualizzata nella finestra
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        # Salva il frame corrente per calcolare il valore medio RGB
        frame[0] = f.copy()
    # Richiama la funzione show_frame dopo 1 millisecondo
    video_label.after(1, show_frame)

frame = [None]
checkpoint_path =  './ckp.pt'
model = MyModel()
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(DEVICE)
model.eval()

# Inizializza la finestra Tkinter
root = tk.Tk()
root.title("Webcam")

# Inizializza la cattura video dalla webcam
cap = cv2.VideoCapture(0)

# Crea una label per visualizzare il video
video_label = tk.Label(root)
video_label.pack()

# Crea una label per il valore medio RGB
text_label = tk.Label(root, font=("Arial", 20))
text_label.place(x=10, y=10)

# Aggiorna il valore medio RGB ogni secondo
update_text()

# Avvia la visualizzazione del video
show_frame()

# Avvia la finestra Tkinter
root.mainloop()

# Rilascia la cattura video e chiudi la finestra
cap.release()
cv2.destroyAllWindows()

