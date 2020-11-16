from sort import sort
import numpy as np
from glob import glob
from os import sep

class TrackingData():
    def __init__(self):
        pass
    
    @staticmethod
    def SORT(self, predictions):
        qtt_frames = len(predictions)
        tracker = sort.Sort()

        tracked_vehicles_trajectory = {} # Trajetória de cada ID identificado
        vehicles_on_the_frame = {} # Veículos que estão presentes no frame X

        # Ids e bounding boxes preditas pelo tracking
        mot_labels = [[0, 0, 0, 0, 0, 0, 0] for _ in range(qtt_frames + 1)]

        for frame_number in range(1, qtt_frames+1):

            bboxes_atual = predictions[frame_number][:]
            # Formatar a lista para alimentar o Sort
            # np.array( [ [x1,y1,x2,y2,score1], [x3,y3,x4,y4,score2], ... ] )

            if len(bboxes_atual) == 0:
                bboxes_atual = np.zeros((0, 5)) # Requerido pelo Algoritmo Sort
            else:
                for idx in range(len(bboxes_atual)):
                    x1, y1, w, h, classe = bboxes_atual[idx]
                    x2 = x1 + w
                    y2 = y1 + h
                    score = np.random.randint(50, 100)/100 # Temporariamente setar score como random.
                    bboxes_atual[idx] = [x1, y1, x2, y2, score, classe]
                
                # Numpy array requerido pelo Sort
                bboxes_atual = np.array(bboxes_atual)
            
                # Analisar o frame atual e identificar os bounding boxes id (update SORT)
                track_bbs_ids = tracker.update(bboxes_atual[:,:-1])
                this_frame_ids = track_bbs_ids[:,-1]

                # Passar as coordenadas para o padrão: [frame,x,y,w,h,idx]
                newboxes_list = [[0,0,0,0,0,0,0] for _ in range(len(track_bbs_ids))]
                
                for i, newbox in enumerate(track_bbs_ids):
                    x1,y1,x2,y2,idx = newbox
                    x, y, w, h = x1, y1, abs(x2-x1), abs(y2-y1)
                    x,y,w,h,idx = int(x), int(y), int(w), int(h), int(idx)
                    newboxes_list[i] = [frame_number, x, y, w, h, classe, idx]

                    # Guardar a trajetória do centro do veículo IDx
                    xc, yc = int(x + w/2) , int(y + h/2)
                    if idx in tracked_vehicles_trajectory:
                        tracked_vehicles_trajectory[idx].append((frame_number,xc,yc))
                    else:
                        tracked_vehicles_trajectory[idx] =  [(frame_number,xc,yc)]
                
                # Atualizar as variáveis
                vehicles_on_the_frame[frame_number] = this_frame_ids
                mot_labels[frame_number] = newboxes_list[:]

        return mot_labels, tracked_vehicles_trajectory, vehicles_on_the_frame


class Detections:
    def __init__(self, detections_file_path, video_folder_path):
        self.detections_file_path = detections_file_path
        self.detections_by_frame = {}
        self.video_folder_path = video_folder_path
    
    def parse_detections(self):
        """
        Arquivo contendo detecções:
            Arranjo de cada linha: "x,y,w,h,class"
            Comentários a serem ignorados: "#"
        
        Essa função retorna um dicionário com todas as detecções de cada frame.
        key = frame
        values = lista com as bounding boxes presentes no frame
                    [ [x1, y1, x2, y2, score], ... ]
        
        """
        qtt_of_frames = len(glob(self.video_folder_path+"*.jpg"))
        detections_by_frame = {key:[] for key in range(1, qtt_of_frames+1)}

        with open(self.detections_file_path) as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == "#": continue
                frame, x, y, w, h, classe = line.split(",")
                frame, classe = int(frame), int(classe)
                x, y, w, h = float(x), float(y), float(w), float(h)

                detections_by_frame[frame].append([x, y, w, h, classe])

        self.detections_by_frame = detections_by_frame
            
# Ler as detecções presentes no arquivo detections.txt
detections_parser = Detections(detections_file_path = f".{sep}detections.txt",
                        video_folder_path = f".{sep}video{sep}")
detections_parser.parse_detections()


# Fazer o Rastreamento dos Veículos
mot_labels, tracked_vehicles_trajectory, vehicles_id_by_frame = TrackingData.SORT(None, detections_parser.detections_by_frame)

# mot_labels (Bounding boxes preditas pelo SORT):
#    [[0,0,0,0,0,0,0], [frame_number, x, y, w, h, classe, idx], [frame_number2, x2, y2, w2, h2, classe2, idx2] ...]

# tracked_vehicles_trajectory (Dict com a trajetória de cada ID ao longo dos frames)

# vehicles_id_by_frame (Dict com todos os ids presentes em cada frame)
pass

#