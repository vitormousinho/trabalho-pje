#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface para o modelo YOLO usado na detecção de veículos.
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class YOLOModel:
    """Classe para gerenciar o modelo YOLO para detecção de veículos."""
    
    def __init__(self, model_path, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Inicializa o modelo YOLO.
        
        Args:
            model_path: Caminho para os arquivos do modelo YOLO
            confidence_threshold: Limiar de confiança para detecções
            nms_threshold: Limiar para non-maximum suppression
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Caminhos para arquivos do modelo
        weights_path = os.path.join(model_path, "yolov4.weights")
        config_path = os.path.join(model_path, "yolov4.cfg")
        
        # Verificar se os arquivos existem
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            logger.error(f"Arquivos do modelo YOLO não encontrados em {model_path}")
            raise FileNotFoundError(f"Arquivos do modelo YOLO não encontrados")
        
        # Carregar o modelo YOLO
        try:
            logger.info("Carregando modelo YOLO...")
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            
            # Definir preferência para backend e target
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Obter nomes das camadas de saída
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Carregar nomes das classes
            classes_path = os.path.join(model_path, "coco.names")
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
                
            logger.info("Modelo YOLO carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo YOLO: {e}")
            raise
    
    def detect(self, frame):
        """
        Detecta veículos em um frame.
        
        Args:
            frame: Imagem em formato numpy array (BGR)
            
        Returns:
            Uma lista de detecções, cada uma contendo:
            [classe, confiança, [x, y, largura, altura]]
        """
        height, width, _ = frame.shape
        
        # Preparar o blob para entrada na rede
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Extrair informações
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filtrar por classes de veículos (car=2, truck=7, bus=5, motorcycle=3 no COCO)
                vehicle_classes = [2, 3, 5, 7]
                if confidence > self.confidence_threshold and class_id in vehicle_classes:
                    # Coordenadas do centro, largura e altura
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Coordenadas do retângulo
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Aplicar non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Preparar resultados
        results = []
        for i in indices:
            # Corrigindo para OpenCV 4.x (índices diferentes)
            if isinstance(i, tuple) or isinstance(i, list):
                i = i[0]
                
            box = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            
            class_name = self.classes[class_id]
            results.append([class_name, confidence, box])
        
        return results