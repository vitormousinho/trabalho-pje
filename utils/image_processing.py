#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funções para processamento de imagens capturadas pelas câmeras.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_frame(frame, roi=None):
    """
    Pré-processa um frame para melhorar a detecção.
    
    Args:
        frame: Frame em formato numpy array (BGR)
        roi: Região de interesse [x, y, largura, altura]
        
    Returns:
        Frame pré-processado
    """
    # Verificar se o frame é válido
    if frame is None or frame.size == 0:
        logger.warning("Frame inválido recebido")
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Aplicar ROI se fornecida
    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
    
    # Redimensionar para melhorar desempenho
    frame = cv2.resize(frame, (640, 480))
    
    # Reduzir ruído
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    return frame

def draw_detections(frame, detections, colors=None):
    """
    Desenha as detecções em um frame.
    
    Args:
        frame: Frame em formato numpy array (BGR)
        detections: Lista de detecções [classe, confiança, [x, y, largura, altura]]
        colors: Dicionário de cores para cada classe
        
    Returns:
        Frame com as detecções desenhadas
    """
    # Definir cores padrão se não fornecidas
    if colors is None:
        colors = {
            "car": (0, 255, 0),       # Verde
            "truck": (0, 0, 255),     # Vermelho
            "bus": (255, 0, 0),       # Azul
            "motorcycle": (255, 255, 0) }
    
    # Criar uma cópia do frame para não modificar o original
    result = frame.copy()
    
    # Desenhar cada detecção
    for detection in detections:
        class_name, confidence, box = detection
        x, y, w, h = box
        
        # Obter cor para a classe
        color = colors.get(class_name, (255, 255, 255))  # Branco como padrão
        
        # Desenhar retângulo
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Desenhar rótulo
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(result, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Desenhar contagem de veículos
    count = len(detections)
    cv2.putText(result, f"Veículos: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return result

def create_mask(frame, lower_hsv, upper_hsv):
    """
    Cria uma máscara baseada em valores HSV para filtrar cores específicas.
    
    Args:
        frame: Frame em formato numpy array (BGR)
        lower_hsv: Limites inferiores dos valores HSV [H, S, V]
        upper_hsv: Limites superiores dos valores HSV [H, S, V]
        
    Returns:
        Máscara binária
    """
    # Converter para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Criar máscara
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    
    # Operações morfológicas para remover ruído
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def extract_background(frames, method='MOG2'):
    """
    Extrai o fundo a partir de uma sequência de frames.
    
    Args:
        frames: Lista de frames
        method: Método a ser usado ('MOG2' ou 'KNN')
        
    Returns:
        Modelo de fundo treinado
    """
    if method == 'MOG2':
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    else:
        bg_subtractor = cv2.createBackgroundSubtractorKNN()
    
    # Treinar com os frames fornecidos
    for frame in frames:
        bg_subtractor.apply(frame)
    
    return bg_subtractor

def detect_motion(frame, bg_subtractor, threshold=500):
    """
    Detecta movimento em um frame usando um modelo de fundo.
    
    Args:
        frame: Frame atual
        bg_subtractor: Modelo de fundo treinado
        threshold: Área mínima para considerar como movimento
        
    Returns:
        (foreground_mask, motion_detected)
    """
    # Aplicar subtração de fundo
    fg_mask = bg_subtractor.apply(frame)
    
    # Limpar a máscara
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Verificar se há movimento significativo
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > threshold:
            motion_detected = True
            break
    
    return fg_mask, motion_detected