#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo para análise de tráfego e tomada de decisão.
"""

import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class TrafficAnalyzer:
    """Classe para analisar o tráfego e tomar decisões sobre os semáforos."""
    
    def __init__(self, congestion_threshold=10):
        """
        Inicializa o analisador de tráfego.
        
        Args:
            congestion_threshold: Número de veículos para considerar congestionamento
        """
        self.congestion_threshold = congestion_threshold
        
        # Para manter o histórico de contagens
        self.vehicle_counts = defaultdict(list)
        self.max_history = 10  # Manter histórico das últimas 10 contagens
    
    def analyze(self, detections):
        """
        Analisa as detecções para determinar o estado do tráfego.
        
        Args:
            detections: Dicionário com as detecções para cada câmera
                       {camera_id: [ [class_name, confidence, box], ... ]}
                       
        Returns:
            Dicionário com o estado do tráfego para cada direção
        """
        traffic_state = {}
        
        # Contar veículos em cada direção
        for camera_id, camera_detections in detections.items():
            # A posição da câmera (norte, sul, leste, oeste) é usada como direção
            direction = camera_id  # Assumindo que camera_id é a direção
            
            # Contar os veículos detectados
            vehicle_count = len(camera_detections)
            
            # Atualizar histórico
            self.vehicle_counts[direction].append(vehicle_count)
            if len(self.vehicle_counts[direction]) > self.max_history:
                self.vehicle_counts[direction].pop(0)
            
            # Calcular média móvel para suavizar flutuações
            avg_count = np.mean(self.vehicle_counts[direction])
            
            # Determinar nível de congestionamento
            congestion_level = avg_count / self.congestion_threshold
            congestion_level = min(congestion_level, 1.0)  # Normalizar entre 0 e 1
            
            # Armazenar estado
            traffic_state[direction] = {
                "vehicle_count": vehicle_count,
                "avg_count": avg_count,
                "congestion_level": congestion_level
            }
        
        return traffic_state
    
    def make_decision(self, traffic_state):
        """
        Toma decisão sobre qual direção deve ter semáforo verde e por quanto tempo.
        
        Args:
            traffic_state: Dicionário com o estado do tráfego para cada direção
            
        Returns:
            Decisão sobre qual direção deve ter o semáforo verde e por quanto tempo
        """
        # Encontrar a direção mais congestionada
        max_congestion = -1
        most_congested_direction = None
        
        for direction, state in traffic_state.items():
            congestion_level = state["congestion_level"]
            
            if congestion_level > max_congestion:
                max_congestion = congestion_level
                most_congested_direction = direction
        
        # Se não houver dados ou congestionamento, manter ciclo normal
        if most_congested_direction is None:
            return {"direction": "north", "green_time": 30}  # Direção padrão
        
        # Calcular tempo verde proporcional ao nível de congestionamento
        # Entre 20 e 60 segundos
        green_time = 20 + int(max_congestion * 40)
        
        return {
            "direction": most_congested_direction,
            "green_time": green_time
        }