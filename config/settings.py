#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configurações globais do sistema de monitoramento de trânsito.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

class Settings:
    """Classe para gerenciar as configurações do sistema."""
    
    def __init__(self, config_file="config/camera_config.json"):
        """Inicializa as configurações do sistema."""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Configurações do modelo YOLO
        self.model_path = os.path.join(self.base_dir, "models", "yolo_weights")
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Carregar configurações das câmeras do arquivo JSON
        self.camera_config_path = os.path.join(self.base_dir, config_file)
        self.camera_config = self._load_camera_config()
        
        # Configurações do controlador de semáforos
        self.traffic_light_config = {
            "default_green_time": 30,  # segundos
            "min_green_time": 10,      # segundos
            "max_green_time": 90,      # segundos
            "yellow_time": 3           # segundos
        }
        
        # Intervalo entre processamentos de frames
        self.processing_interval = 1.0  # segundos
        
        # Tipos de objetos a detectar
        self.classes_to_detect = ['car', 'truck', 'bus', 'motorcycle']
        
        # Limiar para considerar congestionamento
        self.congestion_threshold = 10  # veículos
    
    def _load_camera_config(self):
        """Carrega a configuração das câmeras do arquivo JSON."""
        try:
            with open(self.camera_config_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Erro ao carregar configuração das câmeras: {e}")
            # Configuração padrão caso o arquivo não seja encontrado
            return {
                "cameras": {
                    "north": {
                        "source": 0,  # Câmera local (índice 0)
                        "position": "north",
                        "roi": [0, 0, 640, 480]  # Região de interesse (x, y, width, height)
                    }
                }
            }