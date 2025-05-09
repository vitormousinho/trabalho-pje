#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashboard simplificado para visualização do sistema de monitoramento de trânsito.
"""

import os
import sys
import cv2
import numpy as np
import time
import logging

# Adicionar o diretório pai ao path para importar os módulos do projeto
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.settings import Settings
from controllers.camera_controller import CameraController
from models.yolo_model import YOLOModel
from utils.image_processing import draw_detections
from utils.traffic_analysis import TrafficAnalyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDashboard:
    """Interface simples para monitoramento."""
    
    def __init__(self):
        """Inicializa o dashboard."""
        self.settings = Settings()
        self.camera_controller = CameraController(self.settings.camera_config)
        self.model = YOLOModel(self.settings.model_path, self.settings.confidence_threshold)
        self.traffic_analyzer = TrafficAnalyzer(self.settings.congestion_threshold)
        self.running = True
        
        logger.info("Dashboard iniciado")
    
    def run(self):
        """Executa o loop principal do dashboard."""
        try:
            while self.running:
                # Capturar frames
                frames = self.camera_controller.capture_frames()
                
                # Detectar veículos e mostrar estatísticas
                for camera_id, frame in frames.items():
                    # Detectar veículos
                    detections = self.model.detect(frame)
                    
                    # Processar o frame com as detecções
                    processed_frame = draw_detections(frame, detections)
                    
                    # Mostrar informações
                    cv2.putText(
                        processed_frame, 
                        f"Camera: {camera_id} | Veiculos: {len(detections)}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 255), 
                        2
                    )
                    
                    # Mostrar o frame
                    cv2.imshow(f"Camera: {camera_id}", processed_frame)
                
                # Analisar o tráfego
                traffic_state = self.traffic_analyzer.analyze({
                    camera_id: self.model.detect(frame) 
                    for camera_id, frame in frames.items()
                })
                
                # Tomar decisão
                decision = self.traffic_analyzer.make_decision(traffic_state)
                logger.info(f"Decisão: {decision}")
                
                # Tratar teclas pressionadas
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.running = False
                
                # Aguardar um pouco
                time.sleep(0.03)
                
        except KeyboardInterrupt:
            logger.info("Encerrando pelo usuário...")
        finally:
            # Liberar recursos
            self.camera_controller.release()
            cv2.destroyAllWindows()
            logger.info("Dashboard encerrado")

if __name__ == "__main__":
    dashboard = SimpleDashboard()
    dashboard.run()